# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TLE multicast tile-ready prefill, implemented entirely in this file.

Owns registration, signaling, projection, postprocessing and runtime lifetime.
Requires 128 resident CTAs and complete M128 tiles. No other scheme is imported.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flaggems_vllm.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    import triton.experimental.tle.language as tle
else:
    tle = None

__all__ = ["megapreattention", "clear_megapreattention_cache"]


@dataclass(frozen=True)
class _Config:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    mrope_section: tuple[int, int, int]
    eps: float
    hidden_size: int

    @property
    def q_size(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_size(self):
        return self.num_kv_heads * self.head_dim

    @property
    def packed_size(self):
        return 2 * (self.q_size + self.kv_size)


@dataclass
class _Inputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    positions: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    rope: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor
    group: object
    config: _Config
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None


@triton.jit
def _initialize_metadata(flags, counters, WORDS: tl.constexpr):
    i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(flags + i, 0, i < WORDS)
    tl.store(counters + i, 0, i < 128)


@triton.jit
def _initialize_peer_table(output, ADDRESSES: tl.constexpr):
    for i in tl.static_range(len(ADDRESSES)):
        tl.store(output + i, ADDRESSES[i])


@triton.jit
def _rendezvous(
    local, peers, pid, rank, target, FLAGS_WORDS: tl.constexpr, WORLD: tl.constexpr
):
    ranks = tl.arange(0, WORLD)
    remote = tl.load(peers + ranks).to(tl.pointer_type(tl.uint64))
    tl.atomic_xchg(
        remote + FLAGS_WORDS + pid * WORLD + rank, target, sem="release", scope="sys"
    )
    pending = tl.full((), 1, tl.int32)
    while pending != 0:
        # Volatile PTX loads are relaxed system-scope reads. Pair the final
        # observation with an acquire fence instead of contended atomic RMWs.
        seen = tl.load(
            local.to(tl.pointer_type(tl.uint64)) + FLAGS_WORDS + pid * WORLD + ranks,
            volatile=True,
        )
        pending = tl.sum((seen < target).to(tl.int32), 0)
    tl.inline_asm_elementwise(
        "fence.acq_rel.sys; mov.u32 $0,0;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    # Propagate the acquire performed by signal-reading lanes to the CTA.
    tl.debug_barrier()


class _MappedCommunicator:
    def __init__(self, group, tokens, producers, hidden_size=2048):
        from torch._C._distributed_c10d import _SymmetricMemory

        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.count = tokens * hidden_size
        self.max_programs = 128
        # Handshake slots belong to producer CTAs; ready slots belong to row
        # tiles. A persistent consumer processes multiple tiles, so their count
        # is independent of the number of concurrently resident CTAs.
        self.handshake_words = producers * self.world
        # Reserve per-row capacity; grouped publication uses only a prefix.
        self.ready_words = triton.cdiv(tokens, 128)
        self.flag_words = max(
            self.max_programs * self.world, self.handshake_words + self.ready_words
        )
        data_bytes = self.count * 2
        size = data_bytes + self.flag_words * 8
        self.allocation = _SymmetricMemory.empty_strided_p2p(
            (size,),
            (1,),
            torch.uint8,
            torch.device("cuda", torch.cuda.current_device()),
            group.group_name,
        )
        self.mapping = _SymmetricMemory.rendezvous(self.allocation, group.group_name)
        self.buffer = self.mapping.get_buffer(
            self.rank, (tokens, hidden_size), torch.bfloat16
        )
        self.flags = self.mapping.get_buffer(
            self.rank,
            (self.flag_words,),
            torch.uint64,
            storage_offset=data_bytes // 8,
        )
        self.counters = torch.empty(self.max_programs, device="cuda", dtype=torch.int64)
        _initialize_metadata[(triton.cdiv(self.flag_words, 1024),)](
            self.flags, self.counters, self.flag_words, num_warps=4
        )
        self.peers = torch.empty(self.world, device="cuda", dtype=torch.uint64)
        _initialize_peer_table[(1,)](
            self.peers, tuple(self.mapping.buffer_ptrs), num_warps=1
        )
        assert all(p % 16 == 0 for p in self.mapping.buffer_ptrs)
        self.mc_address = self.mapping.multicast_ptr or 0
        self.probe = {
            "backend": "PyTorch registration only; TLE data + synchronization",
            "multicast": self.mc_address,
            "data_ptr": self.buffer.data_ptr(),
            "signal_words": self.flag_words,
            "handshake_words": self.handshake_words,
            "ready_words": self.ready_words,
        }
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def close(self):
        torch.cuda.synchronize()
        dist.barrier(group=self.group)
        self.buffer = None
        self.flags = None
        self.mapping = None
        self.allocation = None


@triton.jit
def _wgmma_slot_phase(k_iter, num_slots: tl.constexpr):
    return k_iter % num_slots, k_iter // num_slots


@triton.jit
def _store_head(
    acc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    rope,
    positions,
    cos,
    sin,
    cache,
    slots,
    row_start,
    pid_m,
    pid_n,
    LOCAL_T: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    CACHE_BLOCK_SIZE: tl.constexpr,
    READY_ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token = row_start + row
    valid = row < LOCAL_T
    cols = tl.arange(0, 256)
    raw = acc.to(tl.bfloat16)
    # Preserve the materialized BF16 projection and norm rounding boundaries.
    if pid_n < 2 * Q_HEADS and pid_n % 2 == 1:
        head = pid_n // 2
        tl.store(
            gate + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
            raw,
            mask=valid[:, None],
        )
    elif pid_n >= 2 * Q_HEADS + KV_HEADS:
        head = pid_n - 2 * Q_HEADS - KV_HEADS if KV_HEADS > 1 else 0
        tl.store(
            value + token[:, None] * (KV_HEADS * 256) + head * 256 + cols[None, :],
            raw,
            mask=valid[:, None],
        )
        slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(tl.int64)
        target = (
            cache
            + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
            + head * CACHE_HEAD
            + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
        )
        tl.store(
            target + (256 + cols[None, :]) * CACHE_DIM,
            raw,
            mask=(valid & (slot >= 0))[:, None],
        )
    else:
        is_key = pid_n >= 2 * Q_HEADS if KV_HEADS > 1 else pid_n == 2 * Q_HEADS
        head = pid_n // 2
        weight = tl.load(tl.where(is_key, knorm, qnorm) + cols).to(tl.float32)
        x = raw.to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x, 1) / 256.0 + 1.0e-6)
        normalized = (
            (x * scale[:, None] * (weight[None, :] + 1.0))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        frequency = tl.arange(0, 32)
        if READY_ROPE:
            co = tl.load(
                cos + token[:, None] * 32 + frequency[None, :],
                mask=valid[:, None],
                other=1,
            ).to(tl.float32)
            si = tl.load(
                sin + token[:, None] * 32 + frequency[None, :],
                mask=valid[:, None],
                other=0,
            ).to(tl.float32)
        else:
            axis = tl.where(
                frequency % 3 == 1,
                1,
                tl.where((frequency % 3 == 2) & (frequency <= 30), 2, 0),
            )
            pos = tl.load(
                positions + axis[None, :] * POS_AXIS + token[:, None] * POS_TOKEN,
                mask=valid[:, None],
                other=0,
            ).to(tl.int64)
            co = tl.load(
                rope + pos * ROPE_STRIDE + frequency[None, :],
                mask=valid[:, None],
                other=1,
            ).to(tl.float32)
            si = tl.load(
                rope + pos * ROPE_STRIDE + 32 + frequency[None, :],
                mask=valid[:, None],
                other=0,
            ).to(tl.float32)
        # Only the first 64 dimensions rotate; pass the other 192 through.
        # Static splits avoid a full-width gather and unused rotary arithmetic.
        low128, high128 = tl.split(normalized.reshape(BLOCK_M, 2, 128).permute(0, 2, 1))
        low64, high64 = tl.split(low128.reshape(BLOCK_M, 2, 64).permute(0, 2, 1))
        first, second = tl.split(low64.reshape(BLOCK_M, 2, 32).permute(0, 2, 1))
        first_out = first * co - second * si
        second_out = second * co + first * si
        rotated64 = tl.join(first_out, second_out).permute(0, 2, 1).reshape(BLOCK_M, 64)
        result128 = tl.join(rotated64, high64).permute(0, 2, 1).reshape(BLOCK_M, 128)
        result = tl.join(result128, high128).permute(0, 2, 1).reshape(BLOCK_M, 256)
        if is_key:
            kv_head = pid_n - 2 * Q_HEADS if KV_HEADS > 1 else 0
            tl.store(
                key + token[:, None] * (KV_HEADS * 256) + kv_head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )
            slot = tl.load(slots + token * SLOT_STRIDE, mask=valid, other=-1).to(
                tl.int64
            )
            target = (
                cache
                + (slot // CACHE_BLOCK_SIZE)[:, None] * CACHE_BLOCK
                + kv_head * CACHE_HEAD
                + (slot % CACHE_BLOCK_SIZE)[:, None] * CACHE_TOKEN
            )
            tl.store(
                target + cols[None, :] * CACHE_DIM,
                result,
                mask=(valid & (slot >= 0))[:, None],
            )
        else:
            tl.store(
                query + token[:, None] * (Q_HEADS * 256) + head * 256 + cols[None, :],
                result,
                mask=valid[:, None],
            )


@triton.jit
def _fused_projection(
    a_desc,
    b_desc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    rope,
    positions,
    cos,
    sin,
    cache,
    slots,
    row_start,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    M: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HIDDEN: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    READY_ROPE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    ready,
    target,
    WORLD: tl.constexpr = 1,
    COMM: tl.constexpr = 0,
    READY_GROUP: tl.constexpr = 1,
):
    num_n: tl.constexpr = 2 * (Q_HEADS + KV_HEADS)
    k_tiles: tl.constexpr = HIDDEN // BK
    a = tle.gpu.alloc(
        [NUM_SLOTS, BM, BK], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    b = tle.gpu.alloc(
        [NUM_SLOTS, 256, BK], dtype=tl.bfloat16, layout=None, scope=tle.gpu.smem
    )
    af = tle.gpu.alloc_barriers(NUM_SLOTS, expect_bytes=BM * BK * 2)
    bf = tle.gpu.alloc_barriers(NUM_SLOTS, expect_bytes=256 * BK * 2)
    seq = 0
    for tile_id in tl.range(
        tl.program_id(0) - COMM, tl.cdiv(M, BM) * num_n, NUM_PROGRAMS
    ):
        if HIDDEN == 5120 and M == 32768 and (WORLD == 2 or WORLD == 4):
            # Reuse each head pair across four row tiles while alternating
            # norm/RoPE and copy work. K/V pairs retain the packed K-all/V-all order.
            group_id = tile_id // (4 * num_n)
            pair = (tile_id % (4 * num_n)) // 8
            pm = group_id * 4 + (tile_id // 2) % 4
            kind = tile_id % 2
            pn = tl.where(
                pair < Q_HEADS,
                2 * pair + kind,
                2 * Q_HEADS + pair - Q_HEADS + kind * KV_HEADS,
            )
        else:
            pm, pn = tile_id // num_n, tile_id % num_n
        pm = (pm % WORLD) * (M // WORLD // BM) + pm // WORLD
        seen = tl.full((), 0, tl.uint64)
        # The 32K policy uses an acquire read of the producers' acq_rel
        # arrival chain. It need not release the preceding output stores.
        if READY_GROUP > 1:
            while seen < target:
                seen = tl.inline_asm_elementwise(
                    "ld.acquire.sys.global.u64 $0, [$1];",
                    constraints="=l,l",
                    args=[ready + pm // READY_GROUP],
                    dtype=tl.uint64,
                    is_pure=False,
                    pack=1,
                )
        else:
            while seen < target:
                seen = tl.load(ready + pm // READY_GROUP, volatile=True)
            tl.inline_asm_elementwise(
                "fence.acq_rel.sys; mov.u32 $0,0;",
                constraints="=r",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
        tl.debug_barrier()
        # Producers fence multicast aliases before their acq_rel arrivals.
        # The acquire above completes that alias-ordering path; TMA still
        # needs the generic-to-async proxy fence. Keep the old policy elsewhere.
        if READY_GROUP > 1:
            fence: tl.constexpr = "fence.proxy.async.global; mov.u32 $0,0;"
        else:
            fence: tl.constexpr = (
                "fence.proxy.alias; fence.proxy.async.global; mov.u32 $0,0;"
            )
        tl.inline_asm_elementwise(
            fence,
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        for initial_slot in tl.static_range(NUM_SLOTS):
            tle.gpu.copy(
                a_desc,
                a.slot(initial_slot),
                [BM, BK],
                [pm * BM, initial_slot * BK],
                barrier=af[initial_slot],
            )
            tle.gpu.copy(
                b_desc,
                b.slot(initial_slot),
                [256, BK],
                [pn * 256, initial_slot * BK],
                barrier=bf[initial_slot],
            )
        acc = tl.zeros((BM, 256), tl.float32)
        first_phase = seq * k_tiles // NUM_SLOTS
        tle.gpu.barrier_wait(af[0], phaseIdx=first_phase)
        tle.gpu.barrier_wait(bf[0], phaseIdx=first_phase)
        acc = tle.gpu.wgmma(a.slot(0), b.slot(0), acc, trans_b=True)
        for k in tl.range(1, k_tiles):
            slot, phase = _wgmma_slot_phase(seq * k_tiles + k, NUM_SLOTS)
            tle.gpu.barrier_wait(af[slot], phaseIdx=phase)
            tle.gpu.barrier_wait(bf[slot], phaseIdx=phase)
            acc = tle.gpu.wgmma(a.slot(slot), b.slot(slot), acc, trans_b=True)
            acc = tle.gpu.wgmma_wait(1, acc)
            if k - 1 + NUM_SLOTS < k_tiles:
                previous = (k - 1) % NUM_SLOTS
                next_k = (k - 1 + NUM_SLOTS) * BK
                tle.gpu.copy(
                    a_desc,
                    a.slot(previous),
                    [BM, BK],
                    [pm * BM, next_k],
                    barrier=af[previous],
                )
                tle.gpu.copy(
                    b_desc,
                    b.slot(previous),
                    [256, BK],
                    [pn * 256, next_k],
                    barrier=bf[previous],
                )
        acc = tle.gpu.wgmma_wait(0, acc)
        _store_head(
            acc,
            query,
            key,
            value,
            gate,
            qnorm,
            knorm,
            rope,
            positions,
            cos,
            sin,
            cache,
            slots,
            row_start,
            pm,
            pn,
            M,
            Q_HEADS,
            KV_HEADS,
            POS_AXIS,
            POS_TOKEN,
            ROPE_STRIDE,
            SLOT_STRIDE,
            CACHE_BLOCK,
            CACHE_HEAD,
            CACHE_TOKEN,
            CACHE_DIM,
            CACHE_BLOCK_SIZE,
            READY_ROPE,
            BM,
        )
        seq += 1


@triton.jit
def _persistent(
    source,
    local,
    peers,
    counters,
    multicast,
    rank,
    a_desc,
    b_desc,
    query,
    key,
    value,
    gate,
    qnorm,
    knorm,
    rope,
    positions,
    cos,
    sin,
    cache,
    slots,
    POS_AXIS,
    POS_TOKEN,
    ROPE_STRIDE,
    SLOT_STRIDE,
    CACHE_BLOCK,
    CACHE_HEAD,
    CACHE_TOKEN,
    CACHE_DIM,
    M: tl.constexpr,
    WORLD: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HIDDEN: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
    READY_ROPE: tl.constexpr,
    COMM: tl.constexpr,
    COMPUTE: tl.constexpr,
    BLOCK: tl.constexpr,
    BM: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    epoch = tl.atomic_add(counters + pid, 1, sem="relaxed").to(tl.uint64)
    flags: tl.constexpr = M * HIDDEN // 4
    ready = local.to(tl.pointer_type(tl.uint64)) + flags + COMM * WORLD
    if pid < COMM:
        # Previous invocation's local compute has completed before this launch;
        # all producers rendezvous before overwriting any registered input.
        _rendezvous(local, peers, pid, rank, epoch + 1, flags, WORLD)
        ranks = tl.arange(0, WORLD)
        destinations = tl.load(peers + ranks).to(tl.pointer_type(tl.uint64))
        for row_chunk in range(M // WORLD // BM // GROUP):
            row_tile = row_chunk * GROUP
            pm = rank * (M // WORLD // BM) + row_tile
            src = source.to(tl.pointer_type(tl.uint32))
            # Stage input loads asynchronously only for the validated 32K policy.
            for block in tl.range(
                pid,
                tl.cdiv(BM * GROUP * HIDDEN // 8, BLOCK),
                COMM,
                num_stages=3 if GROUP > 1 else 1,
            ):
                group = block * BLOCK + tl.arange(0, BLOCK)
                valid = group < BM * GROUP * HIDDEN // 8
                offset = group * 4
                values = tl.load(
                    src
                    + row_tile * (BM * HIDDEN // 2)
                    + offset[:, None]
                    + tl.arange(0, 4)[None, :],
                    valid[:, None],
                    0,
                )
                even, odd = tl.split(values.reshape(BLOCK, 2, 2))
                x0, x2 = tl.split(even)
                x1, x3 = tl.split(odd)
                address = (
                    multicast + (pm * (BM * HIDDEN // 2) + offset).to(tl.uint64) * 4
                )
                tl.inline_asm_elementwise(
                    "{.reg .pred p; setp.ne.u32 p,$6,0; "
                    "@p multimem.st.relaxed.sys.global.v4.f32 [$1],{$2,$3,$4,$5}; "
                    "mov.u32 $0,0;}",
                    constraints="=r,l,r,r,r,r,r",
                    args=[address, x0, x1, x2, x3, valid.to(tl.int32)],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )
            tl.debug_barrier()
            tl.inline_asm_elementwise(
                "fence.proxy.alias; membar.sys; mov.u32 $0,0;",
                constraints="=r",
                args=[],
                dtype=tl.int32,
                is_pure=False,
                pack=1,
            )
            # All row tiles in this group become ready at the same time.
            tl.atomic_add(
                destinations + flags + COMM * WORLD + pm // GROUP,
                1,
                sem="acq_rel",
                scope="sys",
            )
    else:
        _fused_projection(
            a_desc,
            b_desc,
            query,
            key,
            value,
            gate,
            qnorm,
            knorm,
            rope,
            positions,
            cos,
            sin,
            cache,
            slots,
            0,
            POS_AXIS,
            POS_TOKEN,
            ROPE_STRIDE,
            SLOT_STRIDE,
            CACHE_BLOCK,
            CACHE_HEAD,
            CACHE_TOKEN,
            CACHE_DIM,
            M,
            Q_HEADS,
            KV_HEADS,
            HIDDEN,
            CACHE_BLOCK_SIZE,
            READY_ROPE,
            BM,
            64,
            4,
            COMPUTE,
            ready,
            (epoch + 1) * COMM,
            WORLD,
            COMM,
            GROUP,
        )


def _select_device_policy(tp, tokens, sm_count, hidden_size=2048):
    """Validate the persistent scheduler before any collective allocation."""
    # Balance multicast progress against compute work for each 32K TP layout.
    # Keep the compute stride odd so Q/K norm and gate/V copy tiles alternate;
    # an even stride permanently assigns expensive tiles to half the CTAs.
    producers = (13 if tp == 2 else 21) if tp in (2, 4) and tokens == 32768 else 8
    if hidden_size == 5120 and tokens == 32768 and tp in (2, 4):
        producers = 13 if tp == 2 else 9
    block = 256
    if tp not in (2, 4, 8) or tokens < 1024 or tokens % (tp * 128):
        raise NotImplementedError(
            "TLE prefill requires TP2/TP4/TP8, T>=1024 and complete 128-row tiles per rank"
        )
    if sm_count < 128:
        raise NotImplementedError(
            "persistent TLE prefill requires at least 128 available SMs"
        )
    return producers, block


class _DevicePrefill:
    """Fixed-shape persistent window. Create during warmup; close after graph reset.

    Calls for one instance must be serialized on one stream. All ranks must use
    the same transport and call order. The 128 CTAs must be concurrently resident.
    """

    def __init__(self, group, tokens, config):
        self.group = group
        self.tokens = tokens
        self.config = config
        self.comm_programs, self.block = _select_device_policy(
            group.size(),
            tokens,
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count,
            config.hidden_size,
        )
        self.compute_programs = 128 - self.comm_programs
        self.communication = _MappedCommunicator(
            group, tokens, self.comm_programs, config.hidden_size
        )
        if not self.communication.mc_address:
            self.communication.close()
            raise RuntimeError(
                "multicast mapping unavailable; this implementation requires multicast support"
            )

    def __call__(self, x):
        c, t, rt = x.config, self.tokens, self.communication
        if x.group is not self.group or x.positions.shape[1] != t or c != self.config:
            raise NotImplementedError(
                "DevicePrefill requires the original group, token count and model dimensions"
            )
        if (
            x.hidden.shape != (t // rt.world, c.hidden_size)
            or not x.hidden.is_contiguous()
        ):
            raise NotImplementedError(
                "DevicePrefill requires contiguous rank-local BF16 hidden"
            )
        if (
            x.hidden.dtype != torch.bfloat16
            or c.num_kv_heads not in (1, 2)
            or c.head_dim != 256
        ):
            raise NotImplementedError(
                "DevicePrefill requires BF16, head_dim=256 and one or two local KV heads"
            )
        q = x.hidden.new_empty((t, c.q_size))
        k = x.hidden.new_empty((t, c.kv_size))
        v, g = torch.empty_like(k), torch.empty_like(q)
        kernel = _persistent[(128,)](
            x.hidden,
            rt.buffer,
            rt.peers,
            rt.counters,
            rt.mc_address,
            rt.rank,
            TensorDescriptor.from_tensor(rt.buffer, [128, 64]),
            TensorDescriptor.from_tensor(x.weight, [256, 64]),
            q,
            k,
            v,
            g,
            x.q_norm,
            x.k_norm,
            x.rope,
            x.positions,
            x.cos,
            x.sin,
            x.cache,
            x.slots,
            *x.positions.stride(),
            x.rope.stride(0),
            x.slots.stride(0),
            *x.cache.stride(),
            M=t,
            WORLD=rt.world,
            Q_HEADS=c.num_q_heads,
            KV_HEADS=c.num_kv_heads,
            HIDDEN=c.hidden_size,
            CACHE_BLOCK_SIZE=x.cache.shape[2],
            READY_ROPE=False,
            COMM=self.comm_programs,
            COMPUTE=self.compute_programs,
            BLOCK=self.block,
            BM=128,
            GROUP=(
                (2 if c.hidden_size == 5120 else 4)
                if t == 32768 and rt.world in (2, 4)
                else 1
            ),
            num_warps=8,
        )
        self.resources = {
            "registers": kernel.n_regs,
            "spills": kernel.n_spills,
            "shared_bytes": kernel.metadata.shared,
        }
        return q, k, v, g.view(t, c.num_q_heads, c.head_dim)

    def close(self):
        self.communication.close()


def _validate_input(inputs):
    if inputs.positions.ndim != 2 or inputs.positions.shape[0] != 3:
        raise NotImplementedError("positions must have three axes")
    if inputs.hidden.device.type != "cuda":
        raise NotImplementedError("only CUDA tensors are supported")
    c, t, tp = inputs.config, inputs.positions.shape[1], inputs.group.size()
    if tp not in (2, 4, 8) or (
        c.hidden_size,
        c.num_q_heads * tp,
        c.num_kv_heads,
    ) not in ((2048, 16, 1), (5120, 24, max(1, 4 // tp))):
        raise NotImplementedError(
            "only megapreattention TP2/TP4/TP8 head layouts are supported"
        )
    if (c.head_dim, c.rotary_dim, c.mrope_section, c.eps) != (
        256,
        64,
        (11, 11, 10),
        1e-6,
    ):
        raise NotImplementedError(
            "unsupported megapreattention norm/RoPE configuration"
        )
    if (
        inputs.hidden.shape != (t // tp, c.hidden_size)
        or not inputs.hidden.is_contiguous()
    ):
        raise NotImplementedError(
            "megapreattention requires contiguous [T/TP,H] hidden"
        )
    if (
        inputs.weight.shape != (c.packed_size, c.hidden_size)
        or not inputs.weight.is_contiguous()
    ):
        raise NotImplementedError("unsupported projection weight shape/stride")
    if (
        inputs.cache.ndim != 4
        or inputs.slots.shape != (t,)
        or inputs.cache.shape[1] != c.num_kv_heads
        or inputs.cache.shape[3] != 512
    ):
        raise NotImplementedError("unsupported slots/cache metadata")
    if any(
        w.shape != (256,) or not w.is_contiguous()
        for w in (inputs.q_norm, inputs.k_norm)
    ):
        raise NotImplementedError("norm weights must be contiguous length256")
    if (
        inputs.rope.ndim != 2
        or inputs.rope.shape[1] != 64
        or inputs.rope.stride(1) != 1
    ):
        raise NotImplementedError("unsupported RoPE cache shape/stride")
    if (
        inputs.positions.device != inputs.hidden.device
        or inputs.slots.device != inputs.hidden.device
    ):
        raise NotImplementedError("positions/slots must be on the input CUDA device")
    for value in (
        inputs.hidden,
        inputs.weight,
        inputs.q_norm,
        inputs.k_norm,
        inputs.rope,
        inputs.cache,
    ):
        if (
            value.dtype != torch.bfloat16
            or value.device != inputs.hidden.device
            or value.requires_grad
        ):
            raise NotImplementedError(
                "only same-device BF16 inference tensors are supported"
            )
    if inputs.positions.dtype != torch.int64 or inputs.slots.dtype != torch.int64:
        raise NotImplementedError("positions and slots must be int64")
    if torch.cuda.get_device_capability(inputs.hidden.device)[0] != 9:
        raise NotImplementedError("only Hopper SM90 is supported")
    _select_device_policy(
        tp,
        t,
        torch.cuda.get_device_properties(inputs.hidden.device).multi_processor_count,
        c.hidden_size,
    )


_windows = {}


def megapreattention(
    hidden,
    weight,
    positions,
    q_norm,
    k_norm,
    cos_sin_cache,
    kv_cache,
    slot_mapping,
    group,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    mrope_section,
    eps=1e-6,
):
    """AllGather, gated QKV projection, Gemma RMSNorm/MRoPE and KV-cache update.

    Model dimensions are global (before tensor parallel sharding). Currently
    supports (H,Q,KV)=(2048,16,2) or (5120,24,4), with D=256, rotary_dim=64,
    sections=(11,11,10), eps=1e-6, BF16, TP2/4/8 on SM90 GPUs with at least
    128 SMs. KV heads are replicated when TP exceeds the global KV count.
    Other dimensions
    raise NotImplementedError before collective allocation; there is no fallback.

    hidden is contiguous [T/TP, H]. weight is contiguous
    [2 * (local_q_heads + local_kv_heads) * D, H], packed as interleaved Q/gate
    per query head, then all K heads, then all V heads. positions is int64 [3,T]; cos_sin_cache is
    BF16 [max_position, rotary_dim]. Norm weights are [D] (Gemma uses 1+w).
    kv_cache is [blocks, local_kv_heads, block_size, 2*D], with K then V;
    valid slots must be unique and in bounds, and negative slots are ignored.
    Positions must index cos_sin_cache. No tensor-value scans are performed.

    Returns (Q, K, V, gate), BF16 shapes [T,local_q_heads*D],
    [T,local_kv_heads*D], [T,local_kv_heads*D],
    [T,local_q_heads,D]. gate is before activation. kv_cache is updated in-place.
    Inference only; no autograd. Inputs/outputs must not alias kv_cache.

    Warm up each shape before CUDA Graph capture. Calls must be serialized on
    one stream with identical rank order. Reset dependent graphs before calling
    clear_megapreattention_cache on every rank.

    No online autotune: producer CTA counts and ready-signal layouts must match
    on all ranks. Independent per-rank tuning can deadlock this protocol.
    The validated offline policy is kept in _select_device_policy.
    """
    if (
        hidden.ndim != 2
        or (hidden.shape[1], num_heads, num_kv_heads)
        not in ((2048, 16, 2), (5120, 24, 4))
        or (head_dim, rotary_dim, tuple(mrope_section), eps)
        != (256, 64, (11, 11, 10), 1e-6)
    ):
        raise NotImplementedError(
            "megapreattention supports H2048/Q16/KV2 or H5120/Q24/KV4 with D256/R64, "
            "mrope_section=(11,11,10), eps=1e-6 only"
        )
    if group.size() not in (2, 4, 8):
        raise NotImplementedError("megapreattention requires TP2/TP4/TP8")
    if tle is None:
        raise NotImplementedError("megapreattention requires FlagTree/TLE >= 3.6")
    config = _Config(
        num_heads // group.size(),
        max(1, num_kv_heads // group.size()),
        head_dim,
        rotary_dim,
        tuple(mrope_section),
        eps,
        hidden.shape[1],
    )
    inputs = _Inputs(
        hidden,
        weight,
        positions,
        q_norm,
        k_norm,
        cos_sin_cache,
        kv_cache,
        slot_mapping,
        group,
        config,
    )
    _validate_input(inputs)
    tokens = positions.shape[1]
    key = (id(group), tokens, hidden.device, config)
    if key not in _windows:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm up this shape before capture")
        _windows[key] = _DevicePrefill(group, tokens, config)
    return _windows[key](inputs)


def clear_megapreattention_cache():
    """Collective cleanup after synchronizing and releasing dependent graphs."""
    for window in _windows.values():
        window.close()
    _windows.clear()
