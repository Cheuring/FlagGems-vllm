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

"""Shared full-window reference and distributed test/benchmark adapter.

Default inputs are seeded synthetic tensors, without model downloads. An
explicit optional fixture path supports external checkpoint-based regression.
Only tests/benchmarks import this module; production never imports it.
"""

import hashlib
import inspect
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm

from .accuracy_utils import gems_assert_close


@dataclass(frozen=True)
class Config:
    num_q_heads: int
    num_kv_heads: int
    head_dim: int = 256
    rotary_dim: int = 64
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    eps: float = 1e-6
    global_kv_heads: int = 2

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
class Inputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    positions: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    rope: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor
    scale: torch.Tensor
    group: object
    config: Config


@dataclass
class Output:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gate: torch.Tensor  # [tokens, local_q_heads, head_dim], possibly strided.
    workspace: tuple = ()

    def tensors(self):
        return self.query, self.key, self.value, self.gate


def load_fixture(path=None, model_dims=(2048, 16, 2), head_dim=256):
    hidden_size, q_heads, kv_heads = model_dims
    if path is not None:
        fixture = torch.load(path, map_location="cpu", weights_only=True)
        for name, tensor in fixture["weights"].items():
            digest = hashlib.sha256(
                tensor.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            assert digest == fixture["metadata"]["weight_sha256"][name]
        expected = {
            "q_proj.weight": (2 * q_heads * head_dim, hidden_size),
            "k_proj.weight": (kv_heads * head_dim, hidden_size),
            "v_proj.weight": (kv_heads * head_dim, hidden_size),
            "q_norm.weight": (head_dim,),
            "k_norm.weight": (head_dim,),
        }
        for name, shape in expected.items():
            if tuple(fixture["weights"][name].shape) != shape:
                raise ValueError(
                    f"fixture {name} does not match model dimensions {model_dims}"
                )
        return fixture
    generator = torch.Generator().manual_seed(20261010)
    weights = {
        "q_proj.weight": torch.randn(
            2 * q_heads * head_dim,
            hidden_size,
            generator=generator,
            dtype=torch.bfloat16,
        )
        * 0.02,
        "k_proj.weight": torch.randn(
            kv_heads * head_dim, hidden_size, generator=generator, dtype=torch.bfloat16
        )
        * 0.02,
        "v_proj.weight": torch.randn(
            kv_heads * head_dim, hidden_size, generator=generator, dtype=torch.bfloat16
        )
        * 0.02,
        "q_norm.weight": torch.randn(
            head_dim, generator=generator, dtype=torch.bfloat16
        )
        * 0.1,
        "k_norm.weight": torch.randn(
            head_dim, generator=generator, dtype=torch.bfloat16
        )
        * 0.1,
    }
    return {
        "weights": weights,
        "metadata": {"input_origin": "seeded synthetic weights and hidden"},
    }


def run_candidate(inputs):
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        megapreattention,
    )

    c = inputs.config
    return Output(
        *megapreattention(
            inputs.hidden,
            inputs.weight,
            inputs.positions,
            inputs.q_norm,
            inputs.k_norm,
            inputs.rope,
            inputs.cache,
            inputs.slots,
            inputs.group,
            num_heads=c.num_q_heads * inputs.group.size(),
            num_kv_heads=c.global_kv_heads,
            head_dim=c.head_dim,
            rotary_dim=c.rotary_dim,
            mrope_section=c.mrope_section,
            eps=c.eps,
        )
    )


def make_inputs(
    tokens,
    tp,
    rank,
    fixture,
    *,
    edge=False,
    model_dims=(2048, 16, 2),
    head_dim=256,
    rotary_dim=64,
    mrope_section=(11, 11, 10),
    eps=1e-6,
):
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    hidden_size, q_heads, kv_heads = model_dims
    config = Config(
        q_heads // tp,
        max(1, kv_heads // tp),
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section),
        eps=eps,
        global_kv_heads=kv_heads,
    )
    weights = fixture["weights"]
    qwidth = q_heads * head_dim * 2 // tp
    kv_rank = rank // max(1, tp // kv_heads)
    kv_start = kv_rank * config.kv_size
    local = torch.cat(
        (
            weights["q_proj.weight"][rank * qwidth : (rank + 1) * qwidth],
            weights["k_proj.weight"][kv_start : kv_start + config.kv_size],
            weights["v_proj.weight"][kv_start : kv_start + config.kv_size],
        )
    ).to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(20261010 + rank + tokens)
    hidden = (
        torch.randn(
            (tokens // tp, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        * 0.05
    )
    position_base = torch.arange(tokens, device="cuda", dtype=torch.int64)
    positions = position_base.repeat(3, 1)
    slots = position_base.clone()
    if edge:
        storage = torch.empty((3, tokens + 7), device="cuda", dtype=torch.int64)
        positions = storage[:, :tokens]
        positions[0], positions[1], positions[2] = (
            position_base,
            position_base + 13,
            position_base + 29,
        )
        slot_storage = torch.empty(tokens * 2, device="cuda", dtype=torch.int64)
        slots = slot_storage[::2]
        slots.copy_(position_base * 2 + 15)
        slots[-1] = -1
    count = tokens * 2 + 128 if edge else tokens + 128
    blocks = (count + 127) // 128
    cache = torch.full(
        (blocks, config.num_kv_heads, 128, 2 * head_dim),
        0.375,
        device="cuda",
        dtype=torch.bfloat16,
    )
    if edge:
        cache = torch.full(
            (128, config.num_kv_heads, blocks, 2 * head_dim),
            0.375,
            device="cuda",
            dtype=torch.bfloat16,
        ).permute(2, 1, 0, 3)
    rope = MRotaryEmbedding(
        head_dim,
        rotary_dim,
        262144,
        1.0e7,
        True,
        torch.bfloat16,
        mrope_section=list(mrope_section),
        mrope_interleaved=True,
    )
    rope.cos_sin_cache = rope.cos_sin_cache.to("cuda")
    inputs = Inputs(
        hidden,
        local,
        positions,
        weights["q_norm.weight"].to("cuda"),
        weights["k_norm.weight"].to("cuda"),
        rope.cos_sin_cache,
        cache,
        slots,
        torch.ones(1, device="cuda"),
        dist.group.WORLD,
        config,
    )
    return inputs, rope


class NativeWindow:
    """Library AG/GEMM + original vLLM projection method + native cache op.

    The text baseline enables vLLM's own fast fused norm/RoPE/gate path.
    Multimodal checks use its original non-fused MRoPE path instead.
    """

    def __init__(self, inputs, rope, *, edge=False):
        from vllm import _custom_ops
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
        from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

        symm.enable_symm_mem_for_group(inputs.group.group_name)
        qnorm = SimpleNamespace(
            weight=inputs.q_norm, variance_epsilon=inputs.config.eps
        )
        knorm = SimpleNamespace(
            weight=inputs.k_norm, variance_epsilon=inputs.config.eps
        )
        qnorm_call = lambda x: GemmaRMSNorm.forward_native(qnorm, x)
        knorm_call = lambda x: GemmaRMSNorm.forward_native(knorm, x)
        # The projection method reads .weight even for its fused fast path.
        qnorm_call.weight, qnorm_call.variance_epsilon = (
            inputs.q_norm,
            inputs.config.eps,
        )
        knorm_call.weight, knorm_call.variance_epsilon = (
            inputs.k_norm,
            inputs.config.eps,
        )
        self.context = SimpleNamespace(
            use_fused_qk_norm_rope_gate=not edge,
            attn_output_gate=True,
            q_size=inputs.config.q_size,
            kv_size=inputs.config.kv_size,
            num_heads=inputs.config.num_q_heads,
            num_kv_heads=inputs.config.num_kv_heads,
            head_dim=inputs.config.head_dim,
            q_norm=qnorm_call,
            k_norm=knorm_call,
            rotary_emb=rope,
        )
        self.method, self.cache_op = (
            Qwen3NextAttention._project_qkv_gate,
            _custom_ops.reshape_and_cache_flash,
        )
        self.scale = inputs.scale
        self.group_name = inputs.group.group_name
        self.post = torch.compile(self._postprocess, fullgraph=True, dynamic=False)
        self.source_evidence = {
            "projection_method_sha256": hashlib.sha256(
                inspect.getsource(self.method).encode()
            ).hexdigest(),
            "projection_file": inspect.getfile(self.method),
            "text_fused_path": not edge,
            "cache_op": "vllm._custom_ops.reshape_and_cache_flash",
            "communication_gemm": "torch.ops.symm_mem.fused_all_gather_matmul.default",
            "compile": "fullgraph=True,dynamic=False; postprocessing only",
        }

    def _postprocess(self, qkv, positions, cache, slots):
        q, k, v, gate = self.method(self.context, qkv, positions)
        key_cache, value_cache = cache.transpose(1, 2).split(
            self.context.head_dim, dim=-1
        )
        self.cache_op(
            k.view(-1, self.context.num_kv_heads, self.context.head_dim),
            v.view(-1, self.context.num_kv_heads, self.context.head_dim),
            key_cache,
            value_cache,
            slots,
            "auto",
            self.scale,
            self.scale,
        )
        return (
            q,
            k,
            v,
            gate.view(q.shape[0], self.context.num_heads, self.context.head_dim),
        )

    def __call__(self, inputs):
        gathered, products = torch.ops.symm_mem.fused_all_gather_matmul.default(
            inputs.hidden, [inputs.weight.t()], 0, self.group_name
        )
        values = self.post(
            products[0], inputs.positions, inputs.cache, inputs.slots.contiguous()
        )
        return Output(*values, (gathered, products[0]))


def independent_reference(inputs):
    """Oracle starts at original shards; never reads candidate scratch/results."""
    c, t = inputs.config, inputs.positions.shape[1]
    hidden = inputs.hidden.new_empty((t, inputs.hidden.shape[1]))
    dist.all_gather_into_tensor(hidden, inputs.hidden, group=inputs.group)
    qkv = hidden @ inputs.weight.t()
    qgate, key, value = qkv.split((2 * c.q_size, c.kv_size, c.kv_size), dim=-1)
    query, gate = qgate.view(t, c.num_q_heads, 2 * c.head_dim).chunk(2, dim=-1)
    frequency = torch.arange(c.rotary_dim // 2, device="cuda")
    axis = torch.zeros_like(frequency)
    axis[1 : 3 * c.mrope_section[1] : 3] = 1
    axis[2 : 3 * c.mrope_section[2] : 3] = 2
    position = inputs.positions[axis].t()
    cos = inputs.rope[position, frequency].float().unsqueeze(1)
    sin = inputs.rope[position, frequency + c.rotary_dim // 2].float().unsqueeze(1)

    def norm_rope(x, weight):
        x = x.float()
        normalized = (
            (
                x
                * torch.rsqrt(x.square().mean(-1, keepdim=True) + c.eps)
                * (weight.float() + 1)
            )
            .to(torch.bfloat16)
            .float()
        )
        first, second = normalized[..., : c.rotary_dim].chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        return torch.cat((rotated, normalized[..., c.rotary_dim :]), dim=-1).to(
            torch.bfloat16
        )

    query = norm_rope(query, inputs.q_norm).reshape(t, c.q_size)
    key = norm_rope(key.view(t, c.num_kv_heads, c.head_dim), inputs.k_norm)
    valid = inputs.slots >= 0
    slots = inputs.slots[valid]
    block, offset = slots // inputs.cache.shape[2], slots % inputs.cache.shape[2]
    inputs.cache[block, :, offset, : c.head_dim] = key[valid]
    inputs.cache[block, :, offset, c.head_dim :] = value.view(
        t, c.num_kv_heads, c.head_dim
    )[valid]
    return Output(query, key.reshape(t, c.kv_size), value, gate)


def check_outputs(actual, expected, actual_cache, expected_cache):
    result = {}
    for name, a, b in zip(
        ("query", "key", "value", "gate"), actual.tensors(), expected.tensors()
    ):
        gems_assert_close(a, b, torch.bfloat16, atol=0.03)
        assert torch.isfinite(a).all()
        result[name] = {"max_abs": (a.float() - b.float()).abs().max().item()}
    gems_assert_close(actual_cache, expected_cache, torch.bfloat16, atol=0.03)
    result["cache"] = {
        "max_abs": (actual_cache.float() - expected_cache.float()).abs().max().item()
    }
    return result


def distribution(samples):
    ordered = sorted(samples)

    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)

    median = statistics.median(samples)
    span = (percentile(0.9) - percentile(0.1)) / median
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": median,
        "span": span,
        "stable": len(samples) >= 10 and span <= 0.05,
        "samples_ms": samples,
    }


def _worker(
    rank,
    tp,
    tokens,
    mode,
    iterations,
    warmup,
    edge,
    check_only,
    directory,
    fixture_path,
    model_dims,
    kernel_config,
):
    import torch
    import torch.distributed as dist
    from torch._dynamo.utils import counters as dynamo_counters
    from triton.runtime.jit import JITFunction

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=tp,
        init_method=f"file://{directory}/rdzv",
        device_id=torch.device("cuda", rank),
        timeout=timedelta(seconds=180),
    )
    from vllm.config import VllmConfig
    from vllm.config.vllm import set_current_vllm_config

    native_context = set_current_vllm_config(VllmConfig())
    native_context.__enter__()
    fixture = load_fixture(fixture_path, model_dims, kernel_config["head_dim"])
    source, rope = make_inputs(
        tokens, tp, rank, fixture, edge=edge, model_dims=model_dims, **kernel_config
    )
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (
        megapreattention as impl,
    )

    names = ["native", "megapreattention"]
    inputs = {
        name: replace(source, cache=torch.full_like(source.cache, 0.375))
        for name in names
    }
    native = NativeWindow(inputs["native"], rope, edge=edge)
    functions = {"native": native, "megapreattention": run_candidate}
    # Original-input oracle is independent of both native and candidate scratch.
    oracle_cache = torch.full_like(source.cache, 0.375)
    oracle = independent_reference(replace(source, cache=oracle_cache))
    checks = []
    for name in names:
        actual = functions[name](inputs[name])
        torch.cuda.synchronize()
        checks.append(
            {
                "implementation": name,
                "phase": "original_input",
                "errors": check_outputs(
                    actual, oracle, inputs[name].cache, oracle_cache
                ),
            }
        )
    for _ in range(warmup):
        for name in names:
            functions[name](inputs[name])
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    if mode == "cudagraph":
        for name in names:
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = functions[name](inputs[name])
            graphs[name] = graph

    calls = {
        name: (
            graphs[name].replay
            if mode == "cudagraph"
            else lambda name=name: functions[name](inputs[name])
        )
        for name in names
    }
    # Changed input and poisoned payload must work with the same registered runtime/Graph.
    for scale in (-0.75, -1.25):
        source.hidden.mul_(scale)
        oracle_cache.fill_(0.375)
        oracle = independent_reference(replace(source, cache=oracle_cache))
        for window in impl._windows.values():
            window.communication.buffer.fill_(
                float("nan")
            )  # test-only, never production
        for name in names:
            inputs[name].cache.fill_(0.375)
            if mode == "cudagraph":
                graphs[name].replay()
                actual = outputs[name]
            else:
                actual = functions[name](inputs[name])
            torch.cuda.synchronize()
            checks.append(
                {
                    "implementation": name,
                    "phase": f"changed_input_{scale}",
                    "errors": check_outputs(
                        actual, oracle, inputs[name].cache, oracle_cache
                    ),
                }
            )
    # Unsupported metadata is rejected before any collective initialization.
    unsupported = {}
    for name in ("megapreattention",):
        value = inputs[name]
        bad_cases = {
            "fp32": replace(value, hidden=value.hidden.float()),
            "strided_hidden": replace(
                value,
                hidden=torch.empty(
                    (tokens // tp, value.hidden.shape[1] * 2),
                    device="cuda",
                    dtype=torch.bfloat16,
                )[:, ::2],
            ),
            "requires_grad": replace(
                value, weight=value.weight.clone().requires_grad_(True)
            ),
            "wrong_heads": replace(
                value,
                config=replace(value.config, num_q_heads=value.config.num_q_heads + 1),
            ),
            "empty": replace(
                value,
                hidden=value.hidden[:0],
                positions=value.positions[:, :0],
                slots=value.slots[:0],
            ),
        }
        unsupported[name] = {}
        for label, bad in bad_cases.items():
            try:
                run_candidate(bad)
            except NotImplementedError:
                unsupported[name][label] = True
            else:
                raise AssertionError(f"{name} silently accepted unsupported {label}")
        del bad_cases

    samples = {name: {"wall": [], "device": []} for name in names}
    compilations = []
    original_compile = JITFunction._do_compile

    def track_compile(*args, **kwargs):
        compilations.append(True)
        return original_compile(*args, **kwargs)

    before_dynamo = dict(dynamo_counters["stats"])
    JITFunction._do_compile = track_compile
    try:
        if not check_only:
            for repeat in range(iterations):
                shift = repeat % len(names)
                for name in names[shift:] + names[:shift]:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    dist.barrier()
                    torch.cuda.synchronize()
                    begin = time.perf_counter_ns()
                    start.record()
                    calls[name]()
                    end.record()
                    torch.cuda.synchronize()
                    samples[name]["wall"].append(
                        (time.perf_counter_ns() - begin) / 1.0e6
                    )
                    samples[name]["device"].append(start.elapsed_time(end))
    finally:
        JITFunction._do_compile = original_compile
    assert not compilations, "measurement triggered TLE/Triton JIT"
    assert before_dynamo == dict(
        dynamo_counters["stats"]
    ), "measurement triggered Dynamo recompilation"
    record = {
        "rank": rank,
        "samples": samples,
        "checks": checks,
        "unsupported": unsupported,
        "jit_free": True,
        "native_source": native.source_evidence,
    }
    ranks = [None] * tp
    dist.all_gather_object(ranks, record)
    dist.barrier()
    torch.cuda.synchronize()
    for graph in graphs.values():
        graph.reset()
    outputs.clear()
    graphs.clear()
    calls.clear()
    impl.clear_megapreattention_cache()
    if rank == 0:
        metrics = {}
        if not check_only:
            for name in names:
                metrics[name] = {}
                for kind in ("wall", "device"):
                    per_rank_means = [
                        statistics.mean(r["samples"][name][kind]) for r in ranks
                    ]
                    rank_max = [
                        max(r["samples"][name][kind][i] for r in ranks)
                        for i in range(iterations)
                    ]
                    metrics[name][kind] = {
                        "slowest_rank_mean_ms": max(per_rank_means),
                        "rank_mean_ms": per_rank_means,
                        "rank_max_distribution": distribution(rank_max),
                    }
        report = {
            "tp": tp,
            "tokens": tokens,
            "mode": mode,
            "edge": edge,
            "scope": "BF16 megapreattention window, not complete model inference",
            "correctness_passed": True,
            "measurement_jit_free": True,
            "iterations": iterations,
            "warmup": warmup,
            "metrics": metrics,
            "ranks": ranks,
            "fixture": fixture["metadata"],
        }
        (Path(directory) / "report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    dist.destroy_process_group()
    native_context.__exit__(None, None, None)


def run_distributed(
    tp,
    tokens,
    mode,
    *,
    iterations=100,
    warmup=20,
    rounds=5,
    edge=False,
    check_only=False,
    fixture_path=None,
    model_dims=(2048, 16, 2),
    head_dim=256,
    rotary_dim=64,
    mrope_section=(11, 11, 10),
    eps=1e-6,
):
    import torch.multiprocessing as mp

    model_dims = tuple(model_dims)
    if len(model_dims) != 3 or any(dimension <= 0 for dimension in model_dims):
        raise ValueError("model_dims must contain positive hidden, Q and KV sizes")
    _, q_heads, kv_heads = model_dims
    if q_heads % tp or (kv_heads % tp if kv_heads >= tp else tp % kv_heads):
        raise ValueError("head counts must shard or replicate evenly across TP")
    kernel_config = dict(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=tuple(mrope_section),
        eps=eps,
    )
    reports = []
    for round_index in range(rounds):
        with tempfile.TemporaryDirectory(prefix="megapreattention-") as directory:
            context = mp.spawn(
                _worker,
                args=(
                    tp,
                    tokens,
                    mode,
                    iterations,
                    warmup,
                    edge,
                    check_only,
                    directory,
                    fixture_path,
                    model_dims,
                    kernel_config,
                ),
                nprocs=tp,
                join=False,
            )
            deadline = time.monotonic() + 600
            try:
                while not context.join(timeout=1):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "megapreattention worker round exceeded 600s"
                        )
                reports.append(
                    json.loads((Path(directory) / "report.json").read_text())
                )
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                for process in context.processes:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
    summary = {}
    if not check_only:
        kind = "wall" if mode == "operator" else "device"
        for name in ("native", "megapreattention"):
            times = [r["metrics"][name][kind]["slowest_rank_mean_ms"] for r in reports]
            summary[name] = {
                "latency_ms": statistics.mean(times),
                "round_latency_ms": times,
                "stable_all": all(
                    r["metrics"][name][kind]["rank_max_distribution"]["stable"]
                    for r in reports
                ),
            }
        for name in ("megapreattention",):
            summary[name]["speedup"] = (
                summary["native"]["latency_ms"] / summary[name]["latency_ms"]
            )
            summary[name]["performance_threshold_passed"] = (
                summary[name]["speedup"] >= 0.95
            )
            summary[name]["validated_stable_ratio"] = (
                summary["native"]["stable_all"] and summary[name]["stable_all"]
            )
    result = {
        "model_dims": model_dims,
        "kernel_config": kernel_config,
        "tp": tp,
        "tokens": tokens,
        "mode": mode,
        "correctness_passed": True,
        "measurement_jit_free": True,
        "summary": summary,
        "reports": reports,
    }
    return result
