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

"""Distributed accuracy, Graph replay and input-contract coverage."""

import pytest
import torch

from . import conftest as cfg
from .megapreattention_utils import run_distributed

pytestmark = pytest.mark.megapreattention


def require_hopper(tp=1):
    if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
        pytest.skip(f"requires {tp} CUDA GPUs")
    if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(tp)):
        pytest.skip("requires SM90 GPUs")


# The shared worker checks changed-input Graph replay and poisoned transport.
# Cover each scheduling path without a second private-input test suite.
CASES = [
    ((2048, 16, 2), 2, 1024, "cudagraph", True),
    ((5120, 24, 4), 2, 1024, "cudagraph", False),
    ((3072, 32, 8), 2, 1024, "cudagraph", True),
]
if not cfg.QUICK_MODE:
    CASES += [
        ((2048, 16, 2), 2, 1280, "cudagraph", True),
        ((2048, 16, 2), 2, 32768, "cudagraph", True),
        ((2048, 16, 2), 4, 1536, "operator", True),
        ((2048, 16, 2), 4, 32768, "cudagraph", True),
        ((2048, 16, 2), 8, 2048, "cudagraph", True),
        ((2048, 16, 2), 8, 32768, "operator", False),
        ((5120, 24, 4), 4, 1024, "operator", True),
        ((5120, 24, 4), 8, 32768, "cudagraph", True),
        ((1024, 32, 16), 4, 1024, "cudagraph", True),
    ]


@pytest.mark.parametrize("model_dims,tp,tokens,mode,edge", CASES)
def test_megapreattention(model_dims, tp, tokens, mode, edge):
    require_hopper(tp)
    result = run_distributed(
        tp,
        tokens,
        mode,
        warmup=3,
        rounds=1,
        edge=edge,
        check_only=True,
        model_dims=model_dims,
    )
    assert result["correctness_passed"]
    assert result["measurement_jit_free"]
    assert all(len(r["ranks"]) == tp for r in result["reports"])


@pytest.mark.parametrize("words,programs", [(544, 128), (1088, 132), (2050, 132)])
def test_megapreattention_metadata_bounds(words, programs):
    require_hopper()
    import triton

    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        _initialize_metadata,
    )

    flags = torch.full((words + 8,), 17, device="cuda", dtype=torch.uint64)
    counters = torch.full((programs + 8,), 17, device="cuda", dtype=torch.int64)
    _initialize_metadata[(triton.cdiv(words, 1024),)](
        flags[:words], counters[:programs], words, COUNTERS=programs, num_warps=4
    )
    torch.cuda.synchronize()
    assert flags.cpu().tolist() == [0] * words + [17] * 8
    assert counters.cpu().tolist() == [0] * programs + [17] * 8


@pytest.mark.parametrize(
    "dimension,value,error,message",
    [
        ("hidden_size", 0, NotImplementedError, "positive hidden width"),
        ("num_heads", 3, NotImplementedError, "Q head count divisible by TP"),
        ("num_kv_heads", 3, NotImplementedError, "sharded or replicated KV heads"),
        ("head_dim", 0, ValueError, "rotary_dim must be even"),
        ("rotary_dim", 65, ValueError, "rotary_dim must be even"),
        ("mrope_section", (-1, 11, 22), ValueError, "nonnegative MRoPE sections"),
        ("eps", -1e-5, ValueError, "eps must be finite and nonnegative"),
        ("tp", 1, NotImplementedError, "requires TP2/TP4/TP8"),
    ],
)
def test_megapreattention_rejects_incompatible_config(dimension, value, error, message):
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        _validate_config,
    )

    config = dict(
        hidden_size=2048,
        tp=2,
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        eps=1e-6,
    )
    config[dimension] = value
    with pytest.raises(error, match=message):
        _validate_config(**config)


def test_megapreattention_general_dispatch_without_fast_path(monkeypatch):
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops import (
        megapreattention as impl,
    )

    config = impl._validate_config(2048, 2, 16, 2, 256, 64, (11, 11, 10), 1e-6)
    monkeypatch.setattr(impl, "tle", object())
    assert impl._use_persistent_path(config, 1024, 2, 128)
    assert not impl._use_persistent_path(config, 1024, 2, 120)
    monkeypatch.setattr(impl, "tle", None)
    assert not impl._use_persistent_path(config, 1024, 2, 132)


GENERAL_CASES = [
    (2, 1024, (512, 8, 2), 128, 64, (16, 8, 8), 1e-5),
    (4, 1024, (768, 16, 8), 96, 32, (6, 5, 5), 1e-5),
]
if not cfg.QUICK_MODE:
    GENERAL_CASES += [
        (2, 258, (320, 8, 2), 192, 96, (16, 16, 16), 1e-6),
        (2, 256, (257, 8, 2), 64, 0, (0, 0, 0), 1e-5),
        (2, 1024, (512, 8, 2), 512, 128, (22, 21, 21), 1e-6),
        (2, 1024, (2048, 16, 2), 256, 64, (16, 8, 8), 1e-5),
        (4, 1024, (2048, 16, 2), 256, 64, (8, 12, 12), 1e-4),
    ]


@pytest.mark.parametrize(
    "tp,tokens,model_dims,head_dim,rotary_dim,sections,eps", GENERAL_CASES
)
def test_megapreattention_general_geometry(
    tp, tokens, model_dims, head_dim, rotary_dim, sections, eps
):
    require_hopper(tp)
    result = run_distributed(
        tp,
        tokens,
        "cudagraph",
        warmup=3,
        rounds=1,
        edge=True,
        check_only=True,
        model_dims=model_dims,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        mrope_section=sections,
        eps=eps,
        disable_tle=(tp == 2 and head_dim == 256),
    )
    assert result["correctness_passed"]
    assert result["measurement_jit_free"]
