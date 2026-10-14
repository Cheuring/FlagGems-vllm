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

"""Numerical, Graph replay and unsupported-dimension tests for MegaPreAttention."""

from types import SimpleNamespace

import pytest
import torch

from . import conftest as cfg
from .megapreattention_utils import run_distributed

pytestmark = pytest.mark.megapreattention


def require_hopper(tp=1):
    if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
        pytest.skip(f"requires {tp} CUDA GPUs")
    if any(
        torch.cuda.get_device_capability(i) != (9, 0)
        or torch.cuda.get_device_properties(i).multi_processor_count < 128
        for i in range(tp)
    ):
        pytest.skip("requires SM90 GPUs with at least 128 SMs")
    pytest.importorskip("triton.experimental.tle.language")


CASES = [(2, 1024, "cudagraph", False), (2, 1024, "cudagraph", True)]
if not cfg.QUICK_MODE:
    CASES += [
        (8, 1024, "cudagraph", False),
        (4, 1024, "operator", False),
        (8, 1024, "operator", True),
        (2, 32768, "cudagraph", True),
        (4, 32768, "cudagraph", True),
    ]


@pytest.mark.parametrize(
    "model_dims", [(2048, 16, 2), (5120, 24, 4)], ids=["35b", "27b"]
)
@pytest.mark.parametrize("tp,tokens,mode,edge", CASES)
def test_megapreattention(tp, tokens, mode, edge, model_dims):
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


@pytest.mark.parametrize("words", [544, 1088, 2050])
def test_megapreattention_metadata_bounds(words):
    require_hopper()
    import triton

    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        _initialize_metadata,
    )

    flags = torch.full((words + 8,), 17, device="cuda", dtype=torch.uint64)
    counters = torch.full((128 + 8,), 17, device="cuda", dtype=torch.int64)
    _initialize_metadata[(triton.cdiv(words, 1024),)](
        flags[:words], counters[:128], words, num_warps=4
    )
    torch.cuda.synchronize()
    assert flags.cpu().tolist() == [0] * words + [17] * 8
    assert counters.cpu().tolist() == [0] * 128 + [17] * 8


@pytest.mark.parametrize(
    "model_dims", [(2048, 16, 2), (5120, 24, 4)], ids=["35b", "27b"]
)
@pytest.mark.parametrize(
    "dimension,value",
    [
        ("hidden_size", 4096),
        ("mixed_model", None),
        ("num_heads", 32),
        ("num_kv_heads", 8),
        ("head_dim", 128),
        ("rotary_dim", 128),
        ("mrope_section", (16, 8, 8)),
        ("eps", 1e-5),
        ("tp", 1),
    ],
)
def test_megapreattention_rejects_unsupported_model_before_collectives(
    dimension, value, model_dims
):
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        megapreattention,
    )

    hidden_size, q_heads, kv_heads = model_dims
    dims = dict(
        num_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=256,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        eps=1e-6,
    )
    hidden_size = value if dimension == "hidden_size" else hidden_size
    if dimension == "mixed_model":
        hidden_size = 5120 if hidden_size == 2048 else 2048
    tp = value if dimension == "tp" else 2
    if dimension in dims:
        dims[dimension] = value
    # CPU metadata is sufficient: no rank initialization or GPU allocation is allowed.
    hidden, dummy = torch.empty(0, hidden_size), torch.empty(0)
    with pytest.raises(NotImplementedError, match="megapreattention"):
        megapreattention(
            hidden,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            SimpleNamespace(size=lambda: tp),
            **dims,
        )
