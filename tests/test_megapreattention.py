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
    "dimension,value",
    [
        ("hidden_size", 257),
        ("num_heads", 3),
        ("num_kv_heads", 3),
        ("head_dim", 128),
        ("rotary_dim", 128),
        ("mrope_section", (16, 8, 8)),
        ("eps", 1e-5),
        ("tp", 1),
    ],
)
def test_megapreattention_rejects_incompatible_metadata_before_collectives(
    dimension, value
):
    from flaggems_vllm.runtime.backend._nvidia.hopper.ops.megapreattention import (
        megapreattention,
    )

    hidden_size = value if dimension == "hidden_size" else 2048
    dims = dict(
        num_heads=16,
        num_kv_heads=2,
        head_dim=256,
        rotary_dim=64,
        mrope_section=(11, 11, 10),
        eps=1e-6,
    )
    tp = value if dimension == "tp" else 2
    if dimension in dims:
        dims[dimension] = value
    # CPU metadata is sufficient: do not initialize ranks or allocate on CUDA.
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
