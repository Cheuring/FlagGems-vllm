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

"""Paired TP benchmark using the repository shape/metric/reporting framework.

Unlike the default single-device timer, this adapter takes the slowest rank's
mean latency, then averages five independent worker rounds. --warmup/--iter
are counts for this distributed operator, matching the MegaMoE convention.
"""

import sys
from pathlib import Path

import pytest
import torch

from tests.megapreattention_utils import run_distributed

from . import base
from .conftest import Config


class MegaPreAttentionBenchmark(base.Benchmark):
    DEFAULT_METRICS = ["latency_base", "latency", "speedup"]
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def __init__(self):
        super().__init__("megapreattention", self.native, gems_op=self.candidate)
        self.paired_results = {}

    @staticmethod
    def native():
        """Baseline label; worker ranks execute the native window."""

    @staticmethod
    def candidate():
        """Candidate label; worker ranks execute megapreattention."""

    def get_input_iter(self, dtype):
        yield from self.shapes

    def get_latency(
        self, op, tokens, tp, hidden, q_heads, kv_heads, head_dim, rotary_dim
    ):
        if not torch.cuda.is_available() or torch.cuda.device_count() < tp:
            pytest.skip(f"requires {tp} CUDA GPUs")
        if any(
            torch.cuda.get_device_capability(i) != (9, 0)
            or torch.cuda.get_device_properties(i).multi_processor_count < 128
            for i in range(tp)
        ):
            pytest.skip("requires SM90 GPUs with at least 128 SMs")
        pytest.importorskip("triton.experimental.tle.language")
        mode = Config.mode.value
        if mode not in ("kernel", "operator", "cudagraph"):
            raise ValueError(
                "megapreattention requires kernel, operator or cudagraph mode"
            )
        warmup = (
            Config.warm_up
            if any(a == "--warmup" or a.startswith("--warmup=") for a in sys.argv)
            else 20
        )
        iterations = (
            Config.repetition
            if any(a == "--iter" or a.startswith("--iter=") for a in sys.argv)
            else 100
        )
        key = (
            tokens,
            tp,
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            rotary_dim,
            mode,
            warmup,
            iterations,
        )
        if key not in self.paired_results:
            self.paired_results[key] = run_distributed(
                tp,
                tokens,
                mode,
                warmup=warmup,
                iterations=iterations,
                rounds=5,
                model_dims=(hidden, q_heads, kv_heads),
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                mrope_section=(
                    (11, 11, 10)
                    if rotary_dim == 64
                    else (
                        rotary_dim // 6,
                        rotary_dim // 6,
                        rotary_dim // 2 - 2 * (rotary_dim // 6),
                    )
                ),
            )
        name = "native" if op == self.native else "megapreattention"
        return self.paired_results[key]["summary"][name]["latency_ms"]


@pytest.mark.megapreattention
def test_megapreattention():
    MegaPreAttentionBenchmark().run()
