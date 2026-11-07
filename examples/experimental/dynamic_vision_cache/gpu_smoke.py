# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real CUDA lifecycle smoke test, not a Qwen or serving integration test."""

import json

import torch
from controller import Phase, PhaseController


class CUDAResources:
    def __init__(self):
        self.cpu_weight = torch.eye(256, dtype=torch.float32)
        self.weight = None
        self.cache = None

    def load_vision(self):
        self.weight = self.cpu_weight.to("cuda")

    def encode(self, images):
        return images.to("cuda") @ self.weight

    def synchronize(self):
        torch.cuda.synchronize()

    def unload_vision(self):
        self.weight = None
        torch.cuda.empty_cache()

    def available_tokens(self, features):
        free, _ = torch.cuda.mem_get_info()
        return max(0, (free - (128 << 20)) // 4096)

    def allocate_cache(self, tokens):
        self.cache = torch.zeros(tokens * 4096, device="cuda", dtype=torch.uint8)

    def release_cache(self):
        self.cache = None

    def release_features(self, features):
        pass


def measure():
    torch.cuda.synchronize()
    return dict(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        free=torch.cuda.mem_get_info()[0],
    )


torch.cuda.init()
warmup = torch.eye(256, device="cuda")
_ = warmup @ warmup
torch.cuda.synchronize()
del warmup, _
torch.cuda.empty_cache()
r = CUDAResources()
c = PhaseController(r, 4096)
image = torch.arange(256, dtype=torch.float32).reshape(1, 256)
records = []
for index, visual in enumerate([False, True, False, True]):
    with c.request(128, 128, image if visual else None) as features:
        assert r.cache is not None and r.weight is None
        if visual:
            torch.testing.assert_close(features.cpu(), image, rtol=0, atol=0)
        records.append(dict(stage="active", visual=visual, **measure()))
    del features
    torch.cuda.empty_cache()
    assert c.phase is Phase.IDLE and r.cache is None
    records.append(dict(stage="released", visual=visual, **measure()))
try:
    with c.request(128, 128, image) as features:
        raise ValueError("generation cancelled")
except ValueError:
    pass
del features
torch.cuda.empty_cache()
assert c.phase is Phase.IDLE and r.cache is None and r.weight is None
records.append(dict(stage="cancelled_released", **measure()))
released = [x["allocated"] for x in records if x["stage"] != "active"]
assert max(released) == min(released), records
print(json.dumps(dict(status="passed", records=records), indent=2))
