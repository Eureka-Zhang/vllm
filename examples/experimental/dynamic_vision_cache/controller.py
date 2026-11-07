# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental single-request phase orchestration; not wired into serving."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from enum import Enum, auto
from threading import Lock
from typing import Any, Protocol


class Phase(Enum):
    IDLE = auto()
    ENCODING = auto()
    GENERATING = auto()
    RECLAIMING = auto()
    FAILED = auto()


class Resources(Protocol):
    """Worker-side operations. Release methods must tolerate partial setup."""

    def load_vision(self) -> None: ...
    def encode(self, images: Any) -> Any: ...
    def synchronize(self) -> None: ...
    def unload_vision(self) -> None: ...
    def available_tokens(self, features: Any) -> int: ...
    def allocate_cache(self, tokens: int) -> None: ...
    def release_cache(self) -> None: ...
    def release_features(self, features: Any) -> None: ...


class PhaseController:
    """Serialize requests and fail closed when resource cleanup fails.

    The caller must close the context after generation, including cancellation.
    Capacity includes prompt, visual and requested output tokens. The resource
    adapter must account for workspace headroom and reset logical cache state
    before releasing physical backing. Features must outlive vision unloading.
    """

    def __init__(self, resources: Resources, max_model_len: int):
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        self.resources = resources
        self.max_model_len = max_model_len
        self.phase = Phase.IDLE
        self._lock = Lock()

    @contextmanager
    def request(
        self, prompt_tokens: int, output_tokens: int, images: Any = None
    ) -> Iterator[Any]:
        if prompt_tokens < 0 or output_tokens <= 0:
            raise ValueError("Invalid token budget")
        tokens = prompt_tokens + output_tokens
        if tokens > self.max_model_len:
            raise ValueError("Token budget exceeds model length")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A request is already active")
        features = None
        vision_attempted = cache_attempted = False
        try:
            if self.phase is not Phase.IDLE:
                raise RuntimeError("Controller requires recovery after cleanup failure")
            try:
                if images is not None:
                    self.phase = Phase.ENCODING
                    vision_attempted = True
                    self.resources.load_vision()
                    features = self.resources.encode(images)
                    self.resources.synchronize()
                    self.resources.unload_vision()
                    vision_attempted = False
                if tokens > self.resources.available_tokens(features):
                    raise MemoryError(
                        "Insufficient capacity for requested token budget"
                    )
                cache_attempted = True
                self.resources.allocate_cache(tokens)
                self.phase = Phase.GENERATING
                yield features
            finally:
                self.phase = Phase.RECLAIMING
                errors: list[Exception] = []

                def cleanup(operation: Callable[[], None]) -> None:
                    try:
                        operation()
                    except Exception as error:
                        errors.append(error)

                # Do not release storage while device operations may be in flight.
                try:
                    self.resources.synchronize()
                except Exception:
                    self.phase = Phase.FAILED
                    raise
                if cache_attempted:
                    cleanup(self.resources.release_cache)
                if vision_attempted:
                    cleanup(self.resources.unload_vision)
                if features is not None:
                    cleanup(lambda: self.resources.release_features(features))
                self.phase = Phase.FAILED if errors else Phase.IDLE
                if errors:
                    raise RuntimeError(
                        "Resource cleanup failed; restart worker"
                    ) from errors[0]
        finally:
            self._lock.release()
