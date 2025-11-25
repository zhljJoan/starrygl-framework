import torch

from typing import *

import time
from contextlib import contextmanager



class EventTimer:
    def __init__(self):
        self.elapsed_times: Dict[str, float] = {}
        self.record_events: Dict[str, List[Tuple[_Event, _Event]]] = {}
    
    @contextmanager
    def record(self, name: str = "_default_"):
        s = _Event()
        t = _Event()

        s.record()
        yield
        t.record()

        if name not in self.record_events:
            self.record_events[name] = []
        self.record_events[name].append((s, t))
    
    def reset(self):
        self.elapsed_times.clear()
        self.record_events.clear()
    
    def __getitem__(self, name: str) -> float:
        return self.get(name)
    
    def get(self, name: str = "_default_") -> float:
        value = self.elapsed_times.get(name, 0.0)
        if name in self.record_events:
            for s, t in self.record_events[name]:
                value += s.elapsed_time(t)
            del self.record_events[name]

        self.elapsed_times[name] = value
        return value


class _Event:
    def __init__(self):
        if torch.cuda.is_available():
            self.type = "cuda"
        else:
            self.type = "cpu"

        self.inner: torch.cuda.Event | float | None = None
    
    def record(self):
        if self.type == "cuda":
            self.inner = torch.cuda.Event(enable_timing=True)
            self.inner.record()
        elif self.type == "cpu":
            self.inner = time.perf_counter()
    
    def elapsed_time(self, other: '_Event'):
        if self.type == "cuda":
            return self.inner.elapsed_time(other.inner) / 1000
        elif self.type == "cpu":
            return other.inner - self.inner
    
    def synchronize(self):
        if self.type == "cuda":
            self.inner.synchronize()
