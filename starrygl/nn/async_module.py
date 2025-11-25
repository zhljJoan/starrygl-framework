import torch
import torch.nn as nn

from torch import Tensor
from typing import *

import inspect
import asyncio


__all__ = [
    'AsyncModule',
]


class AsyncModule(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def layerwise(self, *iters: Iterable[Any]):
        iters = zip(*iters)
        iters = reversed(list(iters)) # spatial block first

        coros = [self.async_forward(*args) for args in iters]
        outs = _apply_layerwise_batch(coros)

        outs = list(reversed(outs))
        return outs
    
    # def adaptive_layerwise(self, *iters: Iterable[Any]):
    #     from flare.core.adaptive import _adaptive_layerwise_batch
    #     iters = zip(*iters)
    #     iters = reversed(list(iters)) # spatial block first

    #     coros = [self.async_forward(*args) for args in iters]
    #     outs = _adaptive_layerwise_batch(coros)

    #     outs = list(reversed(outs))
    #     return outs

    async def async_forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    async def yield_forward(self) -> None:
        await asyncio.sleep(0.0)
    
    def forward(self, *args, **kwargs) -> Any:
        c = self.async_forward(*args, **kwargs)
        while True:
            try:
                c.send(None)
            except StopIteration as e:
                return e.value


def _apply_layerwise_batch(coros: Coroutine | List[Coroutine]):
    if isinstance(coros, Coroutine):
        coros = [coros]
        _list = False
    elif isinstance(coros, (list, tuple)):
        coros = list(coros)
        _list = True
    else:
        raise TypeError(f"Expected coroutine or list of coroutines, but got {type(coros)}")
    
    n = len(coros)
    next_coros = []
    for i, c in enumerate(coros):
        if not inspect.iscoroutine(c):
            raise TypeError(f"Expected coroutine, but got {type(c)}")
        next_coros.append((i, c))

    fins = [False] * n
    fcnt = n

    outs = [None] * n
    while fcnt > 0:
        coros, next_coros = next_coros, []
        for i, c in coros: # early ready only
            if fins[i]:
                continue
            try:
                cast(Coroutine, c).send(None)
            except StopIteration as e:
                outs[i] = e.value
                fins[i] = True
                fcnt -= 1
            else:
                next_coros.append((i, c))
    return outs if _list else outs[0]
