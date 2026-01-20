import torch

class AsyncTransferFuture:
    """
    基于 CUDA Event 的异步传输句柄。
    """
    def __init__(self, tensor: torch.Tensor, event: torch.cuda.Event):
        self._tensor = tensor
        self._event = event

    def wait(self) -> torch.Tensor:
        """
        非阻塞同步：让当前计算流等待 Event 记录时刻。
        """
        torch.cuda.current_stream().wait_event(self._event)
        return self._tensor

    @property
    def is_ready(self):
        return self._event.query()