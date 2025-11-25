import torch
import torch.distributed as dist

import os
import psutil

from torch import Tensor
from typing import *

import io
import time
import json
import logging
import datetime

from tqdm import tqdm
from pathlib import Path
from contextlib import contextmanager
import starrygl

__all__ = [
    "DistributedContext",
]

from starrygl.lib.libstarrygl_comm import DistContext as DistContextCpp
from starrygl.lib.libstarrygl_comm import init_p2p as init_p2p_cpp

class DistributedContext:
    """Global context manager for distributed training
    """
    @classmethod
    def init(cls, backend: str = "nccl") -> 'DistributedContext':
        if cls.is_initialized():
            raise RuntimeError("not allowed to call init method twice.")

        rank = int(os.getenv("RANK") or os.getenv("OMPI_COMM_WORLD_RANK"))
        world_size = int(os.getenv("WORLD_SIZE") or os.getenv("OMPI_COMM_WORLD_SIZE"))
        local_rank = os.getenv("LOCAL_RANK") or os.getenv("OMPI_COMM_WORLD_LOCAL_RANK")
        local_size = os.getenv("LOCAL_SIZE") or os.getenv("OMPI_COMM_WORLD_LOCAL_SIZE")
        if local_rank is None:
            logging.warning(f"LOCAL_RANK has not been set, using the default value 0.")
            os.environ["LOCAL_RANK"] = local_rank = "0"
        local_rank = int(local_rank)

        master_addr = os.environ["MASTER_ADDR"]
        master_port = int(os.environ["MASTER_PORT"])
        init_method = f"tcp://{master_addr}:{master_port}"

        ctx = DistributedContext(
            backend=backend, init_method=init_method,
            rank=rank, world_size=world_size, local_rank=local_rank, local_size = local_size,
            master_addr = master_addr, master_port = master_port
        )
        
        setattr(cls, "_instance_", ctx)
        return ctx
    
    @classmethod
    def get_default_context(cls) -> 'DistributedContext':
        if not cls.is_initialized():
            raise RuntimeError("please call the init method first.")
        return getattr(cls, "_instance_")

    @classmethod
    def is_initialized(cls) -> bool:
        return hasattr(cls, "_instance_")

    def __init__(self,
        backend: str, init_method: str,
        rank: int, world_size: int, local_rank: int, local_size:int,
        master_addr: str, master_port: str
    ) -> None:
        if DistributedContext.is_initialized():
            raise RuntimeError("there has been a context instance.")
        
        self.master_addr = master_addr
        self.master_port = master_port
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank, world_size=world_size,
        )

        self._group = dist.GroupMember.WORLD
        _, self._store = dist.distributed_c10d._world.pg_map[self._group]

        if dist.get_backend(self._group) != "gloo":
            self._cpu_group = dist.new_group(backend="gloo")
        else:
            self._cpu_group = self._group
        
        self._rank = rank
        self._world_size = world_size
        self._local_rank = local_rank
        self._device = device

        t = torch.tensor([local_rank], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self._group)
        self._local_size = t.item() + 1
        
        self.init_hybird_feature()

    def init_hybird_feature(self):
        if self._rank == 0:
            id = starrygl.lib.libstarrygl_comm.create_nccl_id()
            store = torch.distributed.distributed_c10d._get_default_store()
            store.set("nccl_id", id)
            dist.barrier()
        else:
            dist.barrier()
            store = torch.distributed.distributed_c10d._get_default_store()
            id = store.get("nccl_id")
        self.cpp_context = starrygl.lib.libstarrygl_comm.DistContext(
            self._rank, self._world_size, self._local_size, id
        )
        device_list = list(range(self._local_size))
        starrygl.lib.libstarrygl_comm.init_p2p(device_list)
    def __del__(self):
        self.__del_disk_logger__()
        if self._cpu_group is self._group:
            dist.destroy_process_group(self._group)
        else:
            dist.destroy_process_group(self._cpu_group)
            dist.destroy_process_group(self._group)
    
    @property
    def rank(self) -> int:
        return dist.get_rank()
    
    @property
    def size(self) -> int:
        return dist.get_world_size()
    
    @property
    def world_size(self) -> int:
        return dist.get_world_size()
    
    @property
    def local_rank(self) -> int:
        return self._local_rank
    
    @property
    def local_size(self) -> int:
        world_size = self.world_size
        local_size = self._local_size
        assert world_size % local_size == 0
        return local_size
    
    @property
    def cross_rank(self) -> int:
        return self.world_size % self.local_size
    
    @property
    def cross_size(self) -> int:
        return self.world_size // self.local_size

    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def group(self) -> dist.ProcessGroup:
        return self._group
    
    @property
    def store(self) -> dist.Store:
        return self._store
    
    @property
    def cpu_group(self) -> dist.ProcessGroup:
        return self._cpu_group
    
    def sync_print(self, *args, **kwargs):
        for i in range(self.world_size):
            if i == self.rank:
                print(f"rank {self.rank}:", *args, **kwargs)
            dist.barrier(group=self.cpu_group)

    def main_print(self, *args, **kwargs):
        if self.rank == 0:
            print(*args, **kwargs)
        dist.barrier(group=self.cpu_group)
    
    def all_reduce_mean_scalar(self, x: float, n: int = 1) -> float:
        xs = [None] * self.group.size()
        dist.all_gather_object(xs, obj=(x, n), group=self.cpu_group)
        x = [t for t, _ in xs]
        n = [t for _, t in xs]
        return sum(x) / max(1e-10, sum(n))
    
    @contextmanager
    def tqdm(self, total: int | None = None, desc: str | None = None):
        disable = self.local_rank != 0
        with tqdm(total=total, desc=desc, disable=disable) as bar:
            yield bar
    
    def trange_epoch(self, epochs: int, desc: str | None = None):
        assert not hasattr(self, "_epoch_bar_")
        with self.tqdm(total=epochs, desc=desc) as bar:
            setattr(self, "_epoch_bar_", bar)
            for ep in range(epochs):
                yield ep
            delattr(self, "_epoch_bar_")

    def update_epoch(self, desc: str, n: int = 1):
        tmp_bar: tqdm | None = getattr(self, "_epoch_bar_", None)
        if tmp_bar is None:
            return
        tmp_bar.set_description(desc)
        tmp_bar.update(n)
    
    def memory_allocated(self):
        torch.cuda.synchronize()
        max_memory_allocated = torch.cuda.max_memory_allocated()
        # torch.cuda.reset_max_memory_allocated()
        return f"{max_memory_allocated / 1024 / 1024:.2f}MB"
    
    def host_memory_allocated(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return f"{memory_info.rss / 1024 / 1024:.2f}MB"
    
    def memory_desc(self, sep = "=>"):
        dev_mem = self.memory_allocated()
        host_mem = self.host_memory_allocated()
        return f"memory[gpu]{sep}{dev_mem} memory[cpu]{sep}{host_mem}"
    
    def init_disk_logger(self, root: str | Path, params: Dict[str, Any] | None = None):
        root_key = "__disk_logger_root__"
        file_key = "__disk_logger_file_open__"
        time_key = "__disk_logger_init_time__"

        if hasattr(self, root_key):
            raise RuntimeError("Disk logger has already been initialized.")
        
        name = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_GPU{self.world_size:02d}"
        root = Path(root).expanduser().resolve() / name
        setattr(self, root_key, root)

        if self.rank != 0:
            return
        root.mkdir(parents=True, exist_ok=True)

        if params is not None:
            with open(root / "params.txt", "w") as f:
                json.dump(params, f, ensure_ascii=False, indent=4)

        file_open = open(root / "log.txt", "a", encoding="utf-8")
        init_time = time.perf_counter()
        setattr(self, file_key, file_open)
        setattr(self, time_key, init_time)
    
    def __del_disk_logger__(self):
        root_key = "__disk_logger_root__"
        file_key = "__disk_logger_file_open__"
        time_key = "__disk_logger_init_time__"
        
        file_open: io.TextIOWrapper | None = getattr(self, file_key, None)
        if file_open is not None:
            file_open.close()
        
        if hasattr(self, root_key):
            delattr(self, root_key)
        
        if hasattr(self, file_key):
            delattr(self, file_key)
        
        if hasattr(self, time_key):
            delattr(self, time_key)
    
    def info(self, ep: int, train_loss: float, eval_loss: float, comm_time: float = 0.0) -> None:
        file_key = "__disk_logger_file_open__"
        time_key = "__disk_logger_init_time__"
        
        file_open: io.TextIOWrapper | None = getattr(self, file_key, None)
        if file_open is not None:
            memory = self.memory_desc("=>")
            duration = time.perf_counter() - getattr(self, time_key)
            text = f"epoch=>{ep}:{duration:.4f}s train_loss=>{train_loss:.4f} eval_loss=>{eval_loss:.4f} comm_time=>{comm_time:.4f} {memory}\n"
            # print(text)
            file_open.write(text)
            file_open.flush()


