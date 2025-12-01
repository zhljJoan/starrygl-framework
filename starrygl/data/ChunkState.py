import torch
class ChunkState:
    def __init__(self, chunk_index, chunk_count):
        value = torch.stack((torch.arange(chunk_count),chunk_index))
        _value, ind = value.sort(dim = 1)
        self.chunk_nodes = _value[1,:]
        self.chunk_ptr = torch.zeros(chunk_count + 1, dtype=torch.int64)
        self.chunk_ptr[1:] = torch.cumsum(torch.bincount(self.chunk_indicesp[0,:]), dim=0)
        self.order_ind = ind