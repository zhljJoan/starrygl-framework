from typing import Dict
import torch



class NodeState:
    def __init__(self, node_nums, dim, threshold = 0, threshold_time = 0, miss_times = 0, device = torch.device('cpu')):
        
        self.update_buffer = {}
        #torch.zeros(node_nums, dim, dtype=torch.float32, device=device)
        self.last_memory = {}
        #torch.zeros(node_nums, dim, dtype=torch.float32, device=device)
        self.update_time = {}
        #torch.zeros(node_nums, dtype=torch.float32, device=device)
        self.last_time = {}
        #torch.zeros(node_nums, dtype=torch.float32, device=device)
        
        self.increment_mem = torch.zeros(node_nums, dim, dtype=torch.float32, device=device)
        self.threshold = threshold
        self.threshold_time = threshold_time
        self.miss_times = miss_times
        self.node_nums = node_nums
        self.dim = dim
        self.update_count = torch.zeros(node_nums, dtype=torch.int64, device=device)
        
    def get(self, node_id):
        if node_id not in self.update_buffer:
            return None
        return self.update_buffer[node_id]
    
    def update(self, index, value, ts):
        rank = torch.distributed.get_rank()
        pass
    
    @staticmethod
    def generate_node_state_blob(self):
        return NodeStateBlob(self, self.dim)

class NodeStateBlob:
    def __init__(self, ns, dim):
        self.ns = ns
        self.node_state: dict = {}
        self.node_id: dict = {}
        self.sub_graph_mapper = torch.empty(0, dim, device = torch.device('cuda:{}'.format(torch.distributed.get_rank())))
        
        
    def insert(self, layer_id:str, node_id:torch.Tensor, value:torch.Tensor):
        self.node_state[layer_id] = value
        self.node_id[layer_id] = node_id
            
    def persistent(self, layer_id:str, index:torch.Tensor, 
                   node_id:torch.Tensor, value:torch.Tensor):
        pass
        
    
        
        
        
    