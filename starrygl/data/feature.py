import torch
import starrygl
from starrygl.core.route import DistRouteIndex


class FeatureIpcList():
    def __init__(self, local_rank: int, dist_index: torch.Tensor, feature: torch.Tensor):
        shared_tensor_gpu = feature[DistRouteIndex(dist_index).is_shared()].to('cude:{}'.format(local_rank))
        local_tensor_feature = feature[~DistRouteIndex(dist_index).is_shared()].to('cpu')
        

    
class SharedFeature(object):
    
    def __init__(self, tensor
                 
                ):
    def generate_ipc_handle(self, f: torch.Tensor):
        
    def get_new_tensor
    
    @ipc_handle.setter
    def ipc_handle(self, ipc_handle):
        self.ipc_handle_ = ipc_handle

    def share_ipc(self):
        self.cpu_part.share_memory_()
        gpu_ipc_handle_dict = {}
        if self.cache_policy == "device_replicate":
            for device in self.device_tensor_list:
                gpu_ipc_handle_dict[device] = self.device_tensor_list[
                    device].share_ipc()[0]
        else:
            for clique_id in self.clique_tensor_list:
                gpu_ipc_handle_dict[clique_id] = self.clique_tensor_list[
                    clique_id].share_ipc()[0]

        return gpu_ipc_handle_dict, self.cpu_part if self.cpu_part.numel() > 0 else None, self.device_list, self.device_cache_size, self.cache_policy, self.csr_topo

    def from_gpu_ipc_handle_dict(self, gpu_ipc_handle_dict, cpu_tensor):
        if self.cache_policy == "device_replicate":
            ipc_handle = gpu_ipc_handle_dict.get(
                self.rank, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(
                ipc_handle, self.rank)
            self.device_tensor_list[self.rank] = shard_tensor

        else:
            clique_id = self.topo.get_clique_id(self.rank)
            ipc_handle = gpu_ipc_handle_dict.get(
                clique_id, []), cpu_tensor, ShardTensorConfig({})
            shard_tensor = ShardTensor.new_from_share_ipc(
                ipc_handle, self.rank)
            self.clique_tensor_list[clique_id] = shard_tensor

        self.cpu_part = cpu_tensor

    @classmethod
    def new_from_ipc_handle(cls, rank, ipc_handle):
        """Create from ipc handle

        Args:
            rank (int): device rank for feature collection kernels to launch
            ipc_handle (tuple): ipc handle create from `share_ipc`

        Returns:
            [quiver.Feature]: created quiver.Feature
        """
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, csr_topo = ipc_handle
        feature = cls(rank, device_list, device_cache_size, cache_policy)
        feature.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        if csr_topo is not None:
            feature.feature_order = csr_topo.feature_order.to(rank)
        feature.csr_topo = csr_topo
        return feature

    @classmethod
    def lazy_from_ipc_handle(cls, ipc_handle):
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, _ = ipc_handle
        feature = cls(device_list[0], device_list, device_cache_size,
                      cache_policy)
        feature.ipc_handle = ipc_handle
        return feature

    def lazy_init_from_ipc_handle(self):
        if self.ipc_handle is None:
            return

        self.rank = torch.cuda.current_device()
        gpu_ipc_handle_dict, cpu_part, device_list, device_cache_size, cache_policy, csr_topo = self.ipc_handle
        self.from_gpu_ipc_handle_dict(gpu_ipc_handle_dict, cpu_part)
        self.csr_topo = csr_topo
        if csr_topo is not None:
            self.feature_order = csr_topo.feature_order.to(self.rank)

        self.ipc_handle = None


class PartitionInfo:
    """PartitionInfo is the partitioning information of how features are distributed across nodes.
    It is mainly used for distributed feature collection, by DistFeature.

    Args:
        device (int): device for local feature partition
        host (int): host id for current node
        hosts (int): the number of hosts in the cluster
        global2host (torch.Tensor): global feature id to host id mapping
        replicate (torch.Tensor, optional): CSRTopo of the graph for feature reordering
        
    """
    def __init__(self, device, host, hosts, global2host, replicate=None):
        self.global2host = global2host.to(device)
        self.host = host
        self.hosts = hosts
        self.device = device
        self.size = self.global2host.size(0)
        self.replicate = None
        if replicate is not None:
            self.replicate = replicate.to(device)
        self.init_global2local()

    def init_global2local(self):
        total_range = torch.arange(end=self.size,
                                   device=self.device,
                                   dtype=torch.int64)
        self.global2local = torch.arange(end=self.size,
                                         device=self.device,
                                         dtype=torch.int64)
        for host in range(self.hosts):
            mask = self.global2host == host
            host_nodes = torch.masked_select(total_range, mask)
            host_size = host_nodes.size(0)
            if host == self.host:
                local_size = host_size
            host_range = torch.arange(end=host_size,
                                      device=self.device,
                                      dtype=torch.int64)
            self.global2local[host_nodes] = host_range
        if self.replicate is not None:
            self.global2host[self.replicate] = self.host
            replicate_range = torch.arange(start=local_size,
                                           end=local_size +
                                           self.replicate.size(0),
                                           device=self.device,
                                           dtype=torch.int64)
            self.global2local[self.replicate] = replicate_range

    def dispatch(self, ids):
        host_ids = []
        host_orders = []
        ids_range = torch.arange(end=ids.size(0),
                                 dtype=torch.int64,
                                 device=self.device)
        host_index = self.global2host[ids]
        for host in range(self.hosts):
            mask = host_index == host
            host_nodes = torch.masked_select(ids, mask)
            host_order = torch.masked_select(ids_range, mask)
            host_nodes = self.global2local[host_nodes]
            host_ids.append(host_nodes)
            host_orders.append(host_order)
        torch.cuda.current_stream().synchronize()

        return host_ids, host_orders






class FeatureStore:
    def __init__(self, feat = None):
        pass
    def get_feat(self,idx):
        '''
        Retrieves features for the specified indices.

        Args:
            idx: Indices for which to retrieve features.

        Returns:
            Features corresponding to the specified indices.
        '''
        raise NotImplementedError("This method should be implemented by subclasses.")
    def __getitem__(self, idx: int):
        return self.get_feat(idx)
    
class DistributedFeatureStore(FeatureStore):
    def __init__(self, feat=None, device=torch.device('cuda')):
        super(DistributedFeatureStore, self).__init__(feat)
        self.device = device if device.type == 'cpu' else torch.device('cuda:{}'.format(dist.get_rank()))
        if feat is not None:
            self.feat = DistributedTensor(feat.to(self.device))
        else:
            self.feat = None

    def get_feat(self, idx):
        return self.feat.accessor.data[idx].to(self.device)
class CacheFeatureStore(FeatureStore):
    def __init__(self, feat=None, cache_policy='Schedule',cache_settings = None):
        super(CacheFeatureStore,self).__init__(feat)
        if cache_policy == 'Schedule':  
            self.cache = ScheduleCache(
                               cache_ratio = cache_settings['cache_ratio'],
                               num_cache = cache_settings['num_cache'],
                               cache_data = feat,
                               use_local = False,
                               pinned_buffers_shape = feat.shape,
                                #cache_schedule = cache_settings['cache_schedule']
                               )
            self.is_train = False
        #self.feat = feat
            #self.cache.feat_dict = self.cache.feat_dict.to('cuda:{}'.format(dist.get_rank()))
    def train(self):
        self.is_train =True
    def eval(self):
        self.is_train = False
    def get_feat(self, idx, need_update = False, async_pool = None):
        #out = self.cache.feat_dict[idx.to('cuda:{}'.format(dist.get_rank()))].to('cuda:{}'.format(dist.get_rank()))
        #out = self.cache.feat_dict[idx.to('cpu')].to('cuda:{}'.format(dist.get_rank()))
        #print(async_pool)
        out = self.cache.fetch_data(idx,self.cache.get_uncache_feat,idx,
                                   async_pool = async_pool)
        if need_update is True:
           pass
           #self.cache.update_cache(async_pool = async_pool)
        return out
    def reset(self):
        self.cache.reset()
        
    def set_write_back_cpu(self, pool = None):
        self.cache.set_write_back_cpu(pool)
        
    @property
    def shape(self):
        return self.cache.shape
        
class MixtureFeatureStore(FeatureStore):
    def __init__(self, ids, eids, edge_index, ts,
                 shared_nids=None, nids_mapper=None, eids_mapper=None, shared_nids_list=None,
                 nfeat=None, efeat=None, device=torch.device('cuda'), feature_device=torch.device('cuda'),
                 whole_node_feature=None, whole_edge_feature=None):
        super(MixtureFeatureStore, self).__init__(nfeat)
       