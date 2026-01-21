import torch
from torch import Tensor
from typing import Optional, Union

from starrygl.cache.replica_table import CSRReplicaTable, UVACSRReplicaTable

class DistRouteIndex:
    # 编码方案: 
    # [63]: 符号位 (尽量不用)
    # [62]: shared flag
    # [48-61]: part_id (14 bits, 支持 16384 个分区)
    # [0-47]: local_index (48 bits, 支持 281 万亿个 ID)
    
    PART_SHIFT = 48
    SHARED_BIT = 62
    LOC_MASK = 0xFFFFFFFFFFFF # 48 bits
    PART_MASK = 0x3FFF         # 14 bits (1<<14 - 1)

    def __init__(self, index: Tensor, part_ids: Optional[Tensor] = None) -> None:
        if part_ids is None:
            self._data = index.long()
        else:
            # 确保在同一设备
            index = index.long()
            part_ids = part_ids.long().to(index.device)
            self._data = (index & self.LOC_MASK) | ((part_ids & self.PART_MASK) << self.PART_SHIFT)
       
    @property
    def loc(self) -> Tensor:
        return self._data & self.LOC_MASK
    
    @property
    def part(self) -> Tensor:
        return (self._data >> self.PART_SHIFT) & self.PART_MASK
    
    def set_shared(self, indx: Union[slice, Tensor]):
        self._data[indx] |= (1 << self.SHARED_BIT)

    @property
    def is_shared(self) -> Tensor:
        return (self._data >> self.SHARED_BIT).to(torch.bool)
    
    @property
    def dist(self) -> Tensor:
        return self._data
    @property
    def device(self): return self._data.device
    @property
    def shape(self): return self._data.shape
    
    def to(self, device):
        return DistRouteIndex(self._data.to(device))



class IDMapper:
    """
    通用 ID 映射器。
    专门负责管理一类实体（Node 或 Edge）的 Local/Global/Dist ID 转换及 Master 判断。
    """
    def __init__(self, 
                 loc_ids: Tensor, 
                 num_master: int,
                 mode: str = 'identity',
                 dist_mapper: Optional[Tensor] = None,
                 replica_table: Optional[Union[CSRReplicaTable, UVACSRReplicaTable]] = None,
                 offset: int = 0):
        
        self.loc_ids = loc_ids          # 本地存储的 ID (Global)
        self._num_master = num_master    # Master 数量 (Owned)
        self.mode = mode                # 'identity', 'map', 'offset', 'dist_route_index'
        self.dist_mapper = dist_mapper  # 用于获取 Dist ID (可选)
        self.offset = offset
        self._replica_table = replica_table
        # 内部缓存的反向查表 (Global -> Local)
        self._g2l_map: Optional[Tensor] = None

    @property
    def loc_nums(self):
        return self.loc_ids.size(0)
    @property
    def num_master(self):
        return self._num_master
    @property
    def global_nums(self):
        return self.dist_mapper.size(0) 
    @property
    def replica_table(self):
        return self._replica_table
    @property
    def device(self):
        return self.loc_ids.device if self.loc_ids is not None else torch.device('cpu')

    def build_g2l_map(self):
        """构建 Global -> Local 的反向查表 (仅 'map' 模式需要)"""
        if self.mode == 'map' and self.loc_ids is not None:
            max_id = self.loc_ids.max().item()
            self._g2l_map = torch.full((max_id + 1,), -1, dtype=torch.long, device=self.device)
            ids = self.loc_ids
            self._g2l_map[ids] = torch.arange(ids.size(0), dtype=torch.long, device=self.device)

    def to_local(self, gids: Union[Tensor, 'DistRouteIndex'], device: torch.device = None) -> Tensor:
        """
        [核心转换] Global/Dist ID -> Local Index
        """
        if device is None: device = self.device
        
        raw_gids = gids
        if isinstance(gids, DistRouteIndex):
            raw_gids = gids.loc # 提取低 48 位
        elif self.mode == 'dist_route_index':
            if isinstance(gids, Tensor):
                raw_gids = DistRouteIndex(gids).loc

        if isinstance(raw_gids, Tensor) and raw_gids.device != device:
            raw_gids = raw_gids.to(device)

        if self.mode == 'identity':
            return raw_gids
        elif self.mode == 'offset':
            return raw_gids - self.offset
        elif self.mode == 'map':
            if self._g2l_map is None:
                raise RuntimeError("G2L Map not built! Call build_g2l_map() first.")
            return self._g2l_map[raw_gids]
        elif self.mode == 'dist_route_index':
            return raw_gids 

        return raw_gids

    def to_dist(self, local_idx: Tensor) -> Tensor:
        """Local Index -> Dist ID"""
        # 如果有专门的 dist_mapper (Local -> Dist)，直接查
        if self.dist_mapper is not None:
            # 注意设备对齐
            return self.dist_mapper[local_idx.to(self.dist_mapper.device)].to(local_idx.device)
        target_ids = self.loc_ids.to(local_idx.device)
        return target_ids[local_idx]
    
    def gid_to_dist_route_index(self, local_idx: Tensor) -> DistRouteIndex:
        assert self.dist_mapper is not None, "dist_mapper is required for DistRouteIndex conversion"
        dist_ids = self.dist_mapper[local_idx.to(self.dist_mapper.device)].to(local_idx.device)
        return DistRouteIndex(dist_ids)
    
    def get_master_part_by_dist(self, dist_idx: DistRouteIndex) -> Tensor:
        """获取 Partition ID"""
        assert isinstance(dist_idx, DistRouteIndex), "dist_idx must be DistRouteIndex"
        return dist_idx.part

    def get_master_part_by_global(self, gid: Tensor) -> Tensor:
        """根据 Global ID 获取 Partition ID"""
        dist_idx = self.dist_mapper[gid.to(self.dist_mapper.device)]
        dist_idx = DistRouteIndex(dist_idx.to(gid.device))
        return self.get_master_part_by_dist(dist_idx)
    
    def is_master_by_local(self, local_idx: Tensor) -> Tensor:
        return (local_idx >= 0) & (local_idx < self._num_master)
    
    def is_store_local(self, global_idx: Tensor) -> Tensor:
        if self.mode == 'identity':
            local_idx = global_idx
        else:
            local_idx = self.to_local(global_idx, device=global_idx.device)
            return local_idx >= 0
        return local_idx >= 0


class PartitionState:
    def __init__(self, 
                 # 节点参数
                 loc_ids: Tensor, 
                 num_master_nums: int,
                 dist_nid_mapper: Optional[Tensor] = None,
                 node_mode: str = 'identity',
                 node_replica_table: Optional[Union[CSRReplicaTable, UVACSRReplicaTable]] = None,
                 
                 # 边参数 (可选)
                 loc_eids: Optional[Tensor] = None,
                 num_master_edges: int = 0,
                 dist_eid_mapper: Optional[Tensor] = None,
                 edge_mode: str = 'identity',
                 edge_replica_table: Optional[Union[CSRReplicaTable, UVACSRReplicaTable]] = None,
                 
                ):
        
        # --- 组合模式: 实例化 Node Mapper ---
        self.node_mapper = IDMapper(
            loc_ids=loc_ids,
            num_master=num_master_nums,
            mode=node_mode,
            dist_mapper=dist_nid_mapper,
            replica_table=node_replica_table
        )
        
        # --- 组合模式: 实例化 Edge Mapper (按需) ---
        self.edge_mapper = None
        if loc_eids is not None:
            self.edge_mapper = IDMapper(
                loc_ids=loc_eids,
                num_master=num_master_edges,
                mode=edge_mode,
                dist_mapper=dist_eid_mapper,
                replica_table=edge_replica_table
            )

    def build_g2l_maps(self):
        """统一构建"""
        self.node_mapper.build_g2l_map()
        if self.edge_mapper:
            self.edge_mapper.build_g2l_map()

    # =================================================
    # (Node API)
    # =================================================
    
    def to_local_nid(self, gids: Union[Tensor, 'DistRouteIndex'], device: torch.device = None):
        """全局点 ID -> 本地索引"""
        return self.node_mapper.to_local(gids, device)
    
    def get_dist_nid(self, local_idx: Tensor):
        """本地索引 -> 分布式点 ID"""
        return self.node_mapper.to_dist(local_idx)
    
    def get_node_part_by_dist(self, global_idx: DistRouteIndex):
        """获取点的分区号"""
        return self.node_mapper.get_master_part_by_dist(global_idx)

    def is_master_node_by_local(self, local_idx: Tensor) -> Tensor:
        """[新增] 判断点是否为 Master"""
        return self.node_mapper.is_master_by_local(local_idx)

    def is_master_node_by_global(self, gid: Tensor) -> Tensor:
        """[新增] 根据全局 ID 判断是否为 Master"""
        local_id = self.to_local_nid(gid)
        return self.is_master_node_by_local(local_id)

    # =================================================
    #  (Edge API)
    # =================================================

    def _check_edge(self):
        if self.edge_mapper is None:
            raise RuntimeError("Edge data not initialized in PartitionState")

    def to_local_eid(self, gids: Union[Tensor, 'DistRouteIndex'], device: torch.device = None):
        self._check_edge()
        return self.edge_mapper.to_local(gids, device)

    def get_dist_eid(self, local_idx: Tensor):
        self._check_edge()
        return self.edge_mapper.to_dist(local_idx)
    
    def get_edge_part_by_dist(self, global_idx: DistRouteIndex):
        self._check_edge()
        return self.edge_mapper.get_master_part_by_dist(global_idx)

    def is_master_edge_by_local(self, local_idx: Tensor) -> Tensor:
        """[新增] 判断边是否为 Master"""
        self._check_edge()
        return self.edge_mapper.is_master_by_local(local_idx)

    
# class PartitionState:
#     def __init__(self, loc_ids: Tensor, loc_eids: Tensor,
#                  num_master_nums: int,
#                  num_master_edges: int,
#                  mode = 'identity',
#                  is_shared: Tensor = None,
#                  replica_table:CSRReplicaTable|UVACSRReplicaTable = None, 
#                  dist_nid_mapper: Optional[Tensor] = None,
#                  dist_eid_mapper: Optional[Tensor] = None):
#         """
#         partition_book: 分区映射，通常是一个字典或类似结构
#         is_shared: 一个布尔张量，指示每个节点是否是共享的
#         """
#         self.replica_table = replica_table
#         self.loc_ids = loc_ids
#         self.loc_eids = loc_eids
#         self.is_shared = is_shared
#         self.dist_nid_mapper = dist_nid_mapper
#         self.dist_eid_mapper = dist_eid_mapper
#         self.num_master_nums = num_master_nums
#         self.num_master_edges = num_master_edges
        
#     def build_g2l_maps(self):
#         """
#         构建 Global ID 到 Local Buffer Index 的映射表。
#         用于在 DataLoader 中将 batch 的全局 ID 转换为本地特征索引。
#         """
#         # --- Node Mapping ---
#         if self.loc_ids is not None:
#             max_nid = self.loc_ids.max().item()
#             self._nid_g2l_map = torch.full((max_nid + 1,), -1, dtype=torch.long, device=self.device)
#             ids = self.loc_ids.to(self.device)
#             self._nid_g2l_map[ids] = torch.arange(ids.size(0), dtype=torch.long, device=self.device)
            
#         # --- Edge Mapping ---
#         if self.loc_eids is not None:
#             max_eid = self.loc_eids.max().item()
#             self._eid_g2l_map = torch.full((max_eid + 1,), -1, dtype=torch.long, device=self.device)
#             eids = self.loc_eids.to(self.device)
#             self._eid_g2l_map[eids] = torch.arange(eids.size(0), dtype=torch.long, device=self.device)

    
#     def get_replica_table(self):
#         return self.replica_table
    
#     def is_shared(self, index: Tensor) -> Tensor:
#         return self.is_shared[index]
    
#     def to_local(self, gids: torch.Tensor, device: torch.device) -> Tensor:
#         if gids.device != device:
#             gids = gids.to(device)  
#         if self.mode == 'identity':
#             return gids
#         elif self.mode == 'dist_route_index':
#             if isinstance(gids, DistRouteIndex):
#                 local_ids = gids.loc
#                 return local_ids
#             else:
#                 dist_index = DistRouteIndex(gids)
#                 local_ids = dist_index.loc
#             return local_ids
#         elif self.mode == 'offset':
#             return gids - self.offset
#         elif self.mode == 'map':
#             return self._nid_g2l_map[gids]

#     def get_dist_id(self, local_id: Tensor) -> Tensor:
#         """
#         获取分布式 ID。
#         """
#         #print(self.dist_nid_mapper)
#         if self.dist_nid_mapper is not None:
#             print(local_id.device, self.loc_ids.device, self.dist_nid_mapper.device)
#             return self.dist_nid_mapper[self.loc_ids[local_id.to(self.loc_ids.device)]].to(local_id.device)
#         else:
#             return self.loc_ids[local_id]
    
#     def get_dist_eid(self, local_id: Tensor) -> Tensor:
#         """
#         获取分布式边 ID。
#         """
#         if self.dist_eid_mapper is not None:
#             return self.dist_eid_mapper[self.loc_eids[local_id.to(self.loc_eids.device)]].to(local_id.device)
#         else:
#             return self.loc_eids[local_id]
    
#     def get_part(self, local_id: Tensor):
#         return DistRouteIndex(self.get_dist_id(local_id)).part
    
#     def get_epart(self, local_id:Tensor):
#         return DistRouteIndex(self.get_dist_eid(local_id)).part
    
#     def is_master(self, local_id: Tensor) -> Tensor:
#         return local_id < self.num_master_nums
    
#     def is_master_by_global(self, gid: Tensor) -> Tensor:
#         local_id = self.to_local(gid, self.device)
#         return self.is_master(local_id)