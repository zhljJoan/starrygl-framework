#pragma once
#define TORCH_DISABLE_FP8
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <torch/torch.h>
#include <curand_kernel.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <error.cu.hpp>
#include <cstring>
namespace py = pybind11;
typedef unsigned long long ULL;
#define BLOOM_SIZE 512
#define BLOOM_BITS (BLOOM_SIZE)
#define BLOOM_BYTES ((BLOOM_SIZE + 7) / 8)




using T  = int64_t;
using TS = int64_t;

#define Edge_T Edge<T, TS>
// ------------------------------------------------------------------
// 1. POD 结构体代替 std::tuple
// ------------------------------------------------------------------
template <typename T = int64_t, typename TS = int64_t>
struct Edge {
    T  src, dst, eid;
    TS ts;
    T scc,dcc;
    __host__ __device__ Edge() = default;
    __host__ __device__ Edge(T s, T d, TS t, T e, T scc_, T dcc_) : src(s), dst(d), ts(t), eid(e), scc(scc_), dcc(dcc_){}
};

// // ------------------------------------------------------------------
// // 2. 全局 __device__ 比较函数（Thrust 直接调用）
// // ------------------------------------------------------------------
template <typename T = int64_t, typename TS = int64_t>
__device__ bool edge_cmp(const Edge<T,TS>& a, const Edge<T,TS>& b) {
    T sc = a.scc, dc = a.dcc;
    T scc = b.scc, dcc = b.dcc;
    if (sc != scc) return sc < scc;
    if (a.src != b.src) return a.src < b.src;
    if (a.ts != b.ts) return a.ts < b.ts;
    if (dc != dcc) return dc < dcc;
    return a.dst < b.dst;
}

// ------------------------------------------------------------------
// 5. 所有 kernel 实现
// ------------------------------------------------------------------
template <typename T = int64_t, typename TS = int64_t>
__global__ void hash_based_remapping_kernel(T* nodes_id, int* mappings,
                                          int* hash_table, int* hash_counts,
                                          int table_size, int total_nodes,
                                          float duplication_tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_nodes) {
        int node_id = sampled_nodes[idx];
        int hash_index = node_id % table_size;
        
        // 线性探测解决冲突
        int probe_count = 0;
        while (probe_count < table_size) {
            int current_node = hash_table[hash_index];
            
            if (current_node == -1) {
                // 空槽，插入新节点
                atomicExch(&hash_table[hash_index], node_id);
                atomicAdd(&hash_counts[hash_index], 1);
                mappings[idx] = hash_index;  // 使用哈希索引作为局部ID
                break;
            }
            else if (current_node == node_id) {
                // 找到相同节点
                atomicAdd(&hash_counts[hash_index], 1);
                mappings[idx] = hash_index;
                break;
            }
            else {
                // 冲突，继续探测
                hash_index = (hash_index + 1) % table_size;
                probe_count++;
                
                // 如果冲突过多，容忍重复
                if (probe_count > table_size * duplication_tolerance) {
                    mappings[idx] = hash_index;  // 强制映射，接受重复
                    break;
                }
            }
        }
    }
}
// //线程级别的去重吗？每个chunk单独去重
// template <typename T = int64_t, typename TS = int64_t>
// __global__ void get_reindx_by_hash(T* nodes_id, int* mappings, int *reindex_cnt,
//                                           int* hash_table,
//                                           int table_size,
//                                           float duplication_tolerance) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int hash_index = node_id % table_size;
        
//         // 线性探测解决冲突 
//         int probe_count = 0;
//         while (probe_count < table_size) {
//             int current_node = hash_table[hash_index];
            
//             if (current_node == -1) {
//                 // 空槽，插入新节点
//                 atomicExch(&hash_table[hash_index], node_id);
//                 mappings[nodes_id] = atomicExch(reindex_cnt, 1);  // 使用哈希索引作为局部ID
//                 break;
//             }
//             else if (current_node == node_id) {
//                 // 找到相同节点
//                 break;
//             }
//             else {
//                 // 冲突，继续探测
//                 hash_index = (hash_index + 1) % table_size;
//                 probe_count++;
                
//                 // 如果冲突过多，容忍重复
//                 if (probe_count > table_size * duplication_tolerance) {
//                     mappings[nodes_id] = atomicExch(reindex_cnt, 1);  // 强制映射，接受重复
//                     break;
//                 }
//             }
//         }
//     }
// }

template <typename T = int64_t, typename TS = int64_t>
__device__ T lower_bound(const TS* arr, T l, T r, TS v) {
    while (l < r) { T m = (l + r) >> 1; if (arr[m] < v) l = m + 1; else r = m; } return l;
}
template <typename T = int64_t, typename TS = int64_t>
__device__ T upper_bound(const TS* arr, T l, T r, TS v) {
    while (l < r) { T m = (l + r) >> 1; if (arr[m] <= v) l = m + 1; else r = m; } return l;
}
template <typename T = int64_t, typename TS = int64_t>
__global__ void build_graph_kernel(
    const Edge_T* edges, T* row_ptr, T* col_idx, TS* col_ts,
    T* col_chunk, T* edge_id, T* src_idx, T* chunk_ptr,
     T n, T m, T chunk_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= m) return;
    for (int i = tid; (T)i < m; i += blockDim.x * gridDim.x) {
        //printf("%d %d %lld\n", tid, i, m);
        Edge_T e = edges[i];
        //printf("%d %lld %lld %lld %lld %lld\n", i, m, e.src, e.dst, e.scc, e.dcc);
        T src_chunk = e.scc;
        T dst_chunk = e.dcc;
        if (e.src >= n || e.dst >= n) return;
        KERNEL_CHECK(e.src < n, "src out of range");
        KERNEL_CHECK(e.dst < n, "dst out of range");
        KERNEL_CHECK(src_chunk < chunk_size, "src_chunk out of range");
        KERNEL_CHECK(dst_chunk < chunk_size, "dst_chunk out of range");
        //原子操作更新 row_ptr 和 chunk_ptr
        //计算边的位置并填充数据
        col_idx[i] = e.dst;
        col_ts[i] = e.ts;
        edge_id[i] = e.eid;
        col_chunk[i] = dst_chunk;
        src_idx[i] = e.src;
        if (atomicCAS((ULL *)&row_ptr[e.src], (ULL)0, (ULL)0) == 0) {
            atomicAdd((ULL *)&chunk_ptr[src_chunk], (ULL)1);
        }
        atomicAdd((ULL *)&row_ptr[e.src], (ULL)1);
    }
}
template <typename T = int64_t, typename TS = int64_t>
__global__ void mark_chunks_exists_kernel(const T* chunks, T n, bool* exists, T size) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && chunks[i] < size) exists[chunks[i]] = true;
}

template <typename T = int64_t, typename TS = int64_t>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
template <typename T = int64_t, typename TS = int64_t>
__device__ __forceinline__ T warp_broadcast(T val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

__device__ int warpInclusiveScan(int val) {
    int lane_id = threadIdx.x % 32;
    // 向上遍历树结构
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) {
            val += n;
        }
    }
    return val;
}

__device__ int warpExclusiveScan(int val) {
    int lane_id = threadIdx.x % 32;
    int inclusive = warpInclusiveScan(val);
    // 将 inclusive 结果向右移动一个位置，lane0 从0开始
    int exclusive = __shfl_up_sync(0xFFFFFFFF, inclusive, 1);
    if (lane_id == 0) {
        exclusive = 0;
    }
    return exclusive;
}

__device__ int warpInclusiveMax(int val) {
    int lane_id = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) {
            val = max(val, n);
        }
    }
    return val;
}

__device__ int warpInclusiveMin(int val) {
    int lane_id = threadIdx.x % 32;
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (lane_id >= offset) {
            val = min(val, n);
        }
    }
    return val;
}

__device__ __forceinline__ int warp_lower_bound(const TS* base, int count, TS val) {
    int low = 0;
    int high = count;

    while (low < high) {
        int mid = (low + high) >> 1;
        TS value = base[mid];

        // 所有 lane 一起判断 value >= val
        unsigned active = __ballot_sync(0xffffffff, value >= val);

        // 如果至少有一个 lane 看到 >= val，则搜索区间缩到 [low, mid]
        // 否则缩到 [mid+1, high]
        if (active != 0) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return low;
}

__device__ __forceinline__ int warp_upper_bound(const TS* base, int count, TS val) {
    int low = 0;
    int high = count;

    while (low < high) {
        int mid = (low + high) >> 1;
        TS value = base[mid];

        // 所有 lane 一起判断 value >= val
        unsigned active = __ballot_sync(0xffffffff, value > val);

        // 如果至少有一个 lane 看到 >= val，则搜索区间缩到 [low, mid]
        // 否则缩到 [mid+1, high]
        if (active != 0) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return low;
}
// // ======================
// // 3. 阶段 1：统计边数
// // ======================
// template <typename T = int64_t, typename TS = int64_t>
// __global__ void count_edges_kernel(
//     const T* __restrict__ chunk_ptr,
//     const T* __restrict__ row_ptr,
//     const T* __restrict__ row_idx,
//     const TS* __restrict__ col_ts,
//     const T* __restrict__ col_chunk,
//     const T* __restrict__ chunks,
//     const bool* __restrict__ chunk_exists,
//     T num_chunks,
//     T num_nodes,
//     TS t_begin,
//     TS t_end,
    

//     // T* __restrict__ output_row_idx,
//     // T* __restrict__ output_row_ts,
//     // T* __restrict__ output_row_ts_counts,
//     // T* __restrict__ ts_count
//     T* __restrict__ counts_nodes_buffer,
//     T* __restrict__ counts_ts_buffer,
//     T* __restrict__ counts_col_buffer,
//     bool use_full_timestamps
// ) {
//     //每个block处理一个chunk,每个warp处理一个节点，每个线程处理每条边
//     int block_idx = blockIdx.x;
//     int global_warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
//     int warp_id = threadIdx.x / 32;
//     int warps_per_block = blockDim.x / 32;
//     int lane = threadIdx.x % 32;
//     counts_nodes_buffer[global_warp_id] = 0;
//     counts_ts_buffer[global_warp_id] = 0;
//     counts_col_buffer[global_warp_id] = 0;

//     while(block_idx < num_chunks){
//         T chunk = chunks[block_idx];
//         T chunk_start = chunk_ptr[block_idx];
//         T chunk_end = chunk_ptr[block_idx + 1];
//         if(chunk_start >= chunk_end){
//             block_idx += gridDim.x;
//             continue;
//         }
        
//         for(T row_x = chunk_start + warp_id; row_x < chunk_end ; row_x += warps_per_block){
//             T node_idx = row_idx[row_x];
//             if(node_idx >= num_nodes) break;

//             T row = node_idx;
//             T rs = row_ptr[node_idx];
//             T re = row_ptr[node_idx + 1];
            
//             if(rs >= re) continue;

//             if (rs >= re) continue;
//             //计算时间范围
//             T left  = lower_bound(col_ts, rs, re, t_begin);
//             T right = upper_bound(col_ts, left, re, t_end);
//             if (left >= right) continue;
//             if(lane == 0)
//                 counts_nodes_buffer[global_warp_id] += 1;
//             if(use_full_timestamps){
//                 if(lane == 0)   
//                     counts_ts_buffer[global_warp_id] += t_end - t_begin;
//             }
//             else{
//                 int local_count_ts = 0;
//                 for(T e = left + lane; e < right; e +=32){
//                     if(e > left && col_ts[e] != col_ts[e-1]){
//                         local_count_ts++;
//                     }
//                 }
//                 warp_reduce_sum(local_count_ts);
//                 if(lane == 0){
//                     counts_ts_buffer[global_warp_id] += local_count_ts;
//                 }
//             }
//             int local_count_col = 0;
//             for(T e = left + lane; e < right; e += 32) {
//                 if (e >= right) break;
//                 T dst_c = col_chunk[e];
//                 if (dst_c < num_nodes && chunk_exists[dst_c]) {
//                     local_count_col++;
//                 }
//             }
//             warp_reduce_sum(local_count_col);
//             if(lane == 0)
//                 counts_col_buffer[global_warp_id] += local_count_col;
//         }
//         block_idx += gridDim.x;
//     }
// }


// // ======================
// // 4. 阶段 2：写入数据
// // ======================
// template <typename T = int64_t, typename TS = int64_t>
// __global__ void write_edges_kernel(
//     const T* __restrict__ chunk_ptr,
//     const T* __restrict__ row_ptr,
//     const T* __restrict__ row_idx,
//     const T* __restrict__ col_idx,
//     const T* __restrict__ col_eid,
//     const TS* __restrict__ col_ts,
//     const T* __restrict__ col_chunk,
//     const T* __restrict__ chunks,
//     const bool* __restrict__ chunk_exists,
//     T* __restrict__ out_row_ptr,
//     T* __restrict__ out_row_idx,
//     T* __restrict__ out_row_ts,
//     T* __restrict__ out_ts_ptr,
//     T* __restrict__ out_col_idx,
//     TS* __restrict__ out_col_ts,
//     T * __restrict__ out_col_eid,
//     int num_chunks,
//     T num_nodes,
//     TS t_begin,
//     TS t_end,
//     T* __restrict__ counts_nodes_buffer,
//     T* __restrict__ counts_ts_buffer,
//     T* __restrict__ counts_col_buffer,
//     bool use_full_timestamps
// ) {
//     int block_idx = blockIdx.x;
//     int global_warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
//     while(block_idx < num_chunks){
//         T chunk = chunks[block_idx];
//         T chunk_start = chunk_ptr[block_idx];
//         T chunk_end = chunk_ptr[block_idx + 1];
//         if(chunk_start >= chunk_end){
//             block_idx += gridDim.x;
//             continue;
//         }
//         int warp_id = threadIdx.x / 32;
//         int warps_per_block = blockDim.x / 32;
//         int lane = threadIdx.x % 32;
//         int local_nodes_count = 0;
//         for(T row_x = chunk_start + warp_id; row_x < chunk_end ; row_x += warps_per_block){
//             T node_idx = row_idx[row_x];
//             if(node_idx >= num_nodes) break;

//             T row = node_idx;
//             T rs = row_ptr[node_idx];
//             T re = row_ptr[node_idx + 1];
            
//             if(rs >= re) continue;

//             if (rs >= re) continue;
//             //计算时间范围
//             T left  = lower_bound(col_ts, rs, re, t_begin);
//             T right = upper_bound(col_ts, left, re, t_end);
//             if (left >= right) continue;
//             if(lane == 0){
//                 //更新结束位置的ptr
//                 out_row_ptr[local_nodes_count + counts_nodes_buffer[global_warp_id] + 1]= \
//                     counts_ts_buffer[global_warp_id];
//                 out_row_idx[local_nodes_count + counts_nodes_buffer[global_warp_id]]= row;
//                 local_nodes_count++;
//             }
//             int offset_col_ = 0;
//             int offset_ts_ = 0;
//             if(use_full_timestamps){
//                 for(T e = t_begin; e < t_end ; e ++) {
//                    out_row_ts[counts_ts_buffer[global_warp_id] + e - t_begin] = e;
//                 }
//                 for(T e = left + lane; e < ((right + 31)/32 + 1) * 32; e +=32){
//                     bool active = e < right;
//                     bool dst_in_chunk = active ? chunk_exists[col_chunk[e]] : 0;
//                     bool new_timestamp = active ? (e==left || col_ts[e] != col_ts[e-1]) : 0;
//                     int offset_col_p = dst_in_chunk ? warpExclusiveScan(1): warpExclusiveScan(0);
//                     int offset_ts_p = new_timestamp ? warpExclusiveScan(1) : warpExclusiveScan(0);
//                     int incre_col = warp_broadcast(offset_col_p, 31);
//                     int incre_ts = warp_broadcast(offset_ts_p, 31);
//                     T dst_c = col_chunk[e];
//                     if(new_timestamp){
//                         int offset_ts = offset_ts_ + offset_ts_p;
//                         int offset_col = offset_col + offset_col_p;
//                         for(int i = col_ts[e-1] + 1; e > left && i < col_ts[e]; i++){
//                             out_ts_ptr[counts_ts_buffer[global_warp_id] + i - t_begin] = offset_col + counts_col_buffer[global_warp_id];
//                         }
//                         offset_ts_ += incre_ts;
//                     }
//                     if(dst_in_chunk){
//                             //out_ts_ptr[counts_ts_buffer[global_warp_id] + col_ts[e] - t_begin]++;
//                         int offset_col = offset_col_ + offset_col_p;
//                         out_col_idx[counts_col_buffer[global_warp_id] + offset_col] = col_idx[e];
//                         out_col_ts[counts_col_buffer[global_warp_id] + offset_col] = col_ts[e];
//                         out_col_eid[counts_col_buffer[global_warp_id] + offset_col] = col_eid[e];
//                         offset_col_ += incre_col;
//                     }
//                 }
//             }
//             else{
//                 for(T e = left + lane; e < ((right + 31)/32 + 1) * 32; e +=32){
//                     bool active = e < right;
//                     bool dst_in_chunk = active ? chunk_exists[col_chunk[e]] : 0;
//                     bool new_timestamp = active ? (e==left || col_ts[e] != col_ts[e-1]) : 0;
//                     int offset_col_p = dst_in_chunk ? warpInclusiveScan(1): warpInclusiveScan(0);
//                     int offset_ts_p = new_timestamp ? warpInclusiveScan(1) : warpInclusiveScan(0);
//                     int incre_col = warp_broadcast(offset_col_p, 31);
//                     int incre_ts = warp_broadcast(offset_ts_p, 31);
//                     T dst_c = col_chunk[e];
//                     int offset_col = offset_col_ + offset_col_p - dst_in_chunk;
//                     if(new_timestamp){
//                         int offset_ts = offset_ts_ + offset_ts_p - new_timestamp;
//                         out_ts_ptr[counts_ts_buffer[global_warp_id] + offset_ts] = offset_col + counts_col_buffer[global_warp_id];
//                         out_row_ts[counts_ts_buffer[global_warp_id] + offset_ts] = col_ts[e];
//                         offset_ts_ += incre_ts;
//                     }
//                     if(dst_in_chunk){
//                             //out_ts_ptr[counts_ts_buffer[global_warp_id] + col_ts[e] - t_begin]++;
//                         out_col_idx[counts_col_buffer[global_warp_id] + offset_col] = col_idx[e];
//                         out_col_ts[counts_col_buffer[global_warp_id] + offset_col] = col_ts[e];
//                         out_col_eid[counts_col_buffer[global_warp_id] + offset_col] = col_eid[e];
//                         offset_col_ += incre_col;
//                     }
//                 }
//             }
//         }
//         block_idx += gridDim.x;
//     }
 
// }

__global__ void filter_seeds_kernel(
    const T* chunk_ptr, const T* row_ptr, const T* row_idx, const T* col_idx, const TS* col_ts,
    const T* chunks, int num_chunks, TS t_begin, TS t_end, 
    T* __restrict__ counts_nodes_buffer,
    T* __restrict__ counts_ts_buffer,
    T* __restrict__ counts_col_buffer,
    bool using_full_timestamp,
    T* __restrict__ prefix_different_timestamp
){
    int block_idx = blockIdx.x;
    int global_warp_id = blockIdx.x * blockDim.x + threadIdx.x;//(blockDim.x / 32) + threadIdx.x / 32;
    int warp_id = threadIdx.x; // / 32;
    int warps_per_block = blockDim.x; /// 32;
    int lane = 0;//threadIdx.x % 32;
    int local_nodes_count = 0;
    counts_nodes_buffer[global_warp_id] = 0;
    counts_ts_buffer[global_warp_id] = 0;
    counts_col_buffer[global_warp_id] = 0;
    while(block_idx < num_chunks){  
        T chunk = chunks[block_idx];
        T chunk_start = chunk_ptr[block_idx];
        T chunk_end = chunk_ptr[block_idx + 1];
        if(chunk_start >= chunk_end){
            block_idx += gridDim.x;
            continue;
        }
        for(T row_x = chunk_start + warp_id; row_x < chunk_end ; row_x += warps_per_block){
            T node_idx = row_idx[row_x];
            T row = node_idx;
            T rs = row_ptr[node_idx];
            T re = row_ptr[node_idx + 1];
            if (rs >= re) continue;
            //计算时间范围
            T left  = lower_bound(col_ts, rs, re, t_begin);
            T right = upper_bound(col_ts, left, re, t_end);
            if (left >= right) continue;
            if(using_full_timestamp){
                counts_nodes_buffer [global_warp_id] += 1;
                counts_ts_buffer[global_warp_id] += t_end - t_begin;
                counts_col_buffer[global_warp_id] += right - left;
                
            }
            else{
                counts_nodes_buffer [global_warp_id] += 1;
                if(left == 0)
                    counts_ts_buffer[global_warp_id] += prefix_different_timestamp[right - 1];
                else{
                    counts_ts_buffer[global_warp_id] += prefix_different_timestamp[right - 1] - prefix_different_timestamp[left - 1];
                }
                counts_col_buffer[global_warp_id] += right - left;
            }
        }
        block_idx += gridDim.x;
    }
}

template <typename T = int64_t, typename TS = int64_t>
__global__ void collect_seeds_kernel( 
    const T* chunk_ptr, const T*chunks,
    const T* row_ptr, const T* row_idx, 
    const T* col_idx, const TS* col_ts, const T* col_eid,
    T* out_nodes, T* out_nodes_ptr, TS* out_ts, TS* out_ts_ptr,
    //奇数是正向边，偶数是反向边
    T* out_eid, int num_chunks, 
    TS t_begin, TS t_end,
    T* __restrict__ counts_nodes_buffer,
    T* __restrict__ counts_ts_buffer,
    T* __restrict__ counts_col_buffer,
    bool using_full_timestamp,
    T* __restrict__ prefix_different_timestamp

) {
    int block_idx = blockIdx.x;
    int global_warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int warp_id = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    int lane = threadIdx.x % 32;
    int local_nodes_count = 0;
    while(block_idx < num_chunks){  
        T chunk = chunks[block_idx];
        T chunk_start = chunk_ptr[block_idx];
        T chunk_end = chunk_ptr[block_idx + 1];
        if(chunk_start >= chunk_end){
            block_idx += gridDim.x;
            continue;
        extern __shared__ int shared_counts[];
        T* s_node_count = shared_counts;
        T* s_ts_count   = shared_counts + warps_per_block;
        T* s_eid_count  = shared_counts + warps_per_block * 2;
        for(T row_x = chunk_start + warp_id; row_x < chunk_end ; row_x += warps_per_block){
            //计算所属的前缀数组的下标
            int prefix_threads = (row_x - chunk_start) % blockDim.x;
            T node_idx = row_idx[row_x];
            T row = node_idx;
            T rs = row_ptr[node_idx];
            T re = row_ptr[node_idx + 1];
            if (rs >= re) continue;
            //计算时间范围
            T left  = lower_bound(col_ts, rs, re, t_begin);
            T right = upper_bound(col_ts, left, re, t_end);
            if (left >= right) continue;
            int thread_offset;
            T row_pos = counts_nodes_buffer[prefix_threads] + counts_nodes[thread_offset];
            T ts_pos = counts_ts_buffer[prefix_threads] + counts_ts[thread_offset];
            T eid_pos = counts_col_buffer[prefix_threads] + counts_eid[thread_offset];
            if(lane == 0){
                out_nodes_ptr[row_pos + 1] = ts_pos;
                out_nodes[row_pos] = row;
            }
            counts_node[thread_offset]++;
            out_nodes[]
            if(using_full_timestamp){
                //写入所有时间戳的边
                for(T e = t_start; e < t_end; e += 32){
                    out_ts[ts_pos + e - t_start] = e;
                }
                for(T e = left + lane; e < right; e +=32){
                    if(e == left || col_ts[e] != col_ts[e-1]){
                        for(int i = col_ts[e-1] + 1; i < col_ts[e]; i++){
                            out_ts_ptr[ts_pos + i - t_start] = eid_pos + e - left;
                        }
                    }
                    out_eid[eid_pos + e - left] = col_eid[e];
                }
                out_ts_ptr[ts_pos + t_end - t_start] = eid_pos + right - left;
                counts_ts[thread_offset] += t_end - t_start;
                counts_eid[thread_offset] += right - left;
                
            }
            else{
                int prefix_different_left = left == 0 ? 0: prefix_different_timestamp[left - 1];
                int local_count_ts = 0;
                for(T e = left + lane; e < right; e +=32){
                    if(e == left || col_ts[e] != col_ts[e-1]){
                        int t_offset = prefix_different_timestamp[e] - prefix_different_left;
                        out_ts[ts_pos + t_offset] = col_ts[e];
                        out_ts_ptr[ts_pos + t_offset] = eid_pos + e - left;
                    }
                    out_eid[eid_pos + e - left] = col_eid[e];
                }
                counts_ts[thread_offset] += prefix_different_timestamp[right] - prefix_different_left;
                counts_eid[thread_offset] += right - left;
            }
        }
        block_idx += gridDim.x;
    }
    
}
// // 第336行 bloom_set 完整实现
// __device__ void bloom_set(unsigned char* b, T x) {
//     unsigned h1 = (x * 11400714819323198485ULL) % BLOOM_BITS;
//     unsigned h2 = (x * 17498005710864076877ULL) % BLOOM_BITS;
//     unsigned h3 = (x * 14000237116378154321ULL) % BLOOM_BITS;
//     atomicOr((unsigned int*)&b[h1>>3], 1u << (h1&7));
//     atomicOr((unsigned int*)&b[h2>>3], 1u << (h2&7));
//     atomicOr((unsigned int*)&b[h3>>3], 1u << (h3&7));
// }


// __device__ bool bloom_test(const unsigned char* b, T x) {
//     unsigned h1 = (x * 11400714819323198485ULL) % BLOOM_BITS;
//     unsigned h2 = (x * 17498005710864076877ULL) % BLOOM_BITS;
//     unsigned h3 = (x * 14000237116378154321ULL) % BLOOM_BITS;
//     return (b[h1>>3] & (1 << (h1&7))) && (b[h2>>3] & (1 << (h2&7))) && (b[h3>>3] & (1 << (h3&7)));
// }
template <typename T = int64_t, typename TS = int64_t>
__global__ void static_recent_neighbor_num(
    const T* nodes, const T* nodes_ptr, const TS* node_ts,
    T num_nodes, T num_ts, int k, T allowed_offset, bool equal_root_time,
    const T* row_ptr, const T* col_idx, const TS* col_ts, const T* col_chunk, 
    T* start_offset, T* start_pos, T* end_pos
){
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = start; i < num_nodes; i+= blockDim.x * blockDim.x){
        if (i >= num_nodes) return;
        T node = nodes[i];
        int last_pos = 0;
        for(int t = nodes_ptr[i]; t < nodes_ptr[i+1]; t++){
            T ts = node_ts[t];
            T rs = row_ptr[node], re = row_ptr[node + 1];
            if (rs >= re) continue;
            T se;
            if(allowed_offset > -1){
                T se = lower_bound(col_ts, rs, re, node_ts[i] - allowed_offset);
            }
            else
                T se = 0;
            T te = lower_bound(col_ts, rs, re, node_ts[i] + equal_root_time);
            start_pos[t] = max(te - k,se);
            end_pos[t] = te;
            start_offset[t] = t == 0 ? 0: end_pos[t-1] - start_pos[t]; 
        }
        
    }
}

template <typename T = int64_t, typename TS = int64_t>
__global__ void recent_sample_single_hop_kernel(
    const T* nodes, const TS* node_ptr, const TS* node_ts,
    T num_nodes, T num_ts, const T* start_pos, const T* end_pos, const T* prefix,
    const T* row_ptr, const T* col_idx, const TS* col_ts, const TS* col_eid, const T* col_chunk, T* out_nbr, TS* out_ts, T* out_eid) {
        int start = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = 0; i < num_ts; i+= blockDim.x * blockDim.x){
            T pos = prefix[i];
            for(int j = 0; j < end_pos - start_pos; j++){
                T e = start_pos[i] + j;
                out_nbr[pos + j] = col_idx[e];
                out_ts[pos + j] = col_ts[e];
                out_eid[pos + j] = col_eid[e];
            }
        }
}
__global__ void static_neighbor_num(
    const T* nodes, const T* nodes_ptr, const TS* node_ts,
    T num_nodes, T num_ts, int k, T allowed_offset, bool equal_root_time,
    const T* row_ptr, const T* col_idx, const TS* col_ts, const T* col_chunk, 
    T* out_cnt, T* start_pos, T* end_pos
){
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = start; i < num_nodes; i+= blockDim.x * blockDim.x){
        if (i >= num_nodes) return;
        T node = nodes[i];
        for(int t = nodes_ptr[i]; t < nodes_ptr[i+1]; t++){
            T ts = node_ts[t];
            T rs = row_ptr[node], re = row_ptr[node + 1];
            T se;
            if (rs >= re) continue;
            if(allowed_offset > -1){
                se = lower_bound(col_ts, rs, re, node_ts[i] - allowed_offset);
            }
            else
                se = 0;
            T te = lower_bound(col_ts, rs, re, node_ts[i] + equal_root_time);
            out_cnt[t] = min((int)(te - se),k);
            start_pos[t] = se;
            end_pos[t] = te;
        }
        
    }
}

template <typename T = int64_t, typename TS = int64_t>
__global__ void uniform_sample_single_hop_kernel(
     const T* nodes, const T* nodes_ptr, const TS* node_ts,
    T num_nodes, T num_ts, int k, const T* prefix,
    const T* row_ptr, const T* col_idx, const TS* col_ts, const TS* col_eid, const T* col_chunk, T* out_nbr, TS* out_ts, T* out_eid,
    T* start_pos, T* end_pos, bool keep_root_time
) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(1317, threadIdx.x, 0, &state); // 初始化状态



    for(int i = start; i < num_nodes; i+= blockDim.x * blockDim.x){
        if (i >= num_nodes) return;
        T node = nodes[i];
        for(int t = nodes_ptr[i]; t < nodes_ptr[i+1]; t++){
            T ts = start_pos[t];
            T rs = end_pos[t];
            if (rs >= ts) continue;
            T pos = prefix[t];
            for(int j = 0; j < k; j++){
                T rand_idx = (int)(curand_uniform(&state) * (rs - ts));
                T e = rand_idx + ts;
                out_nbr[pos + j] = col_idx[e];
                out_ts[pos + j] = keep_root_time?col_ts[e]:node_ts[t];
                out_eid[pos + j] = col_eid[e];
            }    
        }
    }
}

// // ------------------------------------------------------------------
// // 3. 内核前置声明
// // ------------------------------------------------------------------
// template <typename T = int64_t, typename TS = int64_t>
// __global__ void build_graph_kernel(
//     const Edge<T, TS>* edges, T* row_ptr, T* col_idx, TS* col_ts,
//     T* col_chunk, T* edge_id, T* src_idx, T* chunk_ptr,
//     const T* row_chunk_mapper, T n, T m, T chunk_size);

// template <typename T = int64_t, typename TS = int64_t>
// __global__ void mark_chunks_exists_kernel(const T* chunks, T n, bool* exists, T size);

// template <typename T = int64_t, typename TS = int64_t>
// __global__ void count_edges_kernel(
//     const T* chunk_ptr, const T* row_ptr, const T* row_idx, const TS* col_ts,
//     const T* col_chunk, const T* chunks, bool* chunk_exists, 
//     T num_chunks, T num_nodes, TS t_begin, TS t_end,
//     T* warp_counts_buffer, T* counts_buffer, T* chunk_nodes_counts_buffer
// );

// template <typename T = int64_t, typename TS = int64_t>
// __global__ void write_edges_kernel(
//     const T*  chunk_ptr, 
//     const T* row_ptr,
//     const T* row_idx,
//     const T* col_idx,
//     const T* col_eid,
//     const TS* col_ts,
//     const T* col_chunk,
//     const T* chunks,
//     const bool* chunk_exists, 
//     T* out_row_ptr, 
//     T* out_row_idx, 
//     T* out_col_idx, 
//     TS* out_col_ts, T * out_col_eid, T num_chunks, T num_nodes,TS t_begin,TS t_end,
//     const T*  chunk_counts_buffer,
//     const T*  counts_buffer,
//     const T*  chunk_nodes_counts_buffer
// );


// template <typename T = int64_t, typename TS = int64_t>
// __global__ void filter_seeds_kernel(
//     const T* chunk_ptr, const T* row_ptr, const T* row_idx,
//     const T* col_idx, const TS* col_ts,
//     const T* chunks, T num_chunks,
//     TS t_begin, TS t_end,
//     T* out_counts,
//     T* chunk_counts_buffer);

// template <typename T = int64_t, typename TS = int64_t>
// __global__ void collect_seeds_kernel(
//     const T* prefix,
//     const T* chunk_ptr, const T* chunks, const T* row_ptr, const T* row_idx,
//     const T* col_idx, const TS* col_ts, const T* col_eid,
//     T* out_nodes, TS* out_ts, T* out_eid,
//     T num_chunks, T total,
//     T neg_size,
//     const T* neg_seeds,
//     const T* chunk_counts_prefix,
//     TS t_begin, TS t_end
// );
// template <typename T = int64_t, typename TS = int64_t>
// __global__ void static_neighbor_num(
//     const T* nodes, const TS* node_ts, T num_nodes, int k,
//     const T* row_ptr, const T* col_idx, const TS* col_ts, const T* col_chunk, T* out_cnt, const bool* chunk_exists
// );
// template <typename T = int64_t, typename TS = int64_t>
// __global__ void sample_single_hop_kernel(
//     const T* in_nodes, const TS* in_ts,
//     //const T* in_nodes, const TS* in_ts,
//     T n, int k,
//     const T* row_ptr, const T* col_idx, const TS* col_ts,
//     const T* col_eid, const T* col_chunk,
//     TS time_begin, TS time_end,
//     T* out_row_ptr, T* out_nbr, TS* out_nbr_ts, T* out_nbr_eid,
//     const bool* chunk_exists
// );



// ------------------------------------------------------------------
// 4. 图结构
// ------------------------------------------------------------------
struct KeyValue {
    T  idx;
    TS ts;
    __host__ __device__ KeyValue(T i, TS t) : idx(i), ts(t) {}
};

// 自定义比较器：先按 idx 排序，再按 ts
struct CompareKeyValue {
    __host__ __device__ bool operator()(const KeyValue& a, const KeyValue& b) const {
        if (a.idx != b.idx) return a.idx < b.idx;
        return a.ts < b.ts;
    }
};

// 自定义相等性：用于 unique
struct EqualKeyValue {
    __host__ __device__ bool operator()(const KeyValue& a, const KeyValue& b) const {
        return a.idx == b.idx && a.ts == b.ts;
    }
};
template <typename T = int64_t, typename TS = int64_t>
class TemporalRoot{
    public:
    thrust::device_vector<T>  roots;
    thrust::device_vector<T> ts_ptr;
    thrust::device_vector<TS> ts;
    thrust::device_vector<T> q_ptr;
    thrust::device_vector<T> q_eid;
    int root_num;
    int root_ts_num;
    TemporalRoot() = default;
    TemporalRoot(int r_num, int rt_num){
        root_num = r_num;
        root_ts_num = rt_num;
        roots.resize(r_num);
        ts_ptr.resize(r_num + 1);
        ts.resize(rt_num);
    }
    TemporalRoot(thrust::device_vector<T>&& r, thrust::device_vector<T>&& tp, thrust::device_vector<TS>&& t)
        : roots(std::move(r)), ts_ptr(std::move(tp)), ts(std::move(t)){ root_num = r.size(); root_ts_num = t.size();}
    TemporalRoot(thrust::device_vector<T>&& r, thrust::device_vector<T>&& tp, thrust::device_vector<TS>&& t, thrust::device_vector<T>&& qp, thrust::device_vector<T>&& qe)
        : roots(std::move(r)), ts_ptr(std::move(tp)), ts(std::move(t)), q_ptr(std::move(qp)), q_eid(std::move(qe)){ root_num = r.size(); root_ts_num = t.size();}
};
template <typename T = int64_t, typename TS = int64_t>
class NegativeRoot{
    public:
    thrust::device_vector<T>  roots;
    thrust::device_vector<TS> ts;
    int root_num;
    NegativeRoot() = default;
    NegativeRoot(thrust::device_vector<T>&& r, thrust::device_vector<TS>&& t)
        : roots(std::move(r)), ts(std::move(t)){}
};
template <typename T = int64_t, typename TS = int64_t>
class TemporalNeighbor{
    public:
    thrust::device_vector<T> root_start_ptr;
    thrust::device_vector<TS> root_end_ptr;

    thrust::device_vector<T>  neighbors;
    thrust::device_vector<T> neighbors_ts;
    thrust::device_vector<T> neighbors_eid;

    int neighbor_num;
    
    TemporalNeighbor() = default;
    TemporalNeighbor(int root_num, int num){
        neighbor_num = num;
        root_start_ptr.resize(root_num + 1);
        root_end_ptr.resize(root_num + 1);
        neighbors.resize(num);
        neighbors_ts.resize(num);
        neighbors_eid.resize(num);
    }
    TemporalNeighbor(thrust::device_vector<T>&& rsp,
                     thrust::device_vector<T>&& nbr,
                     thrust::device_vector<T>&& nbr_ts,
                     thrust::device_vector<T>&& nbr_eid)
        : root_start_ptr(std::move(rsp)),
          neighbors(std::move(nbr)), neighbors_ts(std::move(nbr_ts)), neighbors_eid(std::move(nbr_eid)){}
    TemporalNeighbor(thrust::device_vector<T>&& rsp, thrust::device_vector<T>&& rep,
                     thrust::device_vector<T>&& nbr,
                     thrust::device_vector<T>&& nbr_ts,
                     thrust::device_vector<T>&& nbr_eid)
        : root_start_ptr(std::move(rsp)), root_end_ptr(std::move(rep)),
          neighbors(std::move(nbr)), neighbors_ts(std::move(nbr_ts)), neighbors_eid(std::move(nbr_eid)){}
};
template <typename T = int64_t, typename TS = int64_t>
class TemporalResult{
    public:
    TemporalRoot<T,TS> roots;
    NegativeRoot<T,TS> neg_roots;
    std::vector<TemporalNeighbor<T,TS>> neighbors_list;
    thrust::device_vector<T>  nodes_remapper_id;
    //thrust::device_vector<T> edges_remapper_id;
    TemporalResult() = default;
    TemporalResult(const TemporalRoot<T,TS>& r)
        : roots(r){}
    TemporalResult(TemporalRoot<T,TS>&& r, std::vector<TemporalNeighbor<T,TS>>&& n)
        : roots(std::move(r)), neighbors_list(std::move(n)){}   
    void append(TemporalNeighbor<T,TS>&& n){
        neighbors_list.push_back(std::move(n));
    }
};
template <typename T = int64_t, typename TS = int64_t>
class TemporalBlock {

    public:
    thrust::device_vector<T>  neighbors;
    thrust::device_vector<TS> neighbors_ts;
    thrust::device_vector<T>  neighbors_eid;

    thrust::device_vector<T>  row_ptr;
    thrust::device_vector<T>  row_idx;
    thrust::device_vector<T> row_ts;
    T row_node_size;
    int layer = 0;
    thrust::device_vector<T> eid_mapper;
    thrust::device_vector<T> nid_mapper;

    thrust::device_vector<T> layer_ptr;
    thrust::device_vector<T> seed_eid;
    
    thrust::device_vector<T> row_ptr_to_ts;
    thrust::device_vector<T> unique_ts;
    thrust::device_vector<T> row_ts_ptr;

    TemporalBlock() = default;
    TemporalBlock(thrust::device_vector<T>&& nbr,
                  thrust::device_vector<TS>&& nbr_ts,
                  thrust::device_vector<T> && nbr_eid,
                  thrust::device_vector<T>&& rp)
        : neighbors(std::move(nbr)), 
        neighbors_ts(std::move(nbr_ts)), 
        neighbors_eid(std::move(nbr_eid)), 
        row_ptr(std::move(rp)){}

    TemporalBlock(
                 thrust::device_vector<T>&& nbr,
                  thrust::device_vector<TS>&& nbr_ts,
                  thrust::device_vector<T> && nbr_eid,
                  thrust::device_vector<T> && rp,
                  thrust::device_vector<T> && rdx,
                  int layer_)
        : neighbors(std::move(nbr)), 
        neighbors_ts(std::move(nbr_ts)), 
        neighbors_eid(std::move(nbr_eid)), 
        row_ptr(std::move(rp)),
        row_idx(std::move(rdx)),layer(layer_){}

    TemporalBlock(thrust::device_vector<T>&& nbr,
                  thrust::device_vector<TS>&& nbr_ts,
                  thrust::device_vector<T> && nbr_eid,
                  thrust::device_vector<T>&& rp,
                  thrust::device_vector<T>&& lp)
        : neighbors(std::move(nbr)), 
        neighbors_ts(std::move(nbr_ts)), 
        neighbors_eid(std::move(nbr_eid)), 
        layer_ptr(std::move(lp)),
        row_ptr(std::move(rp)){}
    

    // void duplicate_same_time(){
    //     size_t n = row_idx.size();
    //     if (n == 0) return;
    //     // Step 1: 构造键值对
    //     thrust::device_vector<KeyValue> kv(n);
    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(row_idx.begin(), row_ts.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(row_idx.end(),   row_ts.end())),
    //         kv.begin(),
    //         [] __host__ __device__ (const thrust::tuple<T, TS>& t) {
    //             return KeyValue(thrust::get<0>(t), thrust::get<1>(t));
    //         }
    //     );
    //     // Step 3: 去重 + 标记变化
    //     thrust::device_vector<int> flags(n, 0);
    //     auto kv_iter = thrust::make_zip_iterator(thrust::make_tuple(kv.begin(), kv.begin() + 1));
    //     thrust::transform(
    //         kv_iter, kv_iter + (n - 1),
    //         flags.begin() + 1,
    //         [] __host__ __device__ (const thrust::tuple<KeyValue, KeyValue>& pair) {
    //             return EqualKeyValue()(thrust::get<0>(pair), thrust::get<1>(pair)) ? 0 : 1;
    //         }
    //     );
    //     thrust::device_vector<T> ptr = thrust::sequence(0, n + 1);
    //     unique_ts.resize(n);
    //     flags[0] = 1;  // 第一个元素总是新组
    //     auto end = thrust::copy_if(
    //         row_ts.begin(), row_ts.end(), flags, unique_ts
    //     );
    //     unique_ts.resize(end - unique_ts.begin());
    //     size_t unique_ts_count = unique_ts.size();

    //     row_ts_ptr.resize(unique_ts_count + 1);
    //     thrust::copy_if(
    //         ptr.begin(), ptr.end(), flags, row_ts_ptr.begin()
    //     );
    //     row_ts_ptr[unique_ts_count] = n;

    //     row_ts_ptr.resize(n + 1);

    //     row

    //     thrust::transform(
    //         thrust::make_zip_iterator(thrust::make_tuple(row_ts.begin(), flags.begin())),
    //         thrust::make_zip_iterator(thrust::make_tuple(row_ts.end(),   flags.end())),
    //         row_ts_ptr.begin(),
    //         [] __host__ __device__ (const thrust::tuple<TS, int>& t) {
    //             return thrust::get<1>(t);
    //         }
    //     );
    //     size_t unique_count = unique_ts.size();
    //     ts_start.resize(unique_count);
    //     ts_end.resize(unique_count);
    //     end = thrust::copy_if(
    //         ptr.begin(), ptr.end(), flags, row_ptr_to_ts
    //     );
    //     row_ptr_to_ts[]
    //     // 扫描 flags，得到每个 row_idx 的起始位置
    //     thrust::device_vector<T> group_start(n + 1);
    //     group_start[0] = 0;
    //     thrust::inclusive_scan(flags.begin(), flags.end(), group_start.begin() + 1);

    //     // 统计每个组的大小
    //     size_t num_groups = group_start[n];
    //     unique_counts.resize(num_groups);

    //     // 计算每组长度：group_start[i+1] - group_start[i]
    //     thrust::transform(
    //         group_start.begin(), group_start.begin() + num_groups,
    //         group_start.begin() + 1,
    //         unique_counts.begin(),
    //         [] __host__ __device__ (T a, T b) { return b - a; }
    //     );
    // }
    //TODO：implement unique1
    // TemporalBlock(thrust::device_vector<T> & row,
    //                 thrust::device_vector<T> & col,
    //                 thrust::device_vector<TS> & ts,
    //                 thrust::device_vector<TS> & eid,
    //                 thrust::device_vector<T> &layer_ptr)
    // {
    //     neighbors = std::move(col);
    //     neighbors_ts = std::move(ts);
    //     neighbors_eid = std::move(eid);
    //     //nid_mapper = thrust::device_vector<T>(col.size());
    //     for (T i = 0; i < layer_ptr.size(); i++) {
    //         T start = layer_ptr[i];
    //         T end = layer_ptr[i + 1];
    //         auto end = thrust::reduce_by_key(
    //             row.begin(), row.end(),
    //             thrust::make_constant_iterator(1), // 每个元素的初始计数为 1
    //             nid_mapper.begin(), // 输出去重后的元素
    //         thrust::make_counting_iterator(0)） // 输出的计数器
    //     )
        
    //     T row_node_size = end - nid_mapper.begin();  
    //     nip_mapper.resize(row_node_size + col.size());
    //     thrust::copy(col.begin(), col.end(), nid_mapper.begin() + row_node_size);
        //nid_mapper.resize(unique_node_size);
    //}
    const thrust::device_vector<T>&  get_neighbors()    const { return neighbors; }
    const thrust::device_vector<TS>& get_neighbors_ts() const { return neighbors_ts; }
    const thrust::device_vector<T>&  get_row_ptr()      const { return row_ptr; }
    const thrust::device_vector<T>&  get_neighbors_eid()     const { return neighbors_eid; }
};

template <typename T = int64_t, typename TS = int64_t>
class AsyncQueueRes{
    public:
    queue<torch::Tensor> insert_idx;
    queue<TemporalResult> res_list;
    queue<bool> follow;
    void top_and_pop(){
        bool op = follow.front();
        follow.pop();
        if(op){
            TemporalResult = res_list.top();
            res_list.pop();
        }
        else{

        }
    }
    void insert(){

    }
    private:
    torch::Tensor clear_mapper;
    void clear_id_mapper(){

    }
    void get_unique_id_mapper(){

    }
}

template <typename T = int64_t, typename TS = int64_t>
class Graph {
    thrust::device_vector<T> slice_ptr;
    thrust::device_vector<T>  chunk_ptr_, row_ptr_, row_idx_, col_idx_, col_chunk_, edge_id_, src_idx_;
    thrust::device_vector<TS> col_ts_;
    T  num_nodes_, num_edges_, chunk_size_;
    cudaStream_t stream_;
    thrust::device_vector<T> counts_buffer;
    thrust::device_vector<T> chunk_counts_buffer;
    thrust::device_vector<T> chunk_nodes_counts_buffer;

    thrust::device_vector<T> counts_nodes_buffer;
    thrust::device_vector<T> counts_ts_buffer;
    thrust::device_vector<T> counts_col_buffer;

    thrust::device_vector<T> prefix_different_timestamp;
    public:
    Graph(T n, T chunk_size, const torch::Tensor& src, const torch::Tensor& dst,
        const torch::Tensor& ts, const torch::Tensor& row_chunk_mapper,
        const torch::Tensor & slice_ptr,
        uint64_t py_stream)
        : num_nodes_(n), chunk_size_(chunk_size)
    {
        try {
            //printf("Building graph with %ld nodes and chunk size %ld...\n", n, chunk_size);
            // ==============================================================
            // 1. 基础检查：CUDA + 类型
            // ==============================================================
            TORCH_CHECK(src.is_cuda() && dst.is_cuda() && ts.is_cuda() , !row_chunk_mapper.is_cuda(),
                        "All input tensors (src, dst, ts must be CUDA tensors. row_chunk_mapper must be CPU tensor. ",
                        "Got src.device=", src.device(), ", dst.device=", dst.device(),
                        ", ts.device=", ts.device(), ", mapper.device=", row_chunk_mapper.device());

            TORCH_CHECK(src.scalar_type() == torch::kInt64, 
                        "src must be int64 (torch.long), got ", src.scalar_type());
            TORCH_CHECK(dst.scalar_type() == torch::kInt64, 
                        "dst must be int64 (torch.long), got ", dst.scalar_type());
            TORCH_CHECK(ts.scalar_type() == torch::kInt64, 
                        "ts must be int64 (torch.long), got ", ts.scalar_type());
            TORCH_CHECK(row_chunk_mapper.scalar_type() == torch::kInt64,
                        "row_chunk_mapper must be int64 (torch.long), got ", row_chunk_mapper.scalar_type());

            // ==============================================================
            // 2. 边数一致性
            // ==============================================================
            T m = src.size(0);
            TORCH_CHECK(dst.size(0) == m, "src and dst must have same size, got ", m, " vs ", dst.size(0));
            TORCH_CHECK(ts.size(0) == m,  "src and ts must have same size, got ", m, " vs ", ts.size(0));
            num_edges_ = m;
            //printf("Graph num_nodes=%ld, num_edges=%ld, chunk_size=%ld\n", num_nodes_, num_edges_, chunk_size_);
            // ==============================================================
            // 3. CUDA Stream
            // ==============================================================
            stream_ = reinterpret_cast<cudaStream_t>(py_stream);
            if (stream_ == nullptr) {
                stream_ = at::cuda::getCurrentCUDAStream();
            }
            initialize_kernel_error(stream_);
            // ==============================================================
            // 4. 拷贝到 Host（带边界检查）
            // ==============================================================
            auto src_cpu = src.cpu().contiguous();
            auto dst_cpu = dst.cpu().contiguous();
            auto ts_cpu  = ts.cpu().contiguous();

            const T*  s_ptr = src_cpu.data_ptr<T>();
            const T*  d_ptr = dst_cpu.data_ptr<T>();
            const TS* t_ptr = ts_cpu.data_ptr<TS>();

            // 边界检查：节点 ID 不能超过 num_nodes_
            for (T i = 0; i < m; ++i) {
                TORCH_CHECK(s_ptr[i] >= 0 && s_ptr[i] < n,
                            "src[", i, "] = ", s_ptr[i], " out of range [0, ", n, ")");
                TORCH_CHECK(d_ptr[i] >= 0 && d_ptr[i] < n,
                            "dst[", i, "] = ", d_ptr[i], " out of range [0, ", n, ")");
            }
            //printf("Input edges copied to host and validated.\n");
            // ==============================================================
            // 5. 构建 Host 边列表
            // ==============================================================
            std::vector<Edge_T> h_edges(m);
            T* d_mapper = row_chunk_mapper.data_ptr<T>();
            for (T i = 0; i < m; ++i) {
                T s = s_ptr[i], d = d_ptr[i];
                T s_chunk = d_mapper[s];
                T d_chunk = d_mapper[d];
                TORCH_CHECK(s_chunk >= 0 && s_chunk < chunk_size,
                            "src chunk id ", s_chunk, " out of range [0, ", chunk_size, ")");
                TORCH_CHECK(d_chunk >= 0 && d_chunk < chunk_size,
                            "dst chunk id ", d_chunk, " out of range [0, ", chunk_size, ")");
                h_edges[i] = Edge_T(s, d, t_ptr[i], i, s_chunk, d_chunk);
                ////printf("%d %d %d %d %d %d\n", s, d, t_ptr[i], i, s_chunk, d_chunk);
            }
            //printf("Edge list constructed on host.\n");
            // ==============================================================
            // 6. 拷贝到 Device + 排序
            // ==============================================================
            thrust::device_vector<Edge_T> d_edges(m);  // 先分配
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_edges.data()),
                h_edges.data(),
                m * sizeof(Edge_T),
                cudaMemcpyHostToDevice,
                stream_);
            cudaStreamSynchronize(stream_);  // 确保拷贝完成！
            //printf("Edges copied to device and sorted.\n");
            // ==============================================================
            // 7. 初始化 CSR 结构
            // ==============================================================
            row_ptr_.resize(n + 1, 0);
            row_idx_.resize(n);
            col_idx_.resize(m);
            col_ts_.resize(m);
            col_chunk_.resize(m);
            edge_id_.resize(m);
            src_idx_.resize(m);
            chunk_ptr_.resize(chunk_size_ + 1, 0);
            //printf("CSR structures initialized.  %d \n", col_idx_.size());
            // ==============================================================
            // 8. 启动 Kernel
            // ==============================================================
            dim3 block(10), grid( 10);
            //dim3 block(256), grid((m + 255) / 256);
            //printf("%d chunk_size\n", chunk_size_);
            build_graph_kernel<<<grid, block, 0, stream_>>>(
                thrust::raw_pointer_cast(d_edges.data()),
                thrust::raw_pointer_cast(row_ptr_.data()),
                thrust::raw_pointer_cast(col_idx_.data()),
                thrust::raw_pointer_cast(col_ts_.data()),
                thrust::raw_pointer_cast(col_chunk_.data()),
                thrust::raw_pointer_cast(edge_id_.data()),
                thrust::raw_pointer_cast(src_idx_.data()),
                thrust::raw_pointer_cast(chunk_ptr_.data()),
                n, m, chunk_size_);
            
            //printf("Graph construction kernel launched.\n");
            // ==============================================================
            // 9. 同步 + 错误检查
            // ==============================================================
            cudaError_t err = cudaStreamSynchronize(stream_);
            CUDA_CHECK(cudaGetLastError());
            KernelError h_error;
            cudaMemcpyFromSymbol(&h_error, g_kernel_error, sizeof(KernelError), 0, cudaMemcpyDeviceToHost);

            if (h_error.code != cudaSuccess) {
                std::ostringstream oss;
                oss << "Kernel error at " << h_error.file << ":" << h_error.line
                    << " - " << h_error.msg
                    << " (code: " << h_error.code << ")";
                throw std::runtime_error(oss.str());
            }
            TORCH_CHECK(err == cudaSuccess,
                        "CUDA kernel launch failed: ", cudaGetErrorString(err));
            //printf("Graph construction kernel finished.\n");
            // ==============================================================
            // 10. 后处理：scan + unique
            // ==============================================================
            row_idx_.resize(num_nodes_);

            thrust::exclusive_scan(thrust::cuda::par.on(stream_), 
                               row_ptr_.begin() , row_ptr_.end(), row_ptr_.begin());
            thrust::exclusive_scan(thrust::cuda::par.on(stream_), 
                               chunk_ptr_.begin(), chunk_ptr_.end(), chunk_ptr_.begin());
            thrust::device_vector<T> unique_src(src_idx_);
            auto new_end = thrust::unique(thrust::cuda::par.on(stream_),
                                       unique_src.begin(), unique_src.end());
            row_idx_.resize(thrust::distance(unique_src.begin(), new_end));
            thrust::copy(thrust::cuda::par.on(stream_),
                       unique_src.begin(), new_end, row_idx_.begin());
            prefix_different_timestamp.resize(m,0);
            thrust::transform(d_edges.begin(),d_edges.end()-1,
                              d_edges.begin()+1,
                               prefix_different_timestamp.begin()+1,
                              [] __host__ __device__ (Edge_T a, Edge_T b) {
                                  return (a.src != b.src || a.ts != b.ts) ? 1 : 0;
                              });
            prefix_different_timestamp[0] = 1;
            thrust::inclusive_scan(thrust::cuda::par.on(stream_),
                                   prefix_different_timestamp.begin(),
                                   prefix_different_timestamp.end(),
                                   prefix_different_timestamp.begin());
            // ==============================================================
            // 11. 初始化计数缓冲
            // ==============================================================
            counts_buffer.resize(num_nodes_, 0);
            chunk_counts_buffer.resize(chunk_size_ + 1, 0);
            chunk_nodes_counts_buffer.resize(chunk_size_ + 1, 0);
            
            counts_nodes_buffer.resize(num_nodes_, 0);
            counts_ts_buffer.resize(num_nodes_, 0);
            counts_col_buffer.resize(num_nodes_, 0);
        }
        catch (const c10::Error& e) {
            throw;  // 让 PyTorch 捕获，打印完整堆栈
        }
        catch (const std::exception& e) {
            throw;
            //TORCH_CHECK(false, "Graph construction failed: ", e.what());
        }
        catch (...) {
            throw;
            //TORCH_CHECK(false, "Graph construction failed with unknown error");
        }
    }
//     // --------------------------------------------------------------
//     // 返回结构（多跳采样）
//     // --------------------------------------------------------------
    TemporalResult<T, TS> slice_by_chunk_ts(
        const thrust::device_vector<T>& chunks,
        TS t_begin, TS t_end, bool use_full_timestamps,
        cudaStream_t stream = nullptr
    ) {
        thrust::device_vector<bool> chunk_exists(chunk_size_, false);
        thrust::device_vector<T> sorted_chunks = chunks;
        stream = stream_;
        thrust::sort(thrust::cuda::par.on(stream), sorted_chunks.begin(), sorted_chunks.end());
        mark_chunks_exists_kernel<<<(sorted_chunks.size() + 255)/256, 256, 0, stream>>>(
            thrust::raw_pointer_cast(sorted_chunks.data()), (T)sorted_chunks.size(),
            thrust::raw_pointer_cast(chunk_exists.data()), chunk_size_);
        // 2. 阶段 1：统计
        int warps_per_block = 8;           // 每个 block 中的 warp 数，可调（1..8）
        int threads_per_block = warps_per_block * 32;
        size_t chunks_n = chunks.size();
        int grid_x = (int)chunks_n;        // 每个 block 一个 chunk
        if (grid_x < 1) grid_x = 1;

        dim3 block(threads_per_block);
        dim3 grid(grid_x);
        thrust::device_vector<T> d_out_counts(num_nodes_, 0);
        thrust::fill(d_out_counts.begin(), d_out_counts.end(), 0);
        count_edges_kernel<<<grid, block, 0, stream>>>(
            thrust::raw_pointer_cast(chunk_ptr_.data()),
            thrust::raw_pointer_cast(row_ptr_.data()),
            thrust::raw_pointer_cast(row_idx_.data()),
            thrust::raw_pointer_cast(col_ts_.data()),
            thrust::raw_pointer_cast(col_chunk_.data()),
            thrust::raw_pointer_cast(chunks.data()),
            thrust::raw_pointer_cast(chunk_exists.data()),
            (T)chunks.size(), num_nodes_, t_begin, t_end,
            thrust::raw_pointer_cast(counts_nodes_buffer.data()),
            thrust::raw_pointer_cast(counts_ts_buffer.data()),
            thrust::raw_pointer_cast(counts_col_buffer.data()),
            use_full_timestamps

        );

        // 3. 计算全局 row_ptr
        int total_warps = warps_per_block * grid_x;
        thrust::device_vector<T> prefix_node_count(total_warps + 1, 0);
        thrust::device_vector<T> prefix_ts_count(total_warps + 1, 0);
        thrust::device_vector<T> prefix_col_count(total_warps + 1, 0);
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            counts_nodes_buffer.begin(), counts_nodes_buffer.begin() + total_warps + 1,
            prefix_node_count.begin()
        );
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            counts_ts_buffer.begin(), counts_ts_buffer.begin() + total_warps + 1,
            prefix_ts_count.begin()
        );
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            counts_col_buffer.begin(), counts_col_buffer.begin() + total_warps + 1,
            prefix_col_count.begin()
        );
        thrust::device_vector<T> out_row_ptr(prefix_node_count.back()+1);
        thrust::device_vector<T> out_row_idx(prefix_node_count.back()+1);
        thrust::device_vector<T> out_ts_ptr(prefix_ts_count.back()+1);
         thrust::device_vector<T> out_row_ts(prefix_ts_count.back()+1);
        thrust::device_vector<T> out_col_idx(prefix_col_count.back());
        thrust::device_vector<TS> out_col_ts(prefix_col_count.back());
        thrust::device_vector<T> out_col_eid(prefix_col_count.back());
        // 4. 阶段 2：写入
        int query_chunk_size = chunks.size();
        write_edges_kernel<<<grid, block, 0, stream>>>(
                       thrust::raw_pointer_cast(chunk_ptr_.data()),
                       thrust::raw_pointer_cast(row_ptr_.data()),
                       thrust::raw_pointer_cast(row_idx_.data()),
                       thrust::raw_pointer_cast(col_idx_.data()),
                       thrust::raw_pointer_cast(edge_id_.data()),      // <-- 正确顺序
                       thrust::raw_pointer_cast(col_ts_.data()),
                       thrust::raw_pointer_cast(col_chunk_.data()),
                       thrust::raw_pointer_cast(chunks.data()),
                       thrust::raw_pointer_cast(chunk_exists.data()),
                       thrust::raw_pointer_cast(out_row_ptr.data()),
                       thrust::raw_pointer_cast(out_row_idx.data()),
                       thrust::raw_pointer_cast(out_row_ts.data()),
                       thrust::raw_pointer_cast(out_ts_ptr.data()),
                       thrust::raw_pointer_cast(out_col_idx.data()),
                       thrust::raw_pointer_cast(out_col_ts.data()),
                       thrust::raw_pointer_cast(out_col_eid.data()),
                       query_chunk_size, num_nodes_, t_begin, t_end,
                          thrust::raw_pointer_cast(counts_nodes_buffer.data()),
                            thrust::raw_pointer_cast(counts_ts_buffer.data()),
                            thrust::raw_pointer_cast(counts_col_buffer.data()),
                          use_full_timestamps
                   );
        // 5. 构造返回
        TemporalRoot<T, TS> block_root(
            std::move(out_row_idx),
            std::move(out_row_ptr),
            std::move(out_row_ts)

        );
        TemporalNeighbor<T, TS> block_nbr(
            std::move(out_ts_ptr),
            std::move(out_col_idx),
            std::move(out_col_ts),
            std::move(out_col_eid)
        );
        TemporalResult<T, TS> block_result(
            std::move(block_root),
            std::vector<TemporalNeighbor<T, TS>>{std::move(block_nbr)}
        );
        return std::move(block_result);
    }
    TemporalRoot<T,TS> get_seeds_in_chunks(const thrust::device_vector<T>& chunks,
                             TS t_begin, TS t_end,
                             bool use_full_timestamps
                             ) {

        int warps = 4, threads = warps * 32;
        int blocks = (chunks.size() + warps - 1) / warps;
        filter_seeds_kernel<<<blocks, threads, 0, stream_>>>(
            thrust::raw_pointer_cast(chunk_ptr_.data()),
            thrust::raw_pointer_cast(row_ptr_.data()),
            thrust::raw_pointer_cast(row_idx_.data()),
            thrust::raw_pointer_cast(col_idx_.data()),
            thrust::raw_pointer_cast(col_ts_.data()),
            thrust::raw_pointer_cast(chunks.data()), (T)chunks.size(),
            t_begin, t_end,
            thrust::raw_pointer_cast(counts_nodes_buffer.data()),
            thrust::raw_pointer_cast(counts_ts_buffer.data()),
            thrust::raw_pointer_cast(counts_col_buffer.data()),
            use_full_timestamps,
            thrust::raw_pointer_cast(prefix_different_timestamp.data())
        );
        // filter_seeds_kernel<<<blocks, threads, 0, stream_>>>(
        //     thrust::raw_pointer_cast(chunk_ptr_.data()),
        //     thrust::raw_pointer_cast(row_ptr_.data()),
        //     thrust::raw_pointer_cast(row_idx_.data()),
        //     thrust::raw_pointer_cast(col_idx_.data()),
        //     thrust::raw_pointer_cast(col_ts_.data()),
        //     thrust::raw_pointer_cast(chunks.data()), (T)chunks.size(),
        //     t_begin, t_end,
        //     thrust::raw_pointer_cast(counts.data()),
        //     thrust::raw_pointer_cast(chunk_counts_buffer.data())
        // );
        T total_threads = warps * blocks;
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream_),
            counts_nodes_buffer.begin(), counts_nodes_buffer.begin() + total_threads + 1,
            counts_nodes_buffer.begin()
        );
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream_),
            counts_ts_buffer.begin(), counts_ts_buffer.begin() + total_threads + 1,
            counts_ts_buffer.begin()
        );
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream_),
            counts_col_buffer.begin(), counts_col_buffer.begin() + total_threads + 1,
            counts_col_buffer.begin()
        );

        thrust::device_vector<T> out_nodes(counts_nodes_buffer.back());
        thrust::device_vector<T> out_nodes_ptr(counts_nodes_buffer.back()+1);
        thrust::device_vector<TS> out_ts(counts_ts_buffer.back());
        thrust::device_vector<T> out_ts_ptr(counts_ts_buffer.back()+1); 
        thrust::device_vector<T> out_eid(counts_col_buffer.back());


        warps = 4;
        threads = warps * 32;
        blocks = (chunks.size() + warps - 1) / warps;
        collect_seeds_kernel<<<blocks, threads, 0, stream_>>>(
            thrust::raw_pointer_cast(chunk_ptr_.data()),
            thrust::raw_pointer_cast(chunks.data()),
            thrust::raw_pointer_cast(row_ptr_.data()),
            thrust::raw_pointer_cast(row_idx_.data()),
            thrust::raw_pointer_cast(col_idx_.data()),
            thrust::raw_pointer_cast(col_ts_.data()),
            thrust::raw_pointer_cast(edge_id_.data()),
            thrust::raw_pointer_cast(out_nodes.data()),
            thrust::raw_pointer_cast(out_nodes_ptr.data()),
            thrust::raw_pointer_cast(out_ts.data()),
            thrust::raw_pointer_cast(out_ts_ptr.data()),
            thrust::raw_pointer_cast(out_eid.data()),
            (int)chunks.size(), 
            t_begin, t_end,
            thrust::raw_pointer_cast(counts_nodes_buffer.data()),
            thrust::raw_pointer_cast(counts_ts_buffer.data()),
            thrust::raw_pointer_cast(counts_col_buffer.data()),
            use_full_timestamps,
            thrust::raw_pointer_cast(prefix_different_timestamp.data())
        );
        TemporalRoot<T,TS> result(
            std::move(out_nodes),
            std::move(out_nodes_ptr),
            std::move(out_ts),
            std::move(out_ts_ptr),
            std::move(out_eid)
        );
        return std::move(result);
    }
    NegativeRoot<T,TS> get_negative_root(thrust::device_vector<T> &negative_root, thrust::device_vector<TS> &negative_time){
        return NegativeRoot<T,TS>(std::move(negative_root),std::move(negative_time));
    }
    thrust::device_vector<T> concat(thrust::device_vector<T> &A, thrust::device_vector<T> &B){
        thrust::device_vector<T> C(A.size() + B.size());
        thrust::copy(A.begin(), A.end(), C.begin());
        thrust::copy(B.begin(), B.end(), C.begin() + A.size());

        return std::move(C);
    }
    TemporalResult<T,TS> sample_src_in_chunks_khop(
            TemporalRoot<T,TS> seeds_root, NegativeRoot<T,TS> neg_root, int k, int layers, TS allowed_offset, bool equal_root_time, bool keep_root_time, std::string type)
    {

        thrust::device_vector<T>  seeds = seeds_root.roots;
        thrust::device_vector<TS> seed_ts = seeds_root.ts;
        thrust::device_vector<T>  seed_ptr = seeds_root.roots;
        thrust::device_vector<T>  neg_seeds = neg_root.roots;
        thrust::device_vector<TS> neg_seed_ts = neg_root.ts;
        int ts_num = seed_ptr.back();
        thrust::device_vector<T>  neg_seed_ptr(neg_seeds.size() + 1);
        thrust::sequence(neg_seed_ptr.begin(), neg_seed_ptr.end(), ts_num + 1);
        //thrust::sequence<T>(ts_num + 1, ts_num + neg_seeds.size()+1,);
        thrust::device_vector<T>  root     = concat(seeds_root.roots, neg_root.roots);
        thrust::device_vector<TS> root_ts  = concat(seeds_root.ts,     neg_root.ts);
        thrust::device_vector<T> root_ptr = concat(seed_ptr, neg_seed_ptr);
        ts_num = root_ptr.back();
        int root_num = root.size();
        int warps = 4, threads = warps * 32;
        int blocks = (root_num + warps - 1) / warps;
        TemporalResult<T,TS> out(seeds_root);
        for(int l = 0; l < layers ; l++){
            TemporalNeighbor<T,TS> result;
            thrust::device_vector<T> prefix_(ts_num, 0);
            thrust::device_vector<T> start_offset_(ts_num+1, 0);
            thrust::device_vector<T> start_pos(ts_num+1, 0);
            thrust::device_vector<T> end_pos(ts_num+1, 0);
            if(type == "recent"){
                static_recent_neighbor_num<<<blocks,threads, 0, stream_>>>(
                    thrust::raw_pointer_cast(root.data()),
                    thrust::raw_pointer_cast(root_ptr.data()),
                    thrust::raw_pointer_cast(root_ts.data()),
                    (T)root_num, (T)ts_num, k, (T) allowed_offset, (bool)equal_root_time,
                    thrust::raw_pointer_cast(row_ptr_.data()),
                    thrust::raw_pointer_cast(col_idx_.data()),
                    thrust::raw_pointer_cast(col_ts_.data()),
                    thrust::raw_pointer_cast(col_chunk_.data()),
                    thrust::raw_pointer_cast(start_offset_.data()),
                    thrust::raw_pointer_cast(start_pos.data()),
                    thrust::raw_pointer_cast(end_pos.data())
                );
                thrust::transform(
                    thrust::make_zip_iterator(thrust::make_tuple(start_pos.begin(), end_pos.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(start_pos.end(),   end_pos.end())),
                    prefix_.begin(),
                    [] __host__ __device__ (const thrust::tuple<T, T>& t) {
                        return thrust::get<1>(t) - thrust::get<0>(t);
                    }
                );
                thrust::exclusive_scan(thrust::cuda::par.on(stream_),
                                   prefix_.begin(),
                                   prefix_.end(),
                                   prefix_.begin());
                thrust::device_vector<T> new_neighbors(prefix_.back());
                thrust::device_vector<TS> new_neighbors_ts(prefix_.back());
                thrust::device_vector<T> new_neighbors_eid(prefix_.back());
                recent_sample_single_hop_kernel<<<blocks, threads, 0, stream_>>>(
                    thrust::raw_pointer_cast(root.data()),
                    thrust::raw_pointer_cast(root_ptr.data()),
                    thrust::raw_pointer_cast(root_ts.data()),
                    (T)root_num, (T)ts_num, 
                    thrust::raw_pointer_cast(start_pos.data()),
                    thrust::raw_pointer_cast(end_pos.data()),
                    thrust::raw_pointer_cast(prefix_.data()),
                    thrust::raw_pointer_cast(row_ptr_.data()),
                    thrust::raw_pointer_cast(col_idx_.data()),
                    thrust::raw_pointer_cast(col_ts_.data()),
                    thrust::raw_pointer_cast(edge_id_.data()),
                    thrust::raw_pointer_cast(col_chunk_.data()),
                    thrust::raw_pointer_cast(new_neighbors.data()),
                    thrust::raw_pointer_cast(new_neighbors_ts.data()),
                    thrust::raw_pointer_cast(new_neighbors_eid.data())
                );
                result.neighbors = std::move(new_neighbors);
                result.neighbors_ts = std::move(new_neighbors_ts);
                result.neighbors_eid = std::move(new_neighbors_eid);
                result.root_start_ptr.resize(new_neighbors.size(),0);
                result.root_end_ptr.resize(new_neighbors.size(),0);
                thrust::transform(
                    thrust::make_zip_iterator(thrust::make_tuple(prefix_.begin(),start_offset_.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(prefix_.end(),start_offset_.end())),
                    result.root_start_ptr.begin(),
                    [] __host__ __device__ (const thrust::tuple<T, T>& t) {
                        return thrust::get<0>(t) - thrust::get<1>(t);
                    }
                );
                thrust::copy(
                    prefix_.begin() + 1, prefix_.end(),
                    result.root_end_ptr.begin()
                );
            }
            else if(type == "uniform"){
                static_neighbor_num<<<blocks,threads, 0, stream_>>>(
                    thrust::raw_pointer_cast(root.data()),
                    thrust::raw_pointer_cast(root_ptr.data()),
                    thrust::raw_pointer_cast(root_ts.data()),
                    (T)root_num, (T)ts_num, k, (T) allowed_offset, (bool)equal_root_time,
                    thrust::raw_pointer_cast(row_ptr_.data()),
                    thrust::raw_pointer_cast(col_idx_.data()),
                    thrust::raw_pointer_cast(col_ts_.data()),
                    thrust::raw_pointer_cast(col_chunk_.data()),
                    thrust::raw_pointer_cast(prefix_.data()),
                    thrust::raw_pointer_cast(start_pos.data()),
                    thrust::raw_pointer_cast(end_pos.data())
                );

                thrust::exclusive_scan(thrust::cuda::par.on(stream_),
                                   prefix_.begin(),
                                   prefix_.end(),
                                   prefix_.begin());
                thrust::device_vector<T> new_neighbors(prefix_.back());
                thrust::device_vector<TS> new_neighbors_ts(prefix_.back());
                thrust::device_vector<T> new_neighbors_eid(prefix_.back());
                uniform_sample_single_hop_kernel<<<blocks, threads, 0, stream_>>>(
                    thrust::raw_pointer_cast(root.data()),
                    thrust::raw_pointer_cast(root_ptr.data()),
                    thrust::raw_pointer_cast(root_ts.data()),
                    (T)root_num, (T)ts_num, k, 
                    thrust::raw_pointer_cast(prefix_.data()),
                    thrust::raw_pointer_cast(row_ptr_.data()),
                    thrust::raw_pointer_cast(col_idx_.data()),
                    thrust::raw_pointer_cast(col_ts_.data()),
                    thrust::raw_pointer_cast(edge_id_.data()),
                    thrust::raw_pointer_cast(col_chunk_.data()),
                    thrust::raw_pointer_cast(new_neighbors.data()),
                    thrust::raw_pointer_cast(new_neighbors_ts.data()),
                    thrust::raw_pointer_cast(new_neighbors_eid.data()),
                    thrust::raw_pointer_cast(start_pos.data()),
                    thrust::raw_pointer_cast(end_pos.data()),
                    keep_root_time
                );
                result.neighbors = std::move(new_neighbors);
                result.neighbors_ts = std::move(new_neighbors_ts);
                result.neighbors_eid = std::move(new_neighbors_eid);
                result.root_start_ptr.resize(new_neighbors.size(),0);
                thrust::copy(
                    prefix_.begin() + 1, prefix_.end(),
                    result.root_start_ptr.begin()
                );
                
            }
            else{
                throw std::runtime_error("undefined sampling strategy");
            }
            out.append(std::move(result));
            root = out.neighbors_list.back().neighbors;
            root_ptr = thrust::device_vector<T>(out.neighbors_list.back().neighbors.size() + 1);

            thrust::sequence(
                root_ptr.begin(),
                root_ptr.end()
            );
            root_ts = out.neighbors_list.back().neighbors_ts;
            root_num = root.size();
            ts_num = root_ptr.back();
        }
        return std::move(out);
    }
};



// __global__ void compress_output_kernel(
//     const T* prefix, const T* cnt, const T* in_nbr, const TS* in_ts,
//     T* out_nbr, TS* out_ts, T n, int k) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     for(;i < n; i += blockDim.x * gridDim.x) {
//         if (i >= n) return;
//         T s = prefix[i], c = cnt[i], b = i * k;
//         for (T j = 0; j < c; ++j) {
//             out_nbr[s + j] = in_nbr[b + j];
//             out_ts[s + j]  = in_ts[b + j];
//         }
//     }
    
// }
// from_edge_index 函数
class CUDAGraph {
public:
    using T = int64_t;
    using TS = int64_t;
    using Graph_T = Graph<T, TS>;
    using Block_T = TemporalBlock<T, TS>;
    
    Graph_T graph_;

    CUDAGraph(Graph_T g) : graph_(std::move(g)) {}


    
    TemporalResult<T,TS>  sample_src_in_chunks_khop(
                                      TemporalRoot<T,TS>& seeds_root,
                                      NegativeRoot<T,TS>& neg_seeds_root,
                                      int k, int layer, 
                                      TS allowed_offset, 
                                      bool equal_root_time,
                                      bool keep_root_time,
                                      std::string type
                                     ) {
        //thrust::device_vector<T> neg_seeds = tensor_to_device_vector<T>(neg_seeds_tensor);
        //thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunks_tensor);
        return graph_.sample_src_in_chunks_khop(
            seeds_root,  neg_seeds_root, k, layer, allowed_offset, equal_root_time, keep_root_time, type);
    
    }

    TemporalRoot<T,TS> get_seeds_in_chunks(const torch::Tensor& chunks_tensor,
                                     TS time_begin, TS time_end,
                                     bool using_full_timestamp) {
        thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunks_tensor);
        return graph_.get_seeds_in_chunks(chunks, time_begin, time_end, using_full_timestamp);
    }

    NegativeRoot<T,TS> get_negative_root(const torch::Tensor& negative_root_tensor,
                                        const torch::Tensor& negative_time_tensor){
        thrust::device_vector<T> negative_root = tensor_to_device_vector<T>(negative_root_tensor);
        thrust::device_vector<TS> negative_time = tensor_to_device_vector<TS>(negative_time_tensor);
        return graph_.get_negative_root(negative_root,negative_time);
    }

    TemporalResult<T,TS> slice_by_chunk_ts(const torch::Tensor& chunks_tensor, 
                        TS time_begin, TS time_end, bool using_full_timestamp, uint64_t py_stream) {
        thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunks_tensor);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(py_stream);
        return graph_.slice_by_chunk_ts(chunks, time_begin, time_end, using_full_timestamp, stream);
    }

    private:
    template <typename DType>
    thrust::device_vector<DType> tensor_to_device_vector(const torch::Tensor& tensor) {
        thrust::device_vector<DType> vec(tensor.numel());
        cudaMemcpy(thrust::raw_pointer_cast(vec.data()), tensor.data_ptr<DType>(), 
                  tensor.numel() * sizeof(DType), cudaMemcpyDeviceToDevice);
        return vec;
    }
};

// // CUDA 转换函数
CUDAGraph from_edge_index(
    T n, T chunk_size,
    const torch::Tensor& src,
    const torch::Tensor& dst,
    const torch::Tensor& ts,
    const torch::Tensor& row_chunk_mapper,
    uint64_t stream = 0
) {

    //print f("from_edge_index\n");
    try{
        Graph<T,TS> g = Graph<T, TS>(n, chunk_size, src, dst, ts, row_chunk_mapper, stream);
        return CUDAGraph(g);
    } 
    catch (const c10::Error& e) {
        throw std::runtime_error(std::string("Graph construction failed: ") + e.msg());
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("Graph construction failed: ") + e.what());
    }
    catch (...) {
        throw std::runtime_error("Graph construction failed with unknown error");
    }
        
}