#pragma once
#define TORCH_DISABLE_FP8
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <torch/torch.h>
#include <curand_kernel.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <iostream>
#include <error.cu.hpp>
#include <cstring>
#include <future>
#include <queue>
#include <memory>
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
struct EdgeComparator {
    __device__ bool operator()(const Edge<T,TS>& a, const Edge<T,TS>& b) const {
        // 逻辑保持完全一致
        T sc = a.scc, dc = a.dcc;
        T scc = b.scc, dcc = b.dcc;
        if (sc != scc) return sc < scc;
        if (a.src != b.src) return a.src < b.src;
        if (a.ts != b.ts) return a.ts < b.ts;
        if (dc != dcc) return dc < dcc;
        return a.dst < b.dst;
    }
};
typedef thrust::tuple<T, T, T> Tuple3;
struct TupleSum {
    __host__ __device__ Tuple3 operator()(const Tuple3& a, const Tuple3& b) const {
        return thrust::make_tuple(
            thrust::get<0>(a) + thrust::get<0>(b),
            thrust::get<1>(a) + thrust::get<1>(b),
            thrust::get<2>(a) + thrust::get<2>(b)
        );
    }
};

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
        unsigned long long old_val = atomicAdd((unsigned long long*)&row_ptr[e.src], 1ULL);
        if (old_val == 0) {
            atomicAdd((unsigned long long*)&chunk_ptr[src_chunk], 1ULL);
        }
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
        unsigned active = __ballot_sync(0xffffffff, value >= val);
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
        unsigned active = __ballot_sync(0xffffffff, value > val);
        if (active != 0) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }

    return low;
}
// ==================================================================
// 0. 辅助函数：混合搜索与剪枝
// ==================================================================

// 线性查找下界 (适用于短数组)
template <typename TS>
__device__ __forceinline__ int linear_lower_bound(const TS* __restrict__ arr, int l, int r, TS val) {
    // 简单的线性扫描，利用缓存行预取
    for (int i = l; i < r; ++i) {
        if (arr[i] >= val) return i;
    }
    return r;
}

// 线性查找上界
template <typename TS>
__device__ __forceinline__ int linear_upper_bound(const TS* __restrict__ arr, int l, int r, TS val) {
    for (int i = l; i < r; ++i) {
        if (arr[i] > val) return i;
    }
    return r;
}

// 传统的二分查找 (保留给大数据量)
template <typename TS>
__device__ __forceinline__ int binary_lower_bound(const TS* __restrict__ arr, int l, int r, TS val) {
    while (l < r) {
        int mid = l + ((r - l) >> 1);
        if (arr[mid] < val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

template <typename TS>
__device__ __forceinline__ int binary_upper_bound(const TS* __restrict__ arr, int l, int r, TS val) {
    while (l < r) {
        int mid = l + ((r - l) >> 1);
        if (arr[mid] <= val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}


template <typename T, typename TS>
__device__ __forceinline__ void find_time_range(
    const TS* __restrict__ col_ts, 
    T rs, T re, 
    TS t_begin, TS t_end, 
    T& out_left, T& out_right
) {
    TS first_ts = __ldg(&col_ts[rs]);
    TS last_ts  = __ldg(&col_ts[re - 1]);
    if (first_ts >= t_end || last_ts < t_begin) {
        out_left = 0;
        out_right = 0; 
        return;
    }

    bool all_in = (first_ts >= t_begin) && (last_ts < t_end);
    if (all_in) {
        out_left = rs;
        out_right = re;
        return;
    }
    int count = re - rs;
    if (count <= 32) {
        out_left = (first_ts >= t_begin) ? rs : linear_lower_bound(col_ts, rs, re, t_begin);
        T search_start = (out_left == 0) ? rs : out_left; 
        out_right = linear_upper_bound(col_ts, search_start, re, t_end);
    } else {
        out_left = (first_ts >= t_begin) ? rs : binary_lower_bound(col_ts, rs, re, t_begin);
        T search_start = out_left;
        out_right = binary_upper_bound(col_ts, search_start, re, t_end);
    }
}
// // ======================
// // 3. 阶段 1：统计边数
// // ======================
template <typename T = int64_t, typename TS = int64_t>
__global__ void count_edges_kernel(
    const T* __restrict__ chunk_ptr,
    const T* __restrict__ row_ptr,
    const T* __restrict__ row_idx,
    const TS* __restrict__ col_ts,
    const T* __restrict__ col_chunk,
    const T* __restrict__ chunks,
    const bool* __restrict__ chunk_exists,
    T num_chunks,
    T num_nodes,
    TS t_begin,
    TS t_end,
    T* __restrict__ counts_nodes_buffer,
    T* __restrict__ counts_ts_buffer,
    T* __restrict__ counts_col_buffer,
    bool use_full_timestamps
) {
    //每个block处理一个chunk,每个warp处理一个节点，每个线程处理每条边
    int block_idx = blockIdx.x;
    int global_warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int warp_id = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    int lane = threadIdx.x % 32;
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
            if(node_idx >= num_nodes) break;

            T row = node_idx;
            T rs = row_ptr[node_idx];
            T re = row_ptr[node_idx + 1];
            
            if(rs >= re) continue;

            if (rs >= re) continue;
            //计算时间范围
            T left  = lower_bound(col_ts, rs, re, t_begin);
            T right = upper_bound(col_ts, left, re, t_end);
            if (left >= right) continue;
            if(lane == 0)
                counts_nodes_buffer[global_warp_id] += 1;
            if(use_full_timestamps){
                if(lane == 0)   
                    counts_ts_buffer[global_warp_id] += t_end - t_begin;
            }
            else{
                int local_count_ts = 0;
                for(T e = left + lane; e < right; e +=32){
                    if(e > left && col_ts[e] != col_ts[e-1]){
                        local_count_ts++;
                    }
                }
                warp_reduce_sum(local_count_ts);
                if(lane == 0){
                    counts_ts_buffer[global_warp_id] += local_count_ts;
                }
            }
            int local_count_col = 0;
            for(T e = left + lane; e < right; e += 32) {
                if (e >= right) break;
                T dst_c = col_chunk[e];
                if (dst_c < num_nodes && chunk_exists[dst_c]) {
                    local_count_col++;
                }
            }
            warp_reduce_sum(local_count_col);
            if(lane == 0)
                counts_col_buffer[global_warp_id] += local_count_col;
        }
        block_idx += gridDim.x;
    }
}


// ======================
// 4. 阶段 2：写入数据
// ======================
template <typename T = int64_t, typename TS = int64_t>
__global__ void write_edges_kernel(
    const T* __restrict__ chunk_ptr,
    const T* __restrict__ row_ptr,
    const T* __restrict__ row_idx,
    const T* __restrict__ col_idx,
    const T* __restrict__ col_eid,
    const TS* __restrict__ col_ts,
    const T* __restrict__ col_chunk,
    const T* __restrict__ chunks,
    const bool* __restrict__ chunk_exists,
    T* __restrict__ out_row_ptr,
    T* __restrict__ out_row_idx,
    T* __restrict__ out_row_ts,
    T* __restrict__ out_ts_ptr,
    T* __restrict__ out_col_idx,
    TS* __restrict__ out_col_ts,
    T * __restrict__ out_col_eid,
    int num_chunks,
    T num_nodes,
    TS t_begin,
    TS t_end,
    T* __restrict__ counts_nodes_buffer,
    T* __restrict__ counts_ts_buffer,
    T* __restrict__ counts_col_buffer,
    bool use_full_timestamps
) {
    int block_idx = blockIdx.x;
    int global_warp_id = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    while(block_idx < num_chunks){
        T chunk = chunks[block_idx];
        T chunk_start = chunk_ptr[block_idx];
        T chunk_end = chunk_ptr[block_idx + 1];
        if(chunk_start >= chunk_end){
            block_idx += gridDim.x;
            continue;
        }
        int warp_id = threadIdx.x / 32;
        int warps_per_block = blockDim.x / 32;
        int lane = threadIdx.x % 32;
        int local_nodes_count = 0;
        for(T row_x = chunk_start + warp_id; row_x < chunk_end ; row_x += warps_per_block){
            T node_idx = row_idx[row_x];
            if(node_idx >= num_nodes) break;

            T row = node_idx;
            T rs = row_ptr[node_idx];
            T re = row_ptr[node_idx + 1];
            
            if(rs >= re) continue;

            if (rs >= re) continue;
            //计算时间范围
            T left  = lower_bound(col_ts, rs, re, t_begin);
            T right = upper_bound(col_ts, left, re, t_end);
            if (left >= right) continue;
            if(lane == 0){
                //更新结束位置的ptr
                out_row_ptr[local_nodes_count + counts_nodes_buffer[global_warp_id] + 1]= \
                    counts_ts_buffer[global_warp_id];
                out_row_idx[local_nodes_count + counts_nodes_buffer[global_warp_id]]= row;
                local_nodes_count++;
            }
            int offset_col_ = 0;
            int offset_ts_ = 0;
            if(use_full_timestamps){
                for(T e = t_begin; e < t_end ; e ++) {
                   out_row_ts[counts_ts_buffer[global_warp_id] + e - t_begin] = e;
                }
                for(T e = left + lane; e < ((right + 31)/32 + 1) * 32; e +=32){
                    bool active = e < right;
                    bool dst_in_chunk = active ? chunk_exists[col_chunk[e]] : 0;
                    bool new_timestamp = active ? (e==left || col_ts[e] != col_ts[e-1]) : 0;
                    int offset_col_p = dst_in_chunk ? warpExclusiveScan(1): warpExclusiveScan(0);
                    int offset_ts_p = new_timestamp ? warpExclusiveScan(1) : warpExclusiveScan(0);
                    int incre_col = warp_broadcast(offset_col_p, 31);
                    int incre_ts = warp_broadcast(offset_ts_p, 31);
                    T dst_c = col_chunk[e];
                    if(new_timestamp){
                        int offset_ts = offset_ts_ + offset_ts_p;
                        int offset_col = offset_col + offset_col_p;
                        for(int i = col_ts[e-1] + 1; e > left && i < col_ts[e]; i++){
                            out_ts_ptr[counts_ts_buffer[global_warp_id] + i - t_begin] = offset_col + counts_col_buffer[global_warp_id];
                        }
                        offset_ts_ += incre_ts;
                    }
                    if(dst_in_chunk){
                            //out_ts_ptr[counts_ts_buffer[global_warp_id] + col_ts[e] - t_begin]++;
                        int offset_col = offset_col_ + offset_col_p;
                        out_col_idx[counts_col_buffer[global_warp_id] + offset_col] = col_idx[e];
                        out_col_ts[counts_col_buffer[global_warp_id] + offset_col] = col_ts[e];
                        out_col_eid[counts_col_buffer[global_warp_id] + offset_col] = col_eid[e];
                        offset_col_ += incre_col;
                    }
                }
            }
            else{
                for(T e = left + lane; e < ((right + 31)/32 + 1) * 32; e +=32){
                    bool active = e < right;
                    bool dst_in_chunk = active ? chunk_exists[col_chunk[e]] : 0;
                    bool new_timestamp = active ? (e==left || col_ts[e] != col_ts[e-1]) : 0;
                    int offset_col_p = dst_in_chunk ? warpInclusiveScan(1): warpInclusiveScan(0);
                    int offset_ts_p = new_timestamp ? warpInclusiveScan(1) : warpInclusiveScan(0);
                    int incre_col = warp_broadcast(offset_col_p, 31);
                    int incre_ts = warp_broadcast(offset_ts_p, 31);
                    T dst_c = col_chunk[e];
                    int offset_col = offset_col_ + offset_col_p - dst_in_chunk;
                    if(new_timestamp){
                        int offset_ts = offset_ts_ + offset_ts_p - new_timestamp;
                        out_ts_ptr[counts_ts_buffer[global_warp_id] + offset_ts] = offset_col + counts_col_buffer[global_warp_id];
                        out_row_ts[counts_ts_buffer[global_warp_id] + offset_ts] = col_ts[e];
                        offset_ts_ += incre_ts;
                    }
                    if(dst_in_chunk){
                            //out_ts_ptr[counts_ts_buffer[global_warp_id] + col_ts[e] - t_begin]++;
                        out_col_idx[counts_col_buffer[global_warp_id] + offset_col] = col_idx[e];
                        out_col_ts[counts_col_buffer[global_warp_id] + offset_col] = col_ts[e];
                        out_col_eid[counts_col_buffer[global_warp_id] + offset_col] = col_eid[e];
                        offset_col_ += incre_col;
                    }
                }
            }
        }
        block_idx += gridDim.x;
    }
 
}

__global__ void __launch_bounds__(128, 8) filter_seeds_kernel(
    const T* __restrict__ chunk_ptr, 
    const T* __restrict__ row_ptr, 
    const T* __restrict__ row_idx, 
    const T* __restrict__ col_idx, 
    const TS* __restrict__ col_ts,
    const T* __restrict__ chunks, 
    int num_chunks, 
    TS t_begin, TS t_end, 
    T* __restrict__ counts_nodes_buffer,
    T* __restrict__ counts_ts_buffer,
    T* __restrict__ counts_col_buffer,
    bool using_full_timestamp,
    const T* __restrict__ prefix_different_timestamp,
    int blocks_per_chunk
){
    // [SMEM Optimization]: Cache Chunk Boundaries
    // 只需要两个 int64 空间，几乎不占资源，但能减少重复的 global load
    __shared__ T smem_chunk_range[2];

    // 1. 计算当前 Block 负责哪个 Chunk
    int chunk_idx = blockIdx.x / blocks_per_chunk;
    if (chunk_idx >= num_chunks) return;
    // 2. 协作加载 Metadata
    if (threadIdx.x == 0) {
        T c_id = chunks[chunk_idx]; // 读取实际的 Chunk ID
        smem_chunk_range[0] = chunk_ptr[c_id];
        smem_chunk_range[1] = chunk_ptr[c_id + 1];
    }
    __syncthreads();

    T chunk_start = smem_chunk_range[0];
    T chunk_end   = smem_chunk_range[1];
    // 3. 计算循环边界 (Grid-Stride Loop logic for specific chunk)
    int block_offset = blockIdx.x % blocks_per_chunk;
    T loop_stride = blocks_per_chunk * blockDim.x;
    T loop_start = chunk_start + block_offset * blockDim.x + threadIdx.x;
    
    // 4. Global Buffer Index (绝对唯一 ID)
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 5. 使用寄存器累加 (比 atomic 或 shared memory 快)
    T local_nodes = 0;
    T local_ts = 0;
    T local_col = 0;

    for(T row_x = loop_start; row_x < chunk_end; row_x += loop_stride){
        T node_idx = row_idx[row_x];
        T rs = row_ptr[node_idx];
        T re = row_ptr[node_idx + 1];
       

        if (rs >= re) continue;

        // 二分查找时间范围
        //T left  = lower_bound(col_ts, rs, re, t_begin);
        //T right = upper_bound(col_ts, left, re, t_end);
        T left = 0, right = 0;
        find_time_range(col_ts, rs, re, t_begin, t_end, left, right);
        if (left >= right) continue;
        local_nodes += 1;
        if(using_full_timestamp){
            local_ts += (t_end - t_begin);
            local_col += (right - left);
        } else {
            if(left == 0)
                local_ts += prefix_different_timestamp[right - 1];
            else
                local_ts += (prefix_different_timestamp[right - 1] - prefix_different_timestamp[left - 1]);
            local_col += (right - left);
        }
    }
    // 6. 写入全局 Buffer (每个线程只写一次)
    counts_nodes_buffer[global_tid] = local_nodes;
    counts_ts_buffer[global_tid]    = local_ts;
    counts_col_buffer[global_tid]   = local_col;
}

// ==================================================================
// 2. 收集 Kernel：写入实际数据
// ==================================================================
template <typename T = int64_t, typename TS = int64_t>
__global__ void __launch_bounds__(128, 8) collect_seeds_kernel( 
    const T* __restrict__ chunk_ptr, 
    const T* __restrict__ chunks,
    const T* __restrict__ row_ptr, 
    const T* __restrict__ row_idx, 
    const T* __restrict__ col_idx, 
    const TS* __restrict__ col_ts, 
    const T* __restrict__ col_eid,
    T* __restrict__ out_nodes, 
    T* __restrict__ out_nodes_ptr, 
    TS* __restrict__ out_ts, 
    T* __restrict__ out_ts_ptr,
    T* __restrict__ out_eid, 
    int num_chunks, 
    TS t_begin, TS t_end,
    const T* __restrict__ scanned_nodes_buffer, // 注意：这是 Scan 后的 Offset
    const T* __restrict__ scanned_ts_buffer,
    const T* __restrict__ scanned_col_buffer,
    bool using_full_timestamp,
    const T* __restrict__ prefix_different_timestamp,
    int blocks_per_chunk
) {
    // Shared Memory 缓存
    __shared__ T smem_chunk_range[2];
    int chunk_idx = blockIdx.x / blocks_per_chunk;
    if (chunk_idx >= num_chunks) return;

    if (threadIdx.x == 0) {
        T c_id = chunks[chunk_idx];
        smem_chunk_range[0] = chunk_ptr[c_id];
        smem_chunk_range[1] = chunk_ptr[c_id + 1];
    }
    __syncthreads();

    T chunk_start = smem_chunk_range[0];
    T chunk_end   = smem_chunk_range[1];
    if(chunk_start >= chunk_end) return;

    int block_offset = blockIdx.x % blocks_per_chunk;
    T loop_stride = blocks_per_chunk * blockDim.x;
    T loop_start = chunk_start + block_offset * blockDim.x + threadIdx.x;
    
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 读取当前线程的起始写入位置 (从 Scan 后的 Buffer 获取)
    T current_node_pos = scanned_nodes_buffer[global_tid];
    T current_ts_pos   = scanned_ts_buffer[global_tid];
    T current_eid_pos  = scanned_col_buffer[global_tid];
    //printf("current pos %lld %lld %lld\n", current_node_pos, current_ts_pos, current_eid_pos);
    for(T row_x = loop_start; row_x < chunk_end; row_x += loop_stride){
        T node_idx = row_idx[row_x];
        T rs = row_ptr[node_idx];
        T re = row_ptr[node_idx + 1];
        if (rs >= re) continue;
        // --- 替换旧逻辑 ---
        T left = 0, right = 0;
        find_time_range(col_ts, rs, re, t_begin, t_end, left, right);
        //printf("node_idx %lld rs %lld re %lld left %lld right %lld\n", node_idx, rs, re, left, right);
        if (left >= right) continue;
        out_nodes[current_node_pos] = node_idx;
        out_nodes_ptr[current_node_pos] = current_ts_pos; 
        //printf("generate nodes %lld %lld %lld %lld %lld %lld\n",row_x,current_node_pos,node_idx, current_ts_pos, out_nodes[current_node_pos], out_nodes_ptr[current_node_pos]);
        // 预先计算下一个位置（用于最后补齐或 Loop）
        // 实际上并行写入 ptr 数组比较 trick，这里假设后续逻辑能处理
        // 或者我们在 Loop 结束前写入 current_node_pos + 1 的位置？
        // 按照你之前的逻辑，这里只写 Start 即可。
        
        // --- 写入 Edges & Timestamps ---
        if(using_full_timestamp){
            // 全量时间戳
            for(T e = t_begin; e < t_end; e++) {
                out_ts[current_ts_pos + (e - t_begin)] = e;
            }
            for(T e = left; e < right; e++) {
                // 这里的逻辑比较复杂，根据你原代码还原：
                // 看起来是要把边ID分配给对应的时间戳桶
                // 这里为简化，仅展示 offset 更新
                if(e == left || col_ts[e] != col_ts[e-1]) {
                    for(int i = col_ts[e-1] + 1; i < col_ts[e]; i++) {
                        out_ts_ptr[current_ts_pos + i - t_begin] = current_eid_pos + (e - left);
                    }
                }
                out_eid[current_eid_pos + (e - left)] = col_eid[e];
            }
            current_ts_pos += (t_end - t_begin);
            current_eid_pos += (right - left);

        } else {
            // 差分时间戳
            T prefix_left = (left == 0) ? 0 : prefix_different_timestamp[left - 1] ;
            
            for(T e = left; e < right; e++){
                if(e == left || col_ts[e] != col_ts[e-1]){
                    T t_offset = prefix_different_timestamp[e] - prefix_left -1;
                    out_ts[current_ts_pos + t_offset] = col_ts[e];
                    out_ts_ptr[current_ts_pos + t_offset] = current_eid_pos + (e - left);
                }
                out_eid[current_eid_pos + (e - left)] = col_eid[e];
            }
            T local_ts_count = (left == 0) ? prefix_different_timestamp[right-1] : (prefix_different_timestamp[right-1] - prefix_different_timestamp[left-1]);
            current_ts_pos += local_ts_count;
            current_eid_pos += (right - left);
        }
        
        current_node_pos++;
    }
}
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
            TS ts = node_ts[t];
            T rs = row_ptr[node], re = row_ptr[node + 1];
            if (rs >= re) continue;
            T se=0,te = 0;
            TS start =  allowed_offset > -1 ? node_ts[t] - allowed_offset:0;
            TS end = ts + (TS)equal_root_time;
            find_time_range(col_ts, rs ,re, start, end, se, te);
            if(se>=te) continue;
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
    const T* row_ptr, const T* col_idx, const TS* col_ts, const TS* col_eid, const T* col_chunk, T* out_nbr, TS* out_ts, T* out_eid, T* out_dt
) {
        int start = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for(int i = start; i < num_ts; i+= blockDim.x * gridDim.x){
            T pos = prefix[i];
            for(int j = 0; j < end_pos[i] - start_pos[i]; j++){
                T e = start_pos[i] + j;
                out_nbr[pos + j] = col_idx[e];
                out_ts[pos + j] = col_ts[e];
                out_eid[pos + j] = col_eid[e];
                out_dt[pos + j] = col_ts[e] - node_ts[i];
            }
        }
}
template <typename T = int64_t, typename TS = int64_t>
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
            TS ts = node_ts[t];
            T rs = row_ptr[node], re = row_ptr[node + 1];
            if (rs >= re) continue;
            T se,te = 0;
            TS start =  allowed_offset > -1 ? node_ts[t] - allowed_offset:0;
            TS end = ts + (TS)equal_root_time;
            find_time_range(col_ts, rs ,re, start, end, se, te);
            if(se>=te) continue;
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
    const T* row_ptr, const T* col_idx, const TS* col_ts, const TS* col_eid, const T* col_chunk, T* out_nbr, TS* out_ts, T* out_eid, T* out_dt,
    T* start_pos, T* end_pos, bool keep_root_time
) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(1317, threadIdx.x, 0, &state); // 初始化状态



    for(int i = start; i < num_nodes; i+= blockDim.x * gridDim.x){
        if (i >= num_nodes) return;
        T node = nodes[i];
        for(int t = nodes_ptr[i]; t < nodes_ptr[i+1]; t++){
            T rs = start_pos[t];
            T re = end_pos[t];
            if (rs >= re) continue;
            T pos = prefix[t];
            if(re-rs <= k){
                for(int j = 0; j < re - rs; j++){
                    T e = rs + j;
                    //printf("%lld %lld %lld %lld %lld %lld\n", pos + j, rs, re, e, col_idx[e], col_ts[e]);
                    out_nbr[pos + j] = col_idx[e];
                    out_ts[pos + j] = keep_root_time?node_ts[t]:col_ts[e];
                    out_eid[pos + j] = col_eid[e];
                    out_dt[pos + j] = col_ts[e] - node_ts[t];
                }
            }
            else{
                for(int j = 0; j < k; j++){
                    T rand_idx = (int)(curand_uniform(&state) * (re - rs));
                    T e = rand_idx + rs;
                    //printf("%lld %lld %lld %lld %lld %lld\n", pos + j, rs, re, e, col_idx[e], col_ts[e]);
                    out_nbr[pos + j] = col_idx[e];
                    out_ts[pos + j] = keep_root_time?node_ts[t]:col_ts[e];
                    out_eid[pos + j] = col_eid[e];
                    out_dt[pos + j] = col_ts[e] - node_ts[t];
                }    
            }
        }
    }
}


struct KeyValue {
    T  idx;
    TS ts;
    __host__ __device__ KeyValue(T i, TS t) : idx(i), ts(t) {}
};


struct CompareKeyValue {
    __host__ __device__ bool operator()(const KeyValue& a, const KeyValue& b) const {
        if (a.idx != b.idx) return a.idx < b.idx;
        return a.ts < b.ts;
    }
};

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
    thrust::device_vector<T> neighbors_dt;

    int neighbor_num;
    
    TemporalNeighbor() = default;
    TemporalNeighbor(int root_num, int num){
        neighbor_num = num;
        root_start_ptr.resize(root_num + 1);
        root_end_ptr.resize(root_num + 1);
        neighbors.resize(num);
        neighbors_ts.resize(num);
        neighbors_eid.resize(num);
        neighbors_dt.resize(num);
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

struct UnifiedResult{
    torch::Tensor unique_nodes; //global ids
    torch::Tensor row_ptr;
    torch::Tensor col_idx;
    torch::Tensor col_eid;
    torch::Tensor col_ts;
    
}
struct IsDiffPrefix {
    __host__ __device__
    int operator()(const Edge<long, long>& a, const Edge<long, long>& b) const {
        // 示例逻辑：如果源节点或时间戳不同，返回 1，否则返回 0
        // 请根据你的实际业务逻辑修改
        if (a.src != b.src || a.ts != b.ts) return 1;
        return 0;
    }
};
thrust::device_vector<T> concat(const thrust::device_vector<T> &A, const thrust::device_vector<T> &B, cudaStream_t stream) {
    thrust::device_vector<T> C(A.size() + B.size());
    // 使用 par.on(stream) 确保在正确的流上执行拷贝
    thrust::copy(thrust::cuda::par.on(stream), A.begin(), A.end(), C.begin());
    thrust::copy(thrust::cuda::par.on(stream), B.begin(), B.end(), C.begin() + A.size());
    return std::move(C);
}
template <typename T = int64_t, typename TS = int64_t>
class Graph {
   // thrust::device_vector<T> slice_ptr_;
    thrust::device_vector<T>  chunk_ptr_, row_ptr_, row_idx_, col_idx_, col_chunk_, edge_id_, src_idx_;
    thrust::device_vector<TS> col_ts_;
    T  num_nodes_, num_edges_, chunk_size_;
   // T slice_size_;
    cudaStream_t stream_;
    thrust::device_vector<T> counts_buffer;
    thrust::device_vector<T> chunk_counts_buffer;
    thrust::device_vector<T> chunk_nodes_counts_buffer;

    thrust::device_vector<T> counts_nodes_buffer;
    thrust::device_vector<T> counts_ts_buffer;
    thrust::device_vector<T> counts_col_buffer;

    thrust::device_vector<T> prefix_different_timestamp;
    int rank_;
    public:
    cudaStream_t get_stream(){
        return stream_;
    }
    T get_num_nodes(){
        return num_nodes_;
    }
    Graph(T n, T chunk_size, //T slice_size,
        const torch::Tensor& src, 
        const torch::Tensor& dst,
        const torch::Tensor& ts, 
        const torch::Tensor& eid,
        const torch::Tensor& row_chunk_mapper,
        uint64_t py_stream,
        int rank)
        : num_nodes_(n), chunk_size_(chunk_size),rank_(rank)
    {
        try {
            //printf("Building graph with %ld nodes and chunk size %ld...\n", n, chunk_size);
            // ==============================================================
            // 1. 基础检查：CUDA + 类型
            // ==============================================================
            cudaSetDevice(rank);
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
            auto eid_cpu = eid.cpu().contiguous();
            const T*  s_ptr = src_cpu.data_ptr<T>();
            const T*  d_ptr = dst_cpu.data_ptr<T>();
            const TS* t_ptr = ts_cpu.data_ptr<TS>();
            const T* eid_ptr = eid_cpu.data_ptr<T>();
            // 边界检查：节点 ID 不能超过 num_nodes_
            for (T i = 0; i < m; ++i) {
                TORCH_CHECK(s_ptr[i] >= 0 && s_ptr[i] < n,
                            "src[", i, "] = ", s_ptr[i], " out of range [0, ", n, ")");
                TORCH_CHECK(d_ptr[i] >= 0 && d_ptr[i] < n,
                            "dst[", i, "] = ", d_ptr[i], " out of range [0, ", n, ")");
            }
            printf("Input edges copied to host and validated.\n");
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
                h_edges[i] = Edge_T(s, d, t_ptr[i], eid_ptr[i], s_chunk, d_chunk);
                ////printf("%d %d %d %d %d %d\n", s, d, t_ptr[i], i, s_chunk, d_chunk);
            }
            printf("Edge list constructed on host.\n");
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

            thrust::sort(thrust::cuda::par.on(stream_), d_edges.begin(), d_edges.end(), EdgeComparator<T,TS>());
            printf("Edges copied to device and sorted.\n");
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
            
            printf("CSR structures initialized.  %d \n", col_idx_.size());
            // ==============================================================
            // 8. 启动 Kernel
            // ==============================================================
            int block_dim = 256;
            int grid_dim = (m + block_dim - 1) / block_dim; // 依然需要用 m 计算 Grid
            dim3 block(block_dim), grid(grid_dim);
            printf("%d chunk_size\n", chunk_size_);
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
            
            printf("Graph construction kernel launched.\n");
            // // ==============================================================
            // // 9. 同步 + 错误检查
            // // ==============================================================
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
            printf("Graph construction kernel finished.\n");
            // // ==============================================================
            // // 10. 后处理：scan + unique
            // // ==============================================================
            // row_idx_.resize(num_nodes_);

            thrust::exclusive_scan(thrust::cuda::par.on(stream_), 
                                row_ptr_.begin() , row_ptr_.end(), row_ptr_.begin());
            thrust::exclusive_scan(thrust::cuda::par.on(stream_), 
                                chunk_ptr_.begin(), chunk_ptr_.end(), chunk_ptr_.begin());
            cudaStreamSynchronize(stream_);
            thrust::device_vector<T> unique_src(m);
            thrust::copy(thrust::cuda::par.on(stream_), src_idx_.begin(), src_idx_.end(), unique_src.begin());
            auto new_end = thrust::unique(thrust::cuda::par.on(stream_),
                                    unique_src.begin(), unique_src.end());
            row_idx_.resize(thrust::distance(unique_src.begin(), new_end));
            thrust::copy(thrust::cuda::par.on(stream_),
                        unique_src.begin(), new_end, row_idx_.begin());
            if(m>0){
                prefix_different_timestamp.resize(m,0);

                thrust::transform(thrust::cuda::par.on(stream_),
                                d_edges.begin(),d_edges.end()-1,
                                d_edges.begin()+1,
                                prefix_different_timestamp.begin()+1,
                                IsDiffPrefix()
                              );
                prefix_different_timestamp[0] = 1;
                thrust::inclusive_scan(
                                    thrust::cuda::par.on(stream_),
                                   prefix_different_timestamp.begin(),
                                   prefix_different_timestamp.end(),
                                   prefix_different_timestamp.begin()
                            );
                thrust::host_vector<T> h_prefix_different_timestamp = prefix_different_timestamp;

            }
            // // ==============================================================
            // // 11. 初始化计数缓冲
            // // ==============================================================
            
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
            std::move(block_root)//,
            //std::vector<TemporalNeighbor<T, TS>>{std::move(block_nbr)}
        );
        block_result.append(std::move(block_nbr));
        return std::move(block_result);
    }
    // ==================================================================
// 3. Host 端调用函数：包含自动调优和内存管理
// ==================================================================
    TemporalRoot<T,TS> get_seeds_in_chunks(const thrust::device_vector<T>& chunks,
                                        TS t_begin, TS t_end,
                                        bool use_full_timestamps) {
        T num_chunks = chunks.size();
        std::cout<<chunks.size()<<std::endl;
        if (num_chunks == 0) return TemporalRoot<T,TS>(0,0);

        // 1. 获取设备属性 (建议在 Graph 构造函数中只做一次，存入成员变量)
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        // 2. 自动调优参数 (Auto-Tuning)
        const int THREADS_PER_BLOCK = 128;
        // 目标：占用 25% 的 SM 资源，给训练任务留出空间
        // 每个 SM 驻留 4 个 Block 足够隐藏延迟
        int target_total_blocks = prop.multiProcessorCount * 4 * 0.25; 
        if (target_total_blocks < 32) target_total_blocks = 32; // 下限保护

        // 计算 blocks_per_chunk
        int blocks_per_chunk = (target_total_blocks + num_chunks - 1) / num_chunks;
        if (blocks_per_chunk < 1) blocks_per_chunk = 1;
        if (blocks_per_chunk > 256) blocks_per_chunk = 256; // 上限保护

        int grid_size = num_chunks * blocks_per_chunk;
        size_t total_buffer_needed = grid_size * THREADS_PER_BLOCK;
        std::cout<<"buffer needed"<<" "<<total_buffer_needed<<std::endl;
        // 3. 动态调整 Buffer 大小
        if (counts_nodes_buffer.size() < total_buffer_needed + 1) {
            size_t new_size = (total_buffer_needed + 1) * 1.2; // 预留 20% 空间防止频繁扩容
            counts_nodes_buffer.resize(new_size);
            counts_ts_buffer.resize(new_size);
            counts_col_buffer.resize(new_size);
        }

        // 4. 启动 Filter Kernel (统计数量)
        std::cout<<"chunk size"<<chunk_ptr_.size()<<std::endl;
        filter_seeds_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream_>>>(
            thrust::raw_pointer_cast(chunk_ptr_.data()),
            thrust::raw_pointer_cast(row_ptr_.data()),
            thrust::raw_pointer_cast(row_idx_.data()),
            thrust::raw_pointer_cast(col_idx_.data()),
            thrust::raw_pointer_cast(col_ts_.data()),
            thrust::raw_pointer_cast(chunks.data()), (int)num_chunks,
            t_begin, t_end,
            thrust::raw_pointer_cast(counts_nodes_buffer.data()),
            thrust::raw_pointer_cast(counts_ts_buffer.data()),
            thrust::raw_pointer_cast(counts_col_buffer.data()),
            use_full_timestamps,
            thrust::raw_pointer_cast(prefix_different_timestamp.data()),
            blocks_per_chunk // 传入调优参数
        );
        std::cout<<"Filter kernel launched with grid size: "<<grid_size<<", blocks per chunk: "<<blocks_per_chunk<<std::endl;
        // 5. Exclusive Scan (计算 Offset)
        // 注意：Scan 范围必须包含 total_buffer_needed + 1 个位置，用于获取总和
        auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(
            counts_nodes_buffer.begin(), 
            counts_ts_buffer.begin(), 
            counts_col_buffer.begin()
        ));
        auto zip_end = zip_begin + total_buffer_needed + 1;

        // 执行一次 Scan 代替三次
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream_),
            zip_begin,
            zip_end,
            zip_begin, // 原地 Scan
            thrust::make_tuple(0, 0, 0), // 初始值
            TupleSum() // 自定义加法
        );

        // 获取总大小 (只需读一次)
        Tuple3 total_counts = zip_begin[total_buffer_needed];
        T total_nodes = thrust::get<0>(total_counts);
        T total_ts    = thrust::get<1>(total_counts);
        T total_eid   = thrust::get<2>(total_counts);

        // 分配输出 Tensor (thrust::device_vector)
        thrust::device_vector<T> out_nodes(total_nodes);
        thrust::device_vector<T> out_nodes_ptr(total_nodes + 1);
        
        // 初始化 ptr 最后一个元素 (total count)
        // 这是一个小优化：直接拷贝 total_ts 到最后
        // thrust::fill(out_nodes_ptr.end()-1, out_nodes_ptr.end(), total_ts); 
        // 上面这行如果流不同步可能会有问题，建议在 kernel 里处理或忽略

        thrust::device_vector<TS> out_ts(total_ts);
        thrust::device_vector<T> out_ts_ptr(total_ts + 1); 
        thrust::device_vector<T> out_eid(total_eid);
        std::cout<<"Output buffers allocated: "
            << "out_nodes=" << total_nodes
            << ", out_ts=" << total_ts
            << ", out_eid=" << total_eid << std::endl;
        // 7. 启动 Collect Kernel (写入数据)
        collect_seeds_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream_>>>(
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
            (int)num_chunks, 
            t_begin, t_end,
            thrust::raw_pointer_cast(counts_nodes_buffer.data()),
            thrust::raw_pointer_cast(counts_ts_buffer.data()),
            thrust::raw_pointer_cast(counts_col_buffer.data()),
            use_full_timestamps,
            thrust::raw_pointer_cast(prefix_different_timestamp.data()),
            blocks_per_chunk
        );

        // 8. 修正 out_nodes_ptr 的最后一个元素 (Total Count)
        // 因为多线程写入无法保证最后一个元素被正确填充（那是 exclusive scan 的 sum）
        // 我们手动把 scan 的结果赋值过去
        // out_nodes_ptr[total_nodes] = total_ts;
        // 使用 cudaMemcpyAsync 拷贝单个值
        cudaMemcpyAsync(
            thrust::raw_pointer_cast(out_nodes_ptr.data()) + total_nodes,
            &total_ts,
            sizeof(T),
            cudaMemcpyHostToDevice,
            stream_
        );

        // 确保拷贝完成
        cudaStreamSynchronize(stream_);
        //for(int i = 0; i < total_nodes; i++){
        //    printf("%d\n", out_nodes[i]);
        //}
        // 构造并返回结果
        TemporalRoot<T,TS> result(
            std::move(out_nodes),
            std::move(out_nodes_ptr),
            std::move(out_ts),
            std::move(out_ts_ptr),
            std::move(out_eid)
        );
        return std::move(result);
    }
    // TemporalRoot<T,TS> get_seeds_in_chunks(const thrust::device_vector<T>& chunks,
    //                          TS t_begin, TS t_end,
    //                          bool use_full_timestamps
    //                          ) {

    //     int warps = 4, threads = warps * 32;
    //     int blocks = (chunks.size() + warps - 1) / warps;
    //     filter_seeds_kernel<<<blocks, threads, 0, stream_>>>(
    //         thrust::raw_pointer_cast(chunk_ptr_.data()),
    //         thrust::raw_pointer_cast(row_ptr_.data()),
    //         thrust::raw_pointer_cast(row_idx_.data()),
    //         thrust::raw_pointer_cast(col_idx_.data()),
    //         thrust::raw_pointer_cast(col_ts_.data()),
    //         thrust::raw_pointer_cast(chunks.data()), (T)chunks.size(),
    //         t_begin, t_end,
    //         thrust::raw_pointer_cast(counts_nodes_buffer.data()),
    //         thrust::raw_pointer_cast(counts_ts_buffer.data()),
    //         thrust::raw_pointer_cast(counts_col_buffer.data()),
    //         use_full_timestamps,
    //         thrust::raw_pointer_cast(prefix_different_timestamp.data())
    //     );
    //     T total_threads = warps * blocks;
    //     thrust::exclusive_scan(
    //         thrust::cuda::par.on(stream_),
    //         counts_nodes_buffer.begin(), counts_nodes_buffer.begin() + total_threads + 1,
    //         counts_nodes_buffer.begin()
    //     );
    //     thrust::exclusive_scan(
    //         thrust::cuda::par.on(stream_),
    //         counts_ts_buffer.begin(), counts_ts_buffer.begin() + total_threads + 1,
    //         counts_ts_buffer.begin()
    //     );
    //     thrust::exclusive_scan(
    //         thrust::cuda::par.on(stream_),
    //         counts_col_buffer.begin(), counts_col_buffer.begin() + total_threads + 1,
    //         counts_col_buffer.begin()
    //     );

    //     thrust::device_vector<T> out_nodes(counts_nodes_buffer.back());
    //     thrust::device_vector<T> out_nodes_ptr(counts_nodes_buffer.back()+1);
    //     thrust::device_vector<TS> out_ts(counts_ts_buffer.back());
    //     thrust::device_vector<T> out_ts_ptr(counts_ts_buffer.back()+1); 
    //     thrust::device_vector<T> out_eid(counts_col_buffer.back());


    //     warps = 4;
    //     threads = warps * 32;
    //     blocks = (chunks.size() + warps - 1) / warps;
    //     collect_seeds_kernel<<<blocks, threads, 0, stream_>>>(
    //         thrust::raw_pointer_cast(chunk_ptr_.data()),
    //         thrust::raw_pointer_cast(chunks.data()),
    //         thrust::raw_pointer_cast(row_ptr_.data()),
    //         thrust::raw_pointer_cast(row_idx_.data()),
    //         thrust::raw_pointer_cast(col_idx_.data()),
    //         thrust::raw_pointer_cast(col_ts_.data()),
    //         thrust::raw_pointer_cast(edge_id_.data()),
    //         thrust::raw_pointer_cast(out_nodes.data()),
    //         thrust::raw_pointer_cast(out_nodes_ptr.data()),
    //         thrust::raw_pointer_cast(out_ts.data()),
    //         thrust::raw_pointer_cast(out_ts_ptr.data()),
    //         thrust::raw_pointer_cast(out_eid.data()),
    //         (int)chunks.size(), 
    //         t_begin, t_end,
    //         thrust::raw_pointer_cast(counts_nodes_buffer.data()),
    //         thrust::raw_pointer_cast(counts_ts_buffer.data()),
    //         thrust::raw_pointer_cast(counts_col_buffer.data()),
    //         use_full_timestamps,
    //         thrust::raw_pointer_cast(prefix_different_timestamp.data())
    //     );
    //     TemporalRoot<T,TS> result(
    //         std::move(out_nodes),
    //         std::move(out_nodes_ptr),
    //         std::move(out_ts),
    //         std::move(out_ts_ptr),
    //         std::move(out_eid)
    //     );
    //     return std::move(result);
    // }
    NegativeRoot<T,TS> get_negative_root(thrust::device_vector<T> &negative_root, thrust::device_vector<TS> &negative_time){
        return NegativeRoot<T,TS>(std::move(negative_root),std::move(negative_time));
    }

    TemporalResult<T,TS> sample_src_in_chunks_khop(
            TemporalRoot<T,TS> seeds_root, NegativeRoot<T,TS> neg_root, int k, int layers, TS allowed_offset, bool equal_root_time, bool keep_root_time, std::string type)
    {

        thrust::device_vector<T>  seeds = seeds_root.roots;
        thrust::device_vector<TS> seed_ts = seeds_root.ts;
        thrust::device_vector<T>  seed_ptr = seeds_root.ts_ptr;
        thrust::device_vector<T>  neg_seeds = neg_root.roots;
        thrust::device_vector<TS> neg_seed_ts = neg_root.ts;
        int ts_num = seed_ptr.back();
        thrust::device_vector<T>  neg_seed_ptr(neg_seeds.size() + 1);
        thrust::sequence(neg_seed_ptr.begin(), neg_seed_ptr.end(), ts_num + 1);
        //thrust::sequence<T>(ts_num + 1, ts_num + neg_seeds.size()+1,);
        thrust::device_vector<T>  root     = concat(seeds_root.roots, neg_root.roots, stream_);
        thrust::device_vector<TS> root_ts  = concat(seeds_root.ts,     neg_root.ts, stream_);
        thrust::device_vector<T> root_ptr = concat(seed_ptr, neg_seed_ptr, stream_);
        ts_num = root_ptr.back();
        int root_num = root.size();
        int warps = 4, threads = warps * 32;
        int blocks = (root_num + warps - 1) / warps;
        TemporalResult<T,TS> out(seeds_root);
        for(int l = 0; l < layers ; l++){
            TemporalNeighbor<T,TS> result;
            thrust::device_vector<T> prefix_(ts_num+1, 0);
            thrust::device_vector<T> start_offset_(ts_num+1, 0);
            thrust::device_vector<T> start_pos(ts_num+1, 0);
            thrust::device_vector<T> end_pos(ts_num+1, 0);
            cudaStreamSynchronize(stream_);
            // for(int i = 0;i < root_num;i++){
            //     printf("%lld\n",seeds_root.roots[i]);
            // }
            // for(int i = 0 ;i < root_num; i++){
            //     printf("%lld %lld\n", root[i], root_ptr[i]);
            // }
            // for(int i = 0;i < ts_num;i++){
            //     printf("%lld\n", root_ts[i]);
            // }
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
                    thrust::make_zip_iterator(thrust::make_tuple(start_pos.begin(), end_pos.begin(), start_offset_.begin())),
                    thrust::make_zip_iterator(thrust::make_tuple(start_pos.end(),   end_pos.end(), start_offset_.end())),
                    thrust::make_zip_iterator(thrust::make_tuple(start_pos.begin(), prefix_.begin())),
                    [] __host__ __device__ (const thrust::tuple<T, T, T>& t) {
                        
                        return thrust::make_tuple(thrust::get<0>(t)+thrust::get<2>(t), thrust::get<1>(t) - thrust::get<0>(t) - thrust::get<2>(t));
                    }
                );
                result.root_start_ptr.resize(prefix_.size(),0);
                result.root_end_ptr.resize(prefix_.size(),0);

                thrust::exclusive_scan(thrust::cuda::par.on(stream_),
                                   prefix_.begin(),
                                   prefix_.end(),
                                   prefix_.begin());
                thrust::copy(
                    prefix_.begin(), prefix_.end() - 1,
                    result.root_start_ptr.begin()
                );
                thrust::copy(
                    prefix_.begin() + 1, prefix_.end(),
                    result.root_end_ptr.begin()
                );
                thrust::device_vector<T> new_neighbors(prefix_.back());
                thrust::device_vector<TS> new_neighbors_ts(prefix_.back());
                thrust::device_vector<T> new_neighbors_eid(prefix_.back());
                thrust::device_vector<T> new_neighbors_dt(prefix_.back());
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
                    thrust::raw_pointer_cast(new_neighbors_eid.data()),
                    thrust::raw_pointer_cast(new_neighbors_dt.data())
                );
                result.neighbors = std::move(new_neighbors);
                result.neighbors_ts = std::move(new_neighbors_ts);
                result.neighbors_eid = std::move(new_neighbors_eid);
                result.neighbors_dt = std::move(new_neighbors_dt);
                
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
                thrust::device_vector<T> new_neighbors_dt(prefix_.back());
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
                    thrust::raw_pointer_cast(new_neighbors_dt.data()),
                    thrust::raw_pointer_cast(start_pos.data()),
                    thrust::raw_pointer_cast(end_pos.data()),
                    keep_root_time
                );
                result.neighbors = std::move(new_neighbors);
                result.neighbors_ts = std::move(new_neighbors_ts);
                result.neighbors_eid = std::move(new_neighbors_eid);
                result.neighbors_dt = std::move(new_neighbors_dt);
                result.root_start_ptr.resize(prefix_.size(),0);
                thrust::copy(
                    prefix_.begin(), prefix_.end(),
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
            // printf("neighbors length is %d %d\n", root_num, ts_num);
            // thrust::host_vector<T> h_root = root;
            // thrust::host_vector<TS> h_root_ts = root_ts;
            // thrust::host_vector<T> h_root_ptr = root_ptr;
            // for(int i = 0; i<root_num ;i++){
            //     printf("%lld %lld %lld\n", h_root[i], h_root_ts[i], h_root_ptr[i]);
            // }
        
            // for(int i = 0 ;i < root_num; i++){
            //     printf("%d %d %d\n", root[i], root_ts[i],root_ptr[i]);
            // }
        }
        return std::move(out);
    }
};

static __global__ void mark_new_ids_kernel(
    const T* global_ids,
    T* local_ids,
    const T* global_to_local,
    const T* global_mask,
    int num_ids, T mask_flag
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_ids) return;
    T gid = global_ids[i];
    //printf("gid: %lld, global_to_local: %lld, global_mask: %lld num ids:%d\n", gid, global_to_local[gid], global_mask[gid],num_ids);
    if(global_mask[gid] == mask_flag){
        local_ids[i] = global_to_local[gid];
    }
    else{
        local_ids[i] = -1;
    }
}
static __global__ void update_new_ids_kernel(
    const T* global_ids,
    T* local_ids,
    const T* global_to_local,
    T* global_mask,
    int num_ids, T mask_flag
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_ids) return;
    T gid = global_ids[i];
    if(local_ids[i] == -1){
        local_ids[i] = global_to_local[gid];
        global_mask[gid] = mask_flag;
    }
}
template <typename T = int64_t, typename TS = int64_t>
class Remapper{
    public:
    thrust::device_vector<T> id_mapper;
    long long times_flag = 1;
    thrust::device_vector<T> global_mask;
    int id_counts;
    void initlization(int num_nodes){
        id_mapper.resize(num_nodes,0);
        global_mask.resize(num_nodes,0);
        times_flag = 1;
        id_counts = 0;
    }
   
    thrust::device_vector<T> remapper(thrust::device_vector<T> ids, 
                                            thrust::device_vector<T> & insert_ids,
                                            thrust::device_vector<T> & insert_local_ids,
                                            cudaStream_t stream_){
        int num_ids = ids.size();
        thrust::device_vector<T> local_ids(num_ids,0);
        int warps = 4, threads = warps * 32;
        int blocks = (num_ids + warps - 1) / warps;
        printf("remapper num ids %d blocks %d\n", num_ids, blocks);
        if(ids.size() == 0){
            return thrust::device_vector<T>();
        }
        mark_new_ids_kernel<<<blocks,threads, 0, stream_>>>(
            thrust::raw_pointer_cast(ids.data()),
            thrust::raw_pointer_cast(local_ids.data()),
            thrust::raw_pointer_cast(id_mapper.data()),
            thrust::raw_pointer_cast(global_mask.data()),
            num_ids, times_flag
        );
    
        thrust::device_vector<T> new_global_ids(num_ids);
        auto end = thrust::copy_if(
            thrust::cuda::par.on(stream_),
            ids.begin(), ids.end(),
            local_ids.begin(),
            new_global_ids.begin(),
            [] __device__ (int flag) { return flag == -1; }
        );
        printf("count %d\n", thrust::distance(new_global_ids.begin(), end));
        thrust::sort(thrust::cuda::par.on(stream_), new_global_ids.begin(), end);
        end = thrust::unique(thrust::cuda::par.on(stream_), new_global_ids.begin(), end);
        int new_id_count = thrust::distance(new_global_ids.begin(), end);
        insert_ids.resize(new_id_count,0);
        insert_local_ids.resize(new_id_count,0);
        thrust::sequence(thrust::cuda::par.on(stream_), insert_local_ids.begin(), insert_local_ids.end(), id_counts);
        id_counts += new_id_count;
        thrust::copy(
            new_global_ids.begin(), end,
            insert_ids.begin()
        );
        thrust::scatter(
            thrust::cuda::par.on(stream_),
            insert_local_ids.begin(), insert_local_ids.end(),
            insert_ids.begin(),
            id_mapper.begin()
        );
        printf("new id count %d\n", new_id_count);
        // 更新 global_mask 和 id_mapper
        if(new_id_count > 0){
            update_new_ids_kernel<<<(new_id_count + 255)/256, 256, 0, stream_>>>(
                thrust::raw_pointer_cast(ids.data()),
                thrust::raw_pointer_cast(local_ids.data()),
                thrust::raw_pointer_cast(id_mapper.data()),
                thrust::raw_pointer_cast(global_mask.data()),
                num_ids, times_flag);
        }
        return std::move(local_ids);
        
    }
    void insert(TemporalResult<T,TS> &result, bool reset, cudaStream_t stream_){
        if(reset){
            times_flag += 1;
            id_counts = 0;

        }
        thrust::device_vector<T>  insert_ids;
        thrust::device_vector<T>  insert_local_ids;
        printf("root number is %d\n",result.roots.roots.size());
        result.roots.roots = remapper(
            result.roots.roots, insert_ids, insert_local_ids, stream_
        );

        // 重映射每一层的 neighbors
        int lr = 0;
        for (auto& layer : result.neighbors_list) {
            thrust::device_vector<T>  layer_insert_ids;
            thrust::device_vector<T>  layer_insert_local_ids;
            printf("layers number is %d %d\n",lr, layer.neighbors.size());
            layer.neighbors = std::move(remapper(
                layer.neighbors, layer_insert_ids, layer_insert_local_ids, stream_
            ));
            lr ++;
            insert_ids = concat(insert_ids, layer_insert_ids, stream_);
            insert_local_ids = concat(insert_local_ids, layer_insert_local_ids, stream_);
        }
        result.nodes_remapper_id = std::move(insert_ids);
    }

};

template<typename T, typename TS>
class AsyncTemporalSampler {
    private:
    struct Task {
        TS time_start;
        TS time_end;
        thrust::device_vector<T> chunk_list;
        TemporalRoot<T,TS> roots;
        NegativeRoot<T,TS> neg_seeds;

        std::string sample_type; // "c"是cluster加载，"r"是recent, "u"是uniform
        int layers;//""采样层数
        int fanout;//采样邻居数
        int k;
        int layer;
        TS allowed_offset; 
        bool equal_root_time;
        bool keep_root_time;
        std::string op; //"f" follow, "r" 重映射
        std::promise<TemporalResult<T,TS>> promise;
        Task() = default;
    };
    template<typename Q>
    class ThreadSafeQueue {
        std::queue<Q> q;
        mutable std::mutex m;
        std::condition_variable cv;
        std::atomic<bool> closed{false};
        public:
        void push(Q &&task) {
            std::lock_guard<std::mutex> lock(m);
            if (closed) return;
            q.push(std::move(task));
            cv.notify_one();
        }
        std::optional<Q> pop() {
            std::unique_lock<std::mutex> lock(m);
            cv.wait(lock, [&] { return !q.empty() || closed; });
            if (q.empty()) return std::nullopt;
            auto task = std::move(q.front());
            q.pop();
            return task;
        }

        void close() { closed = true; cv.notify_all(); }
    };
    std::shared_ptr<ThreadSafeQueue<std::unique_ptr<Task>>> task_queue;
    ThreadSafeQueue<std::future<TemporalResult<T,TS>>> result_queue;
    
    struct Worker {
        cudaStream_t stream;
        std::shared_ptr<Graph<T,TS>> graph;
        std::thread thread;
        std::atomic<bool> running{true};
        int rank;
        std::shared_ptr<ThreadSafeQueue<std::unique_ptr<Task>>>task_queue_ptr;
        ThreadSafeQueue<std::future<TemporalResult<T,TS>>> result_queue_ptr;

        std::shared_ptr<Remapper<T,TS>> res;
        Worker(std::shared_ptr<Graph<T,TS>> g, int rank, cudaStream_t stream_, std::shared_ptr<ThreadSafeQueue<std::unique_ptr<Task>>> task_queue_): //ThreadSafeQueue<std::future<TemporalResult<T,TS>>> *result_queue_) : 
            graph(g), rank(rank),stream(stream_), task_queue_ptr(task_queue_){
            //cudaStreamCreate(&stream);
            //std::cout<<g->chunk_ptr_.size()<<std::endl;
            thread = std::thread([this]() { this->run(); });
            res = std::make_shared<Remapper<T,TS>>();
            res->initlization(g->get_num_nodes());
        }

        ~Worker() {
            running = false;
            if (thread.joinable()) thread.join();
        }


        void run() {
            cudaSetDevice(rank);  // 或多卡支持
            while (running) {
                std::optional<std::unique_ptr<Task>> task_ = task_queue_ptr->pop();
                if (!task_) continue;
                std::unique_ptr<Task> task = std::move(*task_);
                try {
                    std::cout<<"get new tasks"<<std::endl;
                    std::cout<<"task:"<<" "<<task->time_start<<" "<<task->time_end<<" ";
                    for(int i = 0 ;i<task->chunk_list.size();i++)std::cout<<task->chunk_list[i]<<" ";
                    std::cout<<std::endl;
                    TemporalResult<T,TS> result;
                    if (task->sample_type == "recent" || task->sample_type == "uniform") {
                        task->roots = graph->get_seeds_in_chunks(task->chunk_list, 
                        task->time_start, task->time_end, 0);

                        result = graph->sample_src_in_chunks_khop(
                            task->roots, task->neg_seeds,
                            task->fanout, task->layers,
                            task->allowed_offset,
                            task->equal_root_time,
                            task->keep_root_time,
                            task->sample_type
                        );
                    }
                    else{
                        result = graph->slice_by_chunk_ts(
                            task->chunk_list,
                            task->time_start,
                            task->time_end,
                            true,
                            stream
                        );
                    }
                    res->insert(result, (task->op == "f"), stream);
                    task->promise.set_value(result);
                } catch (...) {
                    task->promise.set_exception(std::current_exception());
                }
            }
        }
    };
    std::vector<std::unique_ptr<Worker>> workers;
    // 线程安全的队列


    public:
    AsyncTemporalSampler();
    AsyncTemporalSampler(std::shared_ptr<Graph<T,TS>> graph, int rank, int num_workers)
    {
        task_queue = std::make_shared<ThreadSafeQueue<std::unique_ptr<Task>>>();
        for (int i = 0; i < num_workers; ++i) {
            workers.emplace_back(std::make_unique<Worker>(graph, rank, graph->get_stream(), task_queue));
        }
    }

    ~AsyncTemporalSampler() {
        task_queue->close();
    }
    // Python 侧调用：提交异步采样任务
    void submit_query(
                        TS time_start,
                        TS time_end,
                        thrust::device_vector<T> chunk_list,
                        NegativeRoot<T,TS> negative_root,
                        std::string sample_type, // "c"是cluster加载，"r"是recent, "u"是uniform
                        int layers,//""采样层数
                        int fanout,//采样邻居数x
                        TS allowed_offset, 
                        bool equal_root_time,
                        bool keep_root_time,
                        std:: string op //"f" follow, "r" 重映射
    ) {
        
        auto task = std::make_unique<Task>();
        task->time_start = time_start;
        task->time_end = time_end;
        task->chunk_list = std::move(chunk_list);
        //task->seeds = std::move(seeds);
        task->neg_seeds = std::move(negative_root);
        task->fanout = fanout;
        task->layers = layers;
        task->sample_type = sample_type;
        task->allowed_offset = allowed_offset;
        task->equal_root_time = equal_root_time;
        task->keep_root_time = keep_root_time;
        task->op = op;
        std::future<TemporalResult<T,TS>> fut = task->promise.get_future();
        task_queue->push(std::move(task));
        result_queue.push(std::move(fut));
    }
    std::optional<TemporalResult<T,TS>> get(){
        std::optional<std::future<TemporalResult<T,TS>>> res_ = result_queue.pop();
        if(!res_){
            return std::nullopt;
        }
        else{
            return std::move(res_->get());
        }
        
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
   //using Block_T = TemporalBlock<T, TS>;

    std::shared_ptr<Graph_T> graph_;
    int rank;
    AsyncTemporalSampler<T,TS> *sampler;
    CUDAGraph(std::shared_ptr<Graph_T>  g, int rank) : graph_(std::move(g)),  sampler(nullptr)
    {
        this->rank = rank;
        // 初始化采样器
        try {
            sampler = new AsyncTemporalSampler<T, TS>(graph_, rank, 1);
        } catch (const c10::Error& e) {
            throw std::runtime_error(std::string("AsyncTemporalSampler construction failed: ") + e.msg());
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("AsyncTemporalSampler construction failed: ") + e.what());
        } catch (...) {
            throw std::runtime_error("AsyncTemporalSampler construction failed with unknown error");
        }
    }
    ~CUDAGraph() = default;

    
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
        return graph_->sample_src_in_chunks_khop(
            seeds_root,  neg_seeds_root, k, layer, allowed_offset, equal_root_time, keep_root_time, type);
    
    }

    TemporalRoot<T,TS> get_seeds_in_chunks(const torch::Tensor& chunks_tensor,
                                     TS time_begin, TS time_end,
                                     bool using_full_timestamp) {
        thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunks_tensor);
        return graph_->get_seeds_in_chunks(chunks, time_begin, time_end, using_full_timestamp);
    }

    NegativeRoot<T,TS> get_negative_root(const torch::Tensor& negative_root_tensor,
                                        const torch::Tensor& negative_time_tensor){
        thrust::device_vector<T> negative_root = tensor_to_device_vector<T>(negative_root_tensor);
        thrust::device_vector<TS> negative_time = tensor_to_device_vector<TS>(negative_time_tensor);
        return graph_->get_negative_root(negative_root,negative_time);
    }

    TemporalResult<T,TS> slice_by_chunk_ts(const torch::Tensor& chunks_tensor, 
                        TS time_begin, TS time_end, bool using_full_timestamp, uint64_t py_stream) {
        thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunks_tensor);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(py_stream);
        return graph_->slice_by_chunk_ts(chunks, time_begin, time_end, using_full_timestamp, stream);
    }

    void submit_query(
                        TS time_start,
                        TS time_end,
                        torch::Tensor chunk_list,
                        torch::Tensor test_generate_samples,
                        torch::Tensor test_generate_samples_ts,
                        std::string sample_type, // "c"是cluster加载，"r"是recent, "u"是uniform
                        int layers,//""采样层数
                        int fanout,//采样邻居数
                        TS allowed_offset, 
                        bool equal_root_time,
                        bool keep_root_time,
                        std:: string op //"f" follow, "r" 重映射
    ) 
    {
        thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunk_list);
        sampler->submit_query(
            time_start, time_end,
            chunks,
            get_negative_root(test_generate_samples, test_generate_samples_ts),
            sample_type,
            layers,
            fanout,
            allowed_offset,
            equal_root_time,
            keep_root_time,
            op 
        );
    }
    std::optional<TemporalResult<T,TS>> get() {
        return sampler->get();
    }
    private:
    template <typename DType>
    thrust::device_vector<DType> tensor_to_device_vector(const torch::Tensor& tensor) {
        TORCH_CHECK(tensor.is_cuda(), "Input tensor must be on CUDA device");
        auto cont_tensor = tensor.contiguous();
        size_t num_elements = tensor.numel();
        thrust::device_vector<DType> vec(num_elements);
        cudaMemcpyAsync(
            thrust::raw_pointer_cast(vec.data()), 
            cont_tensor.data_ptr<DType>(), 
            num_elements * sizeof(DType), 
            cudaMemcpyDeviceToDevice, 
            graph_->get_stream()
        );
        return std::move(vec);
    }
};

// // CUDA 转换函数
CUDAGraph from_edge_index(
    T n, T chunk_size,
    const torch::Tensor& src,
    const torch::Tensor& dst,
    const torch::Tensor& ts,
    const torch::Tensor & edge_id,
    const torch::Tensor& row_chunk_mapper,
    uint64_t stream = 0,
    int rank = 0
) {

    //print f("from_edge_index\n");
    try{
        auto g = std::make_shared<Graph<T, TS>>(
            n, chunk_size, src, dst, ts, edge_id, row_chunk_mapper, stream, rank
        );
        return CUDAGraph(g, rank);
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