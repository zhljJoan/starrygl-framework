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

    T* __restrict__ chunk_counts_buffer,
    T* __restrict__ counts_buffer,
    T* __restrict__ chunk_nodes_counts_buffer
) {
    //chunk_num个warp处理每个chunk里面的node，每个warp里面的32个lane处理node的edge
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id;
    if (global_warp_id >= num_chunks) return;

    T chunk = chunks[global_warp_id];
    T chunk_start = chunk_ptr[chunk];
    T chunk_end   = chunk_ptr[chunk + 1];
    if (chunk_start >= chunk_end) return;
    T local_count_row = 0;
    T total_warp_count = 0;
    T total_row_count = 0;
    for (T node_id_ = chunk_start; node_id_ < chunk_end; node_id_ ++) {
        T node_idx = row_idx[node_id_];
        if (node_idx >= num_nodes) break;

        T row = node_idx;
        T rs = row_ptr[node_idx];
        T re = row_ptr[node_idx + 1];
        if (rs >= re) continue;

        T left  = lower_bound(col_ts, rs, re, t_begin);
        T right = upper_bound(col_ts, left, re, t_end);
        if (left >= right) continue;
        T local_count = 0;
        for (T e = left + lane; e < right; e += 32) {
            if (e >= right) break;
            T dst_c = col_chunk[e];
            if (dst_c < num_nodes && chunk_exists[dst_c]) {
                local_count++;
            }
        }
        T warp_count = warp_reduce_sum(local_count);
        if(lane == 0 && warp_count > 0) {
            counts_buffer[row] = total_warp_count;
            total_warp_count += warp_count;
            total_row_count += 1;
        }
    }
    chunk_counts_buffer[global_warp_id] = total_warp_count;
    chunk_nodes_counts_buffer[global_warp_id] = total_row_count;
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
    T* __restrict__ out_col_idx,
    TS* __restrict__ out_col_ts,
    T * __restrict__ out_col_eid,
    T num_chunks,
    T num_nodes,
    TS t_begin,
    TS t_end,
    const T* __restrict__ chunk_counts_buffer,
    const T* __restrict__ counts_buffer,
    const T* __restrict__ chunk_nodes_counts_buffer
) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int warps_per_block = blockDim.x / 32;
    int global_warp_id = blockIdx.x * warps_per_block + warp_id;

    if (global_warp_id >= num_chunks) return;

    T chunk = chunks[global_warp_id];
    T chunk_start = chunk_ptr[chunk];
    T chunk_end   = chunk_ptr[chunk + 1];
    if (chunk_start >= chunk_end) return;

    for (T node_idx = chunk_start + lane; node_idx < chunk_end; node_idx += 32) {
        if (node_idx >= num_nodes) break;

        T row = row_idx[node_idx];
        T rs = row_ptr[node_idx];
        T re = row_ptr[node_idx + 1];
        if (rs >= re) continue;

        T left  = lower_bound(col_ts, rs, re, t_begin);
        T right = upper_bound(col_ts, left, re, t_end);
        if (left >= right) continue;
        out_row_ptr[node_idx - chunk_start + chunk_nodes_counts_buffer[global_warp_id]]= counts_buffer[row] + chunk_counts_buffer[global_warp_id]; 
        out_row_idx[node_idx - chunk_start + chunk_nodes_counts_buffer[global_warp_id]]= row;
        T write_base = counts_buffer[row] + chunk_nodes_counts_buffer[global_warp_id];
        T written = 0;
        for (T e = left + lane; e < right ; e ++) {
            if (e >= right) break;
            T dst_c = col_chunk[e];
            if (dst_c < num_nodes && chunk_exists[dst_c]) {
                T pos = write_base + written;
                out_col_idx[pos] = col_idx[e];
                out_col_ts[pos]  = col_ts[e];
                out_col_eid[pos] = col_eid[e];
                written++;
            }
        }
    }
}

__global__ void filter_seeds_kernel(
    const T* chunk_ptr, const T* row_ptr, const T* row_idx, const T* col_idx, const TS* col_ts,
    const T* chunks, T num_chunks, TS t_begin, TS t_end, T* counts,
    T *chunk_counts_buffer) {
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gw = blockIdx.x * (blockDim.x / 32) + wid;
    if (gw >= num_chunks) return;
    T c = chunks[gw], cs = chunk_ptr[c], ce = chunk_ptr[c + 1];
    __shared__ T shared_counts[32+1];
    for (T r = cs + lane; r < ce; r += 32) {
        T rs = row_ptr[row_idx[r]], re = row_ptr[row_idx[r] + 1];
        if (rs >= re) continue;
        T tb = lower_bound(col_ts, rs, re, t_begin);
        T te = upper_bound(col_ts, tb, re, t_end);
        counts[r] = (int)(te - tb);
        shared_counts[lane+1] += (te-tb);
        atomicAdd((int*)&chunk_counts_buffer[gw], (int)(te - tb));
    }
    __syncwarp();
    if(lane == 0){
        for(int i = 2 ; i<warpSize;i++)shared_counts[i]+=shared_counts[i-1];
    }
    for (T r = cs+lane+32 ; r<ce; r+=32){
        counts[r] += counts[r-32]; 
    }
    for(T r = cs+lane; r < ce; r+=32){
        counts[r] += shared_counts[lane];
    }
}
template <typename T = int64_t, typename TS = int64_t>
__global__ void collect_seeds_kernel( 
    const T* prefix, const T* chunk_ptr, const T*chunks,
    const T* row_ptr, const T* row_idx, 
    const T* col_idx, const TS* col_ts, const T* col_eid,
    T* out_nodes, TS* out_ts, T* out_eid, T num_chunks, T len, T neg_len, const T* neg_node_list, T *chunk_counts_buffer,
    TS t_begin, TS t_end
) {
    
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int gw = blockIdx.x * (blockDim.x / 32) + wid;
    if (gw >= num_chunks) return;
    T c = chunks[gw], cs = chunk_ptr[c], ce = chunk_ptr[c + 1];
    for(int i = cs+lane; i<ce;i+=32){
        T start = prefix[i] + chunk_counts_buffer[gw];
        T r = row_idx[i];
        T rs = row_ptr[r], re = row_ptr[r + 1];
        if (rs >= re) continue;
        T tb = lower_bound(col_ts, rs, re, t_begin);
        T te = upper_bound(col_ts, tb, re, t_end);
        if(te>=tb)return;
        curandState state;
        curand_init(i, i, 0, &state);
        for (T j = tb, k = 0; j < te ; ++j, ++k) {
            out_nodes[start + k ] = r;
            out_nodes[start + k + len] = col_idx[j];
            out_eid[start + k] = col_eid[j];
            out_ts[start + k] = col_ts[j];
            out_ts[start + k + len] = col_ts[j];  
            out_ts[start + k + len + len] = col_ts[j];
            out_nodes[start + k + len + len] = neg_node_list[(T)curand_uniform(&state)*neg_len];
        }
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
__global__ void static_neighbor_num(
    const T* nodes, const TS* node_ts, T num_nodes, T k,
    const T* row_ptr, const T* col_idx, const TS* col_ts, const T* col_chunk, T* out_cnt, const bool* chunk_exists
){
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    for(int i = start; i < num_nodes; i+= blockDim.x * blockDim.x){
        if (i >= num_nodes) return;
        T node = nodes[i];
        T ts = node_ts[i];
        T rs = row_ptr[node], re = row_ptr[node + 1];
        if (rs >= re) continue;
        T te = lower_bound(col_ts, rs, re, node_ts[i]);
        out_cnt[i] = re-rs;
    }
}
template <typename T = int64_t, typename TS = int64_t>
__global__ void sample_single_hop_kernel(
    const T* nodes, const TS* node_ts, T num_nodes, T k,
    const T* row_ptr, const T* col_idx, const TS* col_ts, const TS* col_eid, const T* col_chunk, TS t_begin, TS t_end, const T* counts, T* out_nbr, TS* out_ts, T* out_eid, const bool* chunk_exists) {
        int start = blockIdx.x * blockDim.x + threadIdx.x;
        for(int i = start; i < num_nodes; i+= blockDim.x * blockDim.x){
            if (i >= num_nodes) return;
            T node = nodes[i];
            T ts = node_ts[i];
            T rs = row_ptr[node], re = row_ptr[node + 1];
            if (rs >= re) continue;
            T te = lower_bound(col_ts, rs, re, node_ts[i]);
            T tb = max(rs, te-k);
            for(int e = tb; e < te; e++){
                T pos = counts[i] + e - tb;
                out_nbr[pos] = col_idx[pos];
                out_ts[pos] = col_ts[pos];
                out_eid[pos] = col_eid[pos];
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
//     const T* nodes, const TS* node_ts, T num_nodes, T k,
//     const T* row_ptr, const T* col_idx, const TS* col_ts, const T* col_chunk, T* out_cnt, const bool* chunk_exists
// );
// template <typename T = int64_t, typename TS = int64_t>
// __global__ void sample_single_hop_kernel(
//     const T* in_nodes, const TS* in_ts,
//     //const T* in_nodes, const TS* in_ts,
//     T n, T k,
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
        : roots(std::move(r)), ts_ptr(std::move(tp)), ts(std::move(t)){}
    TemporalRoot(thrust::device_vector<T>&& r, thrust::device_vector<T>&& tp, thrust::device_vector<TS>&& t, thrust::device_vector<T>&& e)
        : roots(std::move(r)), ts_ptr(std::move(tp)), ts(std::move(t)), q_eid(std::move(e)){}
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
template <typename T, typename TS>
class Graph {
    thrust::device_vector<T>  chunk_ptr_, row_ptr_, row_idx_, col_idx_, col_chunk_, edge_id_, src_idx_;
    thrust::device_vector<TS> col_ts_;
    T  num_nodes_, num_edges_, chunk_size_;
    cudaStream_t stream_;
    thrust::device_vector<T> counts_buffer;
    thrust::device_vector<T> chunk_counts_buffer;
    thrust::device_vector<T> chunk_nodes_counts_buffer;
    public:
    Graph(T n, T chunk_size, const torch::Tensor& src, const torch::Tensor& dst,
        const torch::Tensor& ts, const torch::Tensor& row_chunk_mapper,
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

            // ==============================================================
            // 11. 初始化计数缓冲
            // ==============================================================
            counts_buffer.resize(num_nodes_, 0);
            chunk_counts_buffer.resize(chunk_size_ + 1, 0);
            chunk_nodes_counts_buffer.resize(chunk_size_ + 1, 0);

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
    TemporalBlock<T, TS> slice_by_chunk_ts(
        const thrust::device_vector<T>& chunks,
        TS t_begin, TS t_end, cudaStream_t stream = nullptr
    ) {
        thrust::device_vector<bool> chunk_exists(chunk_size_, false);
        thrust::device_vector<T> sorted_chunks = chunks;
        stream = stream_;
        thrust::sort(thrust::cuda::par.on(stream), sorted_chunks.begin(), sorted_chunks.end());
        mark_chunks_exists_kernel<<<(sorted_chunks.size() + 255)/256, 256, 0, stream>>>(
            thrust::raw_pointer_cast(sorted_chunks.data()), (T)sorted_chunks.size(),
            thrust::raw_pointer_cast(chunk_exists.data()), chunk_size_);
        // 2. 阶段 1：统计
        dim3 block(256);
        dim3 grid((chunks.size() + 7) / 8);
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
            thrust::raw_pointer_cast(chunk_counts_buffer.data()),
            thrust::raw_pointer_cast(counts_buffer.data()),
            thrust::raw_pointer_cast(chunk_nodes_counts_buffer.data())

        );

        // 3. 计算全局 row_ptr
        T query_chunk_size = chunks.size();
        thrust::device_vector<T> d_total_warp(query_chunk_size + 1, 0);
        thrust::device_vector<T> d_node_warp(query_chunk_size + 1, 0);
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            counts_buffer.begin(), counts_buffer.begin() + query_chunk_size + 1,
            d_total_warp.begin()
        );
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            chunk_nodes_counts_buffer.begin(), chunk_nodes_counts_buffer.begin() + query_chunk_size + 1,
            d_node_warp.begin()
        );
        thrust::device_vector<T> out_row_ptr(d_node_warp.back()+1);
        thrust::device_vector<T> out_row_idx(d_node_warp.back()+1);  
        thrust::device_vector<T> out_col_idx(d_total_warp.back());
        thrust::device_vector<T> out_col_ts(d_total_warp.back());
        thrust::device_vector<T> out_col_eid(d_total_warp.back());
        // 4. 阶段 2：写入
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
                       thrust::raw_pointer_cast(out_col_idx.data()),
                       thrust::raw_pointer_cast(out_col_ts.data()),
                       thrust::raw_pointer_cast(out_col_eid.data()),
                       query_chunk_size, num_nodes_, t_begin, t_end,
                       thrust::raw_pointer_cast(d_total_warp.data()),
                       thrust::raw_pointer_cast(counts_buffer.data()),
                       thrust::raw_pointer_cast(d_node_warp.data())
                   );
        // 5. 构造返回
        TemporalBlock<T, TS> block_res(
            std::move(out_col_idx),
            std::move(out_col_ts),
            std::move(out_col_eid),
            std::move(out_row_ptr),
            std::move(out_row_idx),
            0
        );
        return block_res;
    }
    TemporalBlock<T,TS> sample_src_in_chunks_khop(
        const thrust::device_vector<T>& chunks, T k, int layers,
        TS time_begin, TS time_end, const thrust::device_vector<T> &neg_seeds) {

        if (chunks.empty() || layers <= 0) return {};

        // ---- 标记存在的 chunk ----
        thrust::device_vector<bool> chunk_exists(chunk_size_, false);
        thrust::device_vector<T> sorted_chunks = chunks;
        thrust::sort(thrust::cuda::par.on(stream_), sorted_chunks.begin(), sorted_chunks.end());
        mark_chunks_exists_kernel<<<(sorted_chunks.size() + 255)/256, 256, 0, stream_>>>(
            thrust::raw_pointer_cast(sorted_chunks.data()), (T)sorted_chunks.size(),
            thrust::raw_pointer_cast(chunk_exists.data()), chunk_size_);

        // ---- 取种子 ----
        thrust::device_vector<T>  seeds, seed_eids;
        thrust::device_vector<TS>  seed_ts;
        get_seeds_in_chunks(chunks, time_begin, time_end, seeds, seed_ts, seed_eids, neg_seeds);
        if (seeds.empty()) return TemporalBlock<T,TS>{};

        // ---- 多跳采样 ----
        //thrust::device_vector<T>  cur_nodes = seeds;
        //thrust::device_vector<TS> cur_ts    = seed_ts;
        T * cur_nodes = thrust::raw_pointer_cast(seeds.data());
        TS * cur_ts = thrust::raw_pointer_cast(seed_ts.data());
        TemporalBlock<T,TS> result; result.layer_ptr.push_back(0);
        result.row_idx.resize(seeds.size());
        result.row_ts.resize(seed_ts.size());
        result.seed_eid = std::move(seed_eids);
        thrust::copy(seeds.begin(), seeds.end(),result.row_idx.begin());
        thrust::copy(seed_ts.begin(), seed_ts.end(),result.row_ts.begin());
        T total_length = 0;
        T total_nbr_length = 0;
        T new_seed_size = seeds.size();
        int warps = 4, threads = warps * 32;
        int blocks = (new_seed_size + warps - 1) / warps;
        T shmem = BLOOM_BYTES * warps;

        for (int l = 0; l < layers && new_seed_size>0; ++l) {
            result.row_ptr.resize(total_length + new_seed_size + 1);

            thrust::device_vector<T> out_counts(new_seed_size, 0);
            static_neighbor_num<<<blocks,threads, shmem, stream_>>>(
                cur_nodes,cur_ts,
                //thrust::raw_pointer_cast(cur_nodes.data()),
                //thrust::raw_pointer_cast(cur_ts.data()),
                new_seed_size,
                k,
                thrust::raw_pointer_cast(row_idx_.data()),
                thrust::raw_pointer_cast(col_idx_.data()),
                thrust::raw_pointer_cast(col_ts_.data()),
                thrust::raw_pointer_cast(col_chunk_.data()),
                thrust::raw_pointer_cast(out_counts.data()),
                thrust::raw_pointer_cast(chunk_exists.data())
            );
            thrust::exclusive_scan(thrust::cuda::par.on(stream_),
                               out_counts.begin(), 
                               out_counts.end(),
                               result.row_ptr.begin() + total_length
                            ); 
           
            T new_neighbor_count = result.row_ptr.back();  
        
            
            result.neighbors.resize(total_nbr_length + new_neighbor_count);
            result.neighbors_eid.resize(total_nbr_length + new_neighbor_count);
            result.neighbors_ts.resize(total_nbr_length + new_neighbor_count);
            // thrust::device_vector<T>  layer_nbr();
            // thrust::device_vector<TS> layer_ts ();
            // thrust::device_vector<T>  layer_cnt();

           

            sample_single_hop_kernel<<<blocks, threads, shmem, stream_>>>(
                cur_nodes,cur_ts,
                //thrust::raw_pointer_cast(cur_nodes.data()),
                //thrust::raw_pointer_cast(cur_ts.data()),
                new_seed_size, k,
                thrust::raw_pointer_cast(row_ptr_.data()),
                thrust::raw_pointer_cast(col_idx_.data()),
                thrust::raw_pointer_cast(col_ts_.data()),
                thrust::raw_pointer_cast(edge_id_.data()),
                thrust::raw_pointer_cast(col_chunk_.data()),
                time_begin, time_end,
                thrust::raw_pointer_cast(result.row_ptr.data())+total_length ,
                thrust::raw_pointer_cast(result.neighbors.data())+total_nbr_length,
                thrust::raw_pointer_cast(result.neighbors_ts.data())+total_nbr_length,
                thrust::raw_pointer_cast(result.neighbors_eid.data())+total_nbr_length,
                thrust::raw_pointer_cast(chunk_exists.data())
            );

            
            total_length += new_seed_size + 1;              
            result.row_ptr.push_back(total_length);
           
            cur_nodes = thrust::raw_pointer_cast(result.neighbors.data())+total_nbr_length;
            cur_ts    = thrust::raw_pointer_cast(result.neighbors_ts.data())+total_nbr_length;
            total_nbr_length += new_neighbor_count;
            new_seed_size = new_neighbor_count;
        }
        return result;
    }

private:
    void get_seeds_in_chunks(const thrust::device_vector<T>& chunks,
                             TS t_begin, TS t_end,
                             thrust::device_vector<T>& out_nodes,
                             thrust::device_vector<TS>& out_ts,
                             thrust::device_vector<T> & out_eid,
                             const thrust::device_vector<T> & neg_seeds
                             ) {
        thrust::device_vector<T> counts(num_nodes_, 0);
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
            thrust::raw_pointer_cast(counts.data()),
            thrust::raw_pointer_cast(chunk_counts_buffer.data())
        );
        T local_chunk_size = num_nodes_;
        thrust::device_vector<T> prefix(local_chunk_size + 1, 0);
        thrust::exclusive_scan(thrust::cuda::par.on(stream_),
                               chunk_counts_buffer.begin(), 
                               chunk_counts_buffer.begin()+local_chunk_size+1,
                               prefix.begin()
                            );
        T total = prefix.back();
        if (total == 0) return;
        if(neg_seeds.empty()){
            out_nodes.resize(total + total); out_ts.resize(total + total);
        }
        else{
            out_nodes.resize(3*total); out_ts.resize(3*total);
        }
        out_eid.resize(total);
        
        collect_seeds_kernel<<<(num_nodes_ + 255)/256, 256, 0, stream_>>>(
            thrust::raw_pointer_cast(counts.data()),
            thrust::raw_pointer_cast(chunk_ptr_.data()),
            thrust::raw_pointer_cast(chunks.data()),
            thrust::raw_pointer_cast(row_ptr_.data()),
            thrust::raw_pointer_cast(row_idx_.data()),
            thrust::raw_pointer_cast(col_idx_.data()),
            thrust::raw_pointer_cast(col_ts_.data()),
            thrust::raw_pointer_cast(edge_id_.data()),
            thrust::raw_pointer_cast(out_nodes.data()),
            thrust::raw_pointer_cast(out_ts.data()),
            thrust::raw_pointer_cast(out_eid.data()),
            (T)chunks.size(), 
            total, 
            (T)neg_seeds.size(), 
            thrust::raw_pointer_cast(neg_seeds.data()),
            thrust::raw_pointer_cast(prefix.data()),
            t_begin, t_end
        );
    }
};



// __global__ void compress_output_kernel(
//     const T* prefix, const T* cnt, const T* in_nbr, const TS* in_ts,
//     T* out_nbr, TS* out_ts, T n, T k) {
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


    Block_T sample_src_in_chunks_khop(const torch::Tensor& chunks_tensor, 
                                     T k, int layer, 
                                     TS time_begin, TS time_end,
                                     const torch::Tensor& neg_seeds_tensor) {
        thrust::device_vector<T> neg_seeds = tensor_to_device_vector<T>(neg_seeds_tensor);
        thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunks_tensor);
        return graph_.sample_src_in_chunks_khop(chunks, k, layer, time_begin, time_end, neg_seeds);
    }

    TemporalBlock<T,TS> slice_by_chunk_ts(const torch::Tensor& chunks_tensor, 
                        TS time_begin, TS time_end, uint64_t py_stream) {
        thrust::device_vector<T> chunks = tensor_to_device_vector<T>(chunks_tensor);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(py_stream);
        return graph_.slice_by_chunk_ts(chunks, time_begin, time_end, stream);
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