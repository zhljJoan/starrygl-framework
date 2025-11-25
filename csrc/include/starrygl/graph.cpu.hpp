#pragma once
#include <ATen/Parallel.h>
#include <algorithm>
#include <random>
#include <vector>
#include <queue>
#include <set> 

#include <starrygl.hpp>
namespace starrygl{
    //template <typename T, typename N>
    // N uniform_sample(const T *begin, const T *end, const N k,T *outputs)
    // {

    //     size_t n = end - begin;

    //     if (k < n) {
    //         thread_local static std::random_device device;
    //         thread_local static std::mt19937 g(device());
    //         indices.resize(n);
    //         std::iota(indices.begin(), indices.end(), 0); 
    //         std::vector<size_t> sampled_indices(k);
    //         std::sample(indices.begin(), indices.end(), sampled_indices.begin(), k, g);
    //         indices = std::move(sampled_indices);
    //         return k;
    //     } else {
    //         indices.resize(n);
    //         std::iota(indices.begin(), indices.end(), 0);
    //         return n;
    //     }
    // }
 
    template <typename T, typename TS> 
    class TemporalEdge{
        T src_;
        T dst_;
        TS ts_;
        public:
        TemporalEdge(T src, T dst, TS ts): src_(src), dst_(dst), ts_(ts){

        };
    };
    static int parallel_worker_num = 10;
    void set_parallel_worker_num(int num){
        at::set_num_threads(num);
        parallel_worker_num = num;
    }
    template <typename T, typename TS>
    class TemporalBlock{
        std::vector<T> col_idx_;
        std::vector<T> row_ptr_;
        std::vector<T> neighbors_;
        std::vector<TS> neighbors_ts_;
        public:
        TemporalBlock( std::vector<T> neighbors, std::vector<T> neighbors_ts, std::vector<T>  row_ptr) :  neighbors_(std::move(neighbors)), neighbors_ts_(std::move(neighbors_ts)),row_ptr_(std::move(row_ptr)){
        }
        TemporalBlock(std::vector<T>  col_idx, std::vector<T> neighbors, std::vector<T> neighbors_ts, std::vector<T> row_ptr) : col_idx_(std::move(col_idx)),neighbors_(std::move(neighbors)), neighbors_ts_(std::move(neighbors_ts)), row_ptr_(std::move(row_ptr)){
        }
        std::vector<T> get_col_idx_(){
            return std::move(col_idx_);
        }
        std::vector<T> get_row_ptr_(){
            return std::move(row_ptr_);
        }
        std::vector<T> get_neighbors_(){
            return std::move(neighbors_);
        }
        std::vector<T> get_neighbors_ts(){
            return std::move(neighbors_ts_);
        }
        torch::Tensor to_tensor(std::vector<T> & vec){
            size_t total = vec.size();
            torch::Tensor out = torch::empty(total, torch::TensorOptions().dtype(torch::kLong));
            std::copy(vec.begin(), vec.end(), out.data_ptr<T>());
            return out;
        }
        torch::Tensor get_col_idx_tensor(){
            return to_tensor(col_idx_);
        }
        torch::Tensor get_row_ptr_tensor(){
            return to_tensor(row_ptr_);
        }
        torch::Tensor get_neighbors_tensor(){
            return to_tensor(neighbors_);
        }
        torch::Tensor get_neighbors_ts_tensor(){
            return to_tensor(neighbors_ts_);
        }
    };
    template <typename T, typename TS>
    class Graph<T, TS, CPU>{
        std::vector<T> chunk_ptr_;
        std::vector<T> row_reindex_;
        std::vector<T> row_idx_;
        std::vector<T> row_ptr_;
        std::vector<T> row_chunk_mapper_;
        std::vector<T> col_idx_;
        std::vector<TS> col_ts_;
        std::vector<T> col_chunk_;
        int chunk_size;
        public:
        Graph(){}
        bool cmp(const std::tuple<T,T,TS> &a, const std::tuple<T,T,TS>&b){
            T row_chunk_a = row_chunk_mapper_[std::get<0>(a)];
            T row_chunk_b = row_chunk_mapper_[std::get<0>(b)];
            if(row_chunk_a!=row_chunk_b){
                return row_chunk_a < row_chunk_b;
            }
            if(std::get<0>(a) != std::get<0>(b)){
                return std::get<0>(a) < std::get<0>(b);
            }
            if(std::get<2>(a)!=std::get<2>(b)){
                return std::get<2>(a) < std::get<2>(b);
            }
            if(row_chunk_mapper_[std::get<1>(a)] != row_chunk_mapper_[std::get<1>(b)]){
                return row_chunk_mapper_[std::get<1>(a)] < row_chunk_mapper_[std::get<1>(b)];
            }
            return std::get<1>(a) < std::get<1>(b);
        }
        Graph(T n, T chunk_size, std::vector<std::tuple<T,T,TS>> edge_index, std::vector<T> row_chunk_mapper): row_ptr_(n+1), col_idx_(edge_index.size()), col_ts_(edge_index.size()), col_chunk_(edge_index.size()){
            chunk_size = chunk_size;
            chunk_ptr_.resize(chunk_size+1);
            row_chunk_mapper_ = std::move(row_chunk_mapper);
            std::sort(edge_index.begin(), edge_index.end(), 
                [this](const auto& a, const auto& b) {
                return this->cmp(a, b);
            });
            T offset_=0, chunk = -1;
            T m = edge_index.size();
            for(T i = 0; i<n; i++){
                T row = std::get<0>(edge_index[offset_]);
                row_ptr_[i] = offset_;
                row_idx_[i] = row;
                row_reindex_[row] = i;
                while(row_chunk_mapper_[row]>chunk){
                    chunk_ptr_[++chunk] = offset_;
                }
                while(offset_<m && std::get<0>(edge_index[offset_])==row){
                    col_idx_[offset_] = std::get<1>(edge_index[offset_]);
                    col_ts_[offset_] = std::get<2>(edge_index[offset_]);
                    col_chunk_[offset_] = row_chunk_mapper_[col_idx_[offset_]];
                    offset_++;
                }
            }
            chunk_ptr_[chunk_size+1] = offset_;
            row_ptr_[n+1] = offset_;
        
         }
         virtual ~Graph() = default;

         size_t size() const{ return row_ptr_.size();}

         size_t edge_count() const { return col_idx_.size();}

         bool chunk_cmp(const T &a, const T &b){
            return chunk_ptr_[a]<chunk_ptr_[b];
         }
        //  std::uniform_sample_kernel(const std::vector<T> &inputs, int k) const{
        //     const size_t bs = inputs.size();
        //     std::vector<T> outputs;
        //     std::vector<T> output_counts(bs,0);
        //     std::vector<T> output_count_prefix_sum(bs+1,0);
        //     at::parallel_for(0, bs, 1, [&](size_t start, size_t end){
        //         for(size_t i = start; i<end; i++){
        //             T node = inputs[i];
        //             T begin = row_ptr_[node];
        //             T end = row_ptr_[node+1];
        //             if(end-begin <= k){
        //                 output_counts[i] = end-begin;
        //                 std::copy(col_idx_.begin()+begin, col_idx_.begin()+end, outputs.begin()+i*k);
        //             }
        //             else{
        //                 output_counts[i] = k;
        //                 uniform_sample(col_idx_.data()+begin, col_idx_.data()+end, k, outputs.data()+i*k);
        //             }
        //         }
        //     });
        //     for(size_t i=0; i<bs; i++){
        //         output_count_prefix_sum[i+1] = output_count_prefix_sum[i]+output_counts[i];
        //     }
        //     outputs.resize(output_count_prefix_sum[bs]);
        //     return outputs;
        //  }
        //  class SampleRequest{
        //     public:
        //         T nid_;
        //         TS ts_;
        //         T request_id_;
        //         int layer_;
        //         SampleRequest(T nid, TS ts, T request_id, int layer):nid_(nid),ts_(ts),request_id_(request_id),layer_(layer){}
        //         bool operator<(const SampleRequest &a){
        //             return layer_ == a.layer_? layer_<a.layer : ( nid_ == a.nid_ ? ts_<a.ts_ : nid_ < a.nid_);

        //         }
        //  }
        //  std::recent_sample_kernel(const std::priority_queue queue<std::pair<T,TS>> &inputs, int k, int l, 
        //                             const parallel_flat_hash_set<std::pair<T,TS>> &unique_output, 
        //                             const parallel_flat_hash_set<std::pair<T,TS>> &local_unique_output,
        //                             const std::vector<std::pair<T,TS>> output,
        //                             std::mutex &unique_mutex) const{
        //         count_id_ = 0;
        //         std::vector<T> neighbors_;
        //         std::vector<T> neighbors_ts_;
        //         std::vector<T> neighbors_row_;
        //         while(!inputs.empty()){
        //             auto input = inputs.top();
        //             if(input_layer_>l)return;
        //             inputs.pop(); 
        //             if(local_unique_output.find({input.nid_, input.ts_})!=local_unique_output.end()
        //                 && unique_output.find({input.nid_, input.ts_})!=unique_output.end()){
        //                 local_unique_output[{input.nid_, input.ts_}] = count_id_++;
        //                 T node = input.nid_;
        //                 T reidx_ = row_reindex_[node];
        //                 TS ts = input.ts_;
        //                 T begin = row_ptr_[reidx_];   
        //                 T end = row_ptr_[reidx_+1];
        //                 if(end - begin <= k){
        //                     neighbors_.insert(neighbors_.end(), col_idx_.begin() + begin, col_idx_.begin() +end);
        //                     neighbors_ts_.insert(neighbors_ts_.end(), col_ts_.begin()+begin, col_ts_.begin()+end);
        //                     neighbors_row_.insert(neighbor_row_.end(), end-begin, );

        //                 }
        //                 else{
        //                     neighbors_.emplace_back(col_idx_.data()+begin, col_idx_.data()+end, k, neighbors_.data()+count_id_*k);
        //                     neighbors_ts_.emplace_back(ts);
        //                 }
        //             }   
        //         }
            
        
        //  }

        
        // TemporalBLock<T,TS> sample_kernel(std::vector<std::pair<T,TS>> &inputs, int k, int layer) const{
        //     std::vector<std:thread> threads;
        //     std::priority_queue<Tuple<T,TS,int>> thread_inputs[parallel_worker_num];
        //     parallel_flat_hash_set<std::pair<T,TS>> unique_outputs;
        //     parallel_flat_hash_map<std::pair<T,TS>,int> local_unique_outputs[parallel_worker_num];
        //     std::mutex unique_mutex;
        //     for(int i = 0 ;i < inputs.size(); i++){
        //         thread_inputs[first%chunk_size].push(SampleRequest(inputs.first, inputs.second, i, 0));
        //     }
        //     std::vector<std::pair<T,TS>> outputs;
        //     std::vector<std::pair<T,TS>> output_local[parallel_work_num];

        //     for(int l = 0; l < layer; l++){   
        //         for(int i = 0; i<parallel_worker_num;i++){
        //             threads.emplace_back([&, i](){
        //                 std::parallel_flat_hash_set<T> unique_outputs;
        //                 for(const auto &input: thread_inputs[i]){}
        //         });
        //     }
            
        // } 

        TemporalBlock<T,TS> sample_kernel(std::vector<std::pair<T,TS>> &inputs, int k, int layer) const{
            //  std::vector<std:thread> threads;
            //  std::priority_queue<Tuple<T,TS,int>> thread_inputs[parallel_worker_num];
            //  parallel_flat_hash_set<std::pair<T,TS>> unique_outputs;
            //  parallel_flat_hash_map<std::pair<T,TS>,int> local_unique_outputs[parallel_worker_num];
            //  std::mutex unique_mutex;
            //  for(int i = 0 ;i < inputs.size(); i++){
            //      thread_inputs[first%chunk_size].push(SampleRequest(inputs.first, inputs.second, i, 0));
            //  }
            std::vector<int> layer_ptr(layer+1,0);
            std::vector<T> output_counts(inputs.size(),0);
            std::vector<T> output_count_prefix_sum(inputs.size()+1,0);
            size_t bs = inputs.size();
            
            at::parallel_for(0, bs, 1, [&](size_t start, size_t end){
                for(size_t i = start; i<end; i++){
                    T node = inputs[i].first;
                    TS ts = inputs[i].second;
                    T reidx_ = row_reindex_[node];
                    T begin = row_ptr_[reidx_];   
                    T end = row_ptr_[reidx_+1];
                    output_counts[i] = std::min(end-begin, (T)k);
                    
                }
            });
            {
                for(int i=0; i<bs; i++){
                    output_count_prefix_sum[i+1] = output_count_prefix_sum[i]+output_counts[i];
                }
            }
            std::vector<T> outputs;
            std::vector<T> outputs_ts_;
            std::vector<T> row_ptr;
            outputs.resize(output_count_prefix_sum[bs]);    
            outputs_ts_.resize(output_count_prefix_sum[bs]);
            row_ptr.resize(bs+1);
            at::parallel_for(0, bs, 1, [&](size_t start, size_t end){
                for(size_t i = start; i<end; i++){
                    T node = inputs[i].first;
                    TS ts = inputs[i].second;
                    T reidx_ = row_reindex_[node];
                    T begin = row_ptr_[reidx_];   
                    T end = row_ptr_[reidx_+1];
                    T out_begin = output_count_prefix_sum[i];
                    T out_end = out_begin + output_counts[i];
                    T sample_end = std::lower_bound(col_ts_.begin()+begin, col_ts_.begin()+end, ts) - col_ts_.begin() - 1;
                    T sample_begin = std::max(begin, sample_end - k);
                    std::copy(col_idx_.begin()+sample_begin, col_idx_.begin()+sample_end, outputs.begin()+out_begin);
                    std::copy(col_ts_.begin()+sample_begin, col_ts_.begin()+sample_end, outputs_ts_.begin()+out_begin);
                    row_ptr[i] = out_begin;
                }
            });
            return TemporalBlock<T,TS>(std::move(outputs), std::move(outputs_ts_), std::move(row_ptr));
            //return std::make_tuple(std::move(outputs), std::move(outputs_ts_), std::move(row_ptr));
         }
        std::set<T> convert_to_set(const std::vector<T> &input_chunks) {
    // 使用 std::set 的构造函数直接从 std::vector 创建 set
            return std::set<T>(input_chunks.begin(), input_chunks.end());
        }
        TemporalBlock<T, TS> sample_chunk_slice(const std::vector<T> &input_chunks, const TS &t_begin, const TS &t_end) const {
            std::set<T> unique_chunks = std::set<T>(input_chunks.begin(), input_chunks.end());
            size_t cs = input_chunks.size();
            std::vector<T> outputs;
            std::vector<T> outputs_ts; // 假设每个 chunk 最多 k 个邻居
            std::vector<T> row_ptr;
            std::vector<std::vector<T>> outputs_local_(cs);
            std::vector<std::vector<T>> outputs_ts_local_(cs);
            std::vector<std::vector<T>> row_ptr_local(cs);
            std::vector<T> output_counts(cs, 0);
            std::vector<T> output_count_prefix_sum(cs + 1, 0);
            std::vector<T> row_count_prefix_sum(cs + 1, 0);  
            at::parallel_for(0, cs, 1, [&](size_t start, size_t end) {
                T local_row_offset_ = 0;
                for (size_t i = start; i < end; i++) {
                    T chunk = input_chunks[i];
                    T begin = chunk_ptr_[chunk];
                    T end = chunk_ptr_[chunk + 1];
                    row_ptr_local[i].resize(end-begin);
                    row_count_prefix_sum[i + 1] = row_count_prefix_sum[i] + (end - begin);
                    for(T j = begin; j<end; j++){
                        row_ptr_local[i][j-begin] = local_row_offset_;

                        T node_begin = row_ptr_[j];
                        T node_end = row_ptr_[j+1];
                        for(T k = node_begin; k < node_end; k++){
                            if(unique_chunks.find(col_chunk_[k])!=unique_chunks.end()){
                                local_row_offset_++;
                                outputs_local_[i].emplace_back(col_idx_[k]);
                                outputs_ts_local_[i].emplace_back(col_ts_[k]);
                            }
                        }
                    }
                    output_counts[i] = outputs_local_[i].size();


                }
            });
            {
                for(int i=0; i<cs; i++){
                    if(i>0)row_count_prefix_sum[i]+=row_count_prefix_sum[i-1];
                    output_count_prefix_sum[i+1] = output_count_prefix_sum[i]+output_counts[i];
                }
            }
            row_ptr.resize(row_count_prefix_sum[cs]);
            outputs.resize(output_count_prefix_sum[cs]);
            at::parallel_for(0, cs, 1, [&](size_t start, size_t end) {
                T local_offset = 0;
                for (size_t i = start; i < end; i++) {
                // 将局部 row_ptr 合并到全局 row_ptr
                    for (size_t j = 0; j < outputs_local_[i].size(); j++) {
                        row_ptr[output_count_prefix_sum[i] + j] = local_offset + j;
                    }
                    local_offset += outputs_local_[i].size();

                // 将局部 outputs 和 outputs_ts 合并到全局 outputs 和 outputs_ts
                    std::copy(outputs_local_[i].begin(), outputs_local_[i].end(), outputs.begin() + output_count_prefix_sum[i]);
                    std::copy(outputs_ts_local_[i].begin(), outputs_ts_local_[i].end(), outputs_ts.begin() + output_count_prefix_sum[i]);
                }
            });

        return TemporalBlock<T, TS>(std::move(outputs), std::move(outputs_ts), std::move(row_ptr));
    }
    };
    class CPUGraph{
        public:
        using T = int64_t;
        using TS = int64_t;
        using Graph_T = Graph<T,T,CPU>;
        using Block_T = TemporalBlock<T,T>;
        Graph_T Graph_;
        CPUGraph(Graph_T g){
            Graph_ = std::move(g);
        }

        Block_T sample_neighbor(const torch::Tensor &vertices, const torch::Tensor &vertices_ts, int k, int layer){
            size_t bs = vertices.size(0);
            std::vector<T> inputs(bs);
            std::vector<TS> inputs_ts(bs);
            std::copy(vertices.data_ptr<T>(), vertices.data_ptr<T>() + bs,
                  inputs.begin());
            std::copy(vertices_ts.data_ptr<TS>(),vertices_ts.data_ptr<TS>() + bs, inputs_ts.begin());
            std::vector<std::pair<T, TS>> inputs_pair;
            inputs_pair.reserve(inputs.size()); // 预分配内存以提高性能

    // 使用 std::transform 合并
            std::transform(inputs.begin(), inputs.end(), inputs_ts.begin(), std::back_inserter(inputs_pair),
                   [](const T& a, const TS& b) {
                       return std::make_pair(a, b);
                   });
            auto res = Graph_.sample_kernel(inputs_pair,k,layer);
            return res;

        }
        Block_T sample_chunk_slice(const torch::Tensor &chunks, const T begin, const T end){
            size_t bs = chunks.size(0);
            std::vector<T> inputs(bs);
            std::copy(chunks.data_ptr<T>(), chunks.data_ptr<T>() + bs,
                  inputs.begin());
            
            auto res = Graph_.sample_chunk_slice(inputs, begin, end);
            return res;
        }
    };
    CPUGraph transform_from_edge_index(size_t n, size_t chunk_size, torch::Tensor src, torch::Tensor dst, torch::Tensor ts, torch::Tensor row_mapper){
        using T = int64_t;
        using TS = int64_t;
        using Graph_T = Graph<T,TS,CPU>;
        size_t e = src.size(1);
        using Vec_T = std::vector<std::tuple<T,T,TS>>;
        Vec_T edge_index(e);
        T *src_ptr = src.data_ptr<T>();
        T *dst_ptr = dst.data_ptr<T>();
        TS *ts_ptr = ts.data_ptr<T>();
        for(size_t i = 0; i < e; i++){
            edge_index[i] = std::make_tuple(*src_ptr,*dst_ptr,*ts_ptr);
            src_ptr++;
            dst_ptr++;
            ts_ptr++;
        }
        std::vector<T> row_mapper_vec(n);
        src_ptr= row_mapper.data_ptr<T>();
        std::copy(row_mapper.data_ptr<T>(),row_mapper.data_ptr<T>() + n, row_mapper_vec.begin());
        Graph_T g(n, chunk_size, std::move(edge_index), std::move(row_mapper_vec));
        return CPUGraph(std::move(g));
    }

}




