#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <omp.h>
#include <random>

#include <unordered_map>
#include <vector>
namespace fs = std::filesystem;
using torch::Tensor;

// =========================================================================
// [Configuration]
// =========================================================================
const std::string RAW_DIR = "/mnt/data/zlj/starrygl-data/nparts/WikiTalk_004";
const std::string OUT_DIR = "/mnt/data/zlj/starrygl-data/processed_atomic/WikiTalk_004";
const int NUM_PARTS = 4;

// Adaptive Packing
const int64_t TARGET_FILE_SIZE_MB = 200; 
const int64_t MAX_PACK_COUNT = 1000;

const int NUM_THREADS = 16;
const int64_t MAX_NODES = 200000000; 

// Sampling Strategy
enum SamplerType {
    CTDG_RECENT,   // 0
    CTDG_UNIFORM,  // 1
    DTDG_CLUSTER,  // 2
    DTDG_FULL      // 3
};

struct LayerConfig { 
    SamplerType type; 
    int fanout; 
};

// [User Config] Matches Python's LAYER_CONFIGS
std::vector<LayerConfig> LAYERS = {
    {CTDG_RECENT, 10}//, 
    //{, 10}
}; 

// =========================================================================
// Utils
// =========================================================================

std::vector<char> read_bytes(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Error opening file: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) return {};
    return buffer;
}

void write_bytes(const std::string& filename, const std::vector<char>& data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Error writing file: " + filename);
    file.write(data.data(), data.size());
}

// Compression (int64 -> int32, float64 -> float32)
Tensor compress(Tensor t) {
    if (!t.defined()) return t;
    // 压缩前确保在 CPU
    if (t.is_cuda()) t = t.cpu();
    
    if (t.scalar_type() == torch::kInt64) return t.to(torch::kInt32);
    if (t.scalar_type() == torch::kFloat64) return t.to(torch::kFloat32);
    return t;
}

// Parse IDs from filename: slot_{TID}_sub_{CID}.pt
void parse_file_ids(const std::string& filename, int& tid, int& cid) {
    tid = -1; cid = -1;
    try {
        size_t p1 = filename.find("slot_");
        if (p1 == std::string::npos) return;
        p1 += 5;
        size_t p2 = filename.find("_", p1);
        tid = std::stoi(filename.substr(p1, p2 - p1));
        
        size_t p3 = filename.find("sub_");
        if (p3 != std::string::npos) {
            p3 += 4;
            size_t p4 = filename.find(".", p3);
            cid = std::stoi(filename.substr(p3, p4 - p3));
        }
    } catch (...) {}
}

int get_rand_int(int min, int max) {
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

// [Fix] 使用 list.get(i) 修复编译错误
int64_t estimate_size(const c10::impl::GenericDict& dict) {
    int64_t total = 0;
    for(auto& item : dict) {
        if(item.value().isTensor()) {
            total += item.value().toTensor().nbytes();
        } 
        else if (item.value().isList()) { 
             auto list = item.value().toList();
             for(size_t i = 0; i < list.size(); ++i) {
                 c10::IValue elem = list.get(i); 
                 if(elem.isGenericDict()) {
                     total += estimate_size(elem.toGenericDict());
                 }
             }
        }
    }
    return total;
}

// =========================================================================
// GraphSampler Class
// =========================================================================
class GraphSampler {
public:
    Tensor indptr, sorted_src, sorted_ts, sorted_eid, sorted_cluster; 
    Tensor node_parts, edge_labels, dst_pool;
    int pid; 
    Tensor rep_indptr, rep_indices, rep_locs;

    GraphSampler(int partition_id) : pid(partition_id) {}

    void load_partition_data(const std::vector<std::string>& slot_files, 
                             const std::string& book_path, 
                             const std::string& rep_table_path,
                             const std::string& edge_label_path) {
        std::cout << "  [Load] Loading history..." << std::endl;

        try {
            auto bytes = read_bytes(book_path);
            auto book = torch::jit::pickle_load(bytes).toTuple()->elements();
            // [Fix] Force CPU
            node_parts = book[1].toTensor().to(torch::kLong).to(torch::kCPU); 
        } catch (...) {}
        try {
            auto bytes = read_bytes(rep_table_path);
            auto dict = torch::jit::pickle_load(bytes).toGenericDict();
            rep_indptr = dict.at("indptr").toTensor().to(torch::kLong).to(torch::kCPU).contiguous();
            rep_indices = dict.at("indices").toTensor().to(torch::kLong).to(torch::kCPU).contiguous();
            rep_locs = dict.at("locs").toTensor().to(torch::kLong).to(torch::kCPU).contiguous();
            std::cout << "  [Load] Replica Table loaded." << std::endl;
        } catch (...) { std::cerr << "Fail to load replica table" << std::endl; }
        try {
            if (fs::exists(edge_label_path)) {
                torch::load(edge_labels, edge_label_path);
                // [Fix] Force CPU immediately after load
                edge_labels = edge_labels.to(torch::kLong).to(torch::kCPU);
            }
        } catch (...) {}

        std::vector<Tensor> s_vec, d_vec, t_vec, e_vec, c_vec;
        
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0; i<slot_files.size(); ++i) {
            try {
                auto bytes = read_bytes(slot_files[i]);
                auto dict = torch::jit::pickle_load(bytes).toGenericDict();
                
                // [Fix] Force CPU for all loaded tensors
                auto s = dict.at("src").toTensor().to(torch::kLong).to(torch::kCPU);
                auto d = dict.at("dst").toTensor().to(torch::kLong).to(torch::kCPU);
                auto t = dict.at("ts").toTensor().to(torch::kLong).to(torch::kCPU);
                auto e = dict.at("eid").toTensor().to(torch::kLong).to(torch::kCPU);
                
                Tensor c;
                if (dict.contains("cid")) {
                    c = dict.at("cid").toTensor().to(torch::kLong).to(torch::kCPU);
                }
                
                if (!c.defined() || c.size(0) != s.size(0)) {
                    c = torch::zeros({s.size(0)}, torch::kLong).to(torch::kCPU);
                }

                #pragma omp critical
                {
                    s_vec.push_back(s); d_vec.push_back(d); t_vec.push_back(t); e_vec.push_back(e); c_vec.push_back(c);
                    s_vec.push_back(d); d_vec.push_back(s); t_vec.push_back(t); e_vec.push_back(e); c_vec.push_back(c);
                }
            } catch (...) {}
        }

        if (s_vec.empty()) return;

        auto full_src = torch::cat(s_vec);
        auto full_dst = torch::cat(d_vec);
        auto full_ts = torch::cat(t_vec);
        auto full_eid = torch::cat(e_vec);
        auto full_cluster = torch::cat(c_vec);

        if (full_src.size(0) != full_cluster.size(0)) {
            std::cerr << "Size Mismatch: " << full_src.size(0) << " vs " << full_cluster.size(0) << std::endl;
            exit(1);
        }

        dst_pool = std::get<0>(at::_unique(full_dst));

        std::cout << "  [Build] Sorting " << full_src.size(0) << " edges..." << std::endl;

        int64_t num_edges = full_src.size(0);
        auto sort_indices = torch::empty({num_edges}, torch::kLong).to(torch::kCPU);
        int64_t* sort_ptr = sort_indices.data_ptr<int64_t>();
        
        #pragma omp parallel for
        for(int64_t i=0; i<num_edges; ++i) sort_ptr[i] = i;

        auto dst_acc = full_dst.accessor<int64_t,1>();
        auto ts_acc = full_ts.accessor<int64_t,1>();

        std::sort(sort_ptr, sort_ptr + num_edges, [&](int64_t i, int64_t j) {
            if (dst_acc[i] != dst_acc[j]) return dst_acc[i] < dst_acc[j];
            return ts_acc[i] < ts_acc[j];
        });

        sorted_src = full_src.index_select(0, sort_indices);
        sorted_eid = full_eid.index_select(0, sort_indices);
        sorted_cluster = full_cluster.index_select(0, sort_indices);
        sorted_ts = full_ts.index_select(0, sort_indices);
        auto sorted_dst_tmp = full_dst.index_select(0, sort_indices);

        indptr = torch::zeros({MAX_NODES + 1}, torch::kLong).to(torch::kCPU);
        auto indptr_ptr = indptr.data_ptr<int64_t>();
        auto dst_ptr_raw = sorted_dst_tmp.data_ptr<int64_t>();

        for(int64_t i=0; i<num_edges; ++i) {
            if (dst_ptr_raw[i] < MAX_NODES) indptr_ptr[dst_ptr_raw[i] + 1]++;
        }
        for(int64_t i=0; i<MAX_NODES; ++i) indptr_ptr[i+1] += indptr_ptr[i];
    }
    void load_aux_tables(const std::string& book_path, 
                         const std::string& rep_table_path,
                         const std::string& edge_label_path) {
        std::cout << "  [Load] Loading aux tables..." << std::endl;
        try {
            auto bytes = read_bytes(book_path);
            auto book = torch::jit::pickle_load(bytes).toTuple()->elements();
            node_parts = book[1].toTensor().to(torch::kLong).to(torch::kCPU); 
        } catch (...) { std::cerr << "Warning: Failed to load partition book." << std::endl; }

        try {
            auto bytes = read_bytes(rep_table_path);
            auto dict = torch::jit::pickle_load(bytes).toGenericDict();
            rep_indptr = dict.at("indptr").toTensor().to(torch::kLong).to(torch::kCPU).contiguous();
            rep_indices = dict.at("indices").toTensor().to(torch::kLong).to(torch::kCPU).contiguous();
            rep_locs = dict.at("locs").toTensor().to(torch::kLong).to(torch::kCPU).contiguous();
            std::cout << "  [Load] Replica Table loaded." << std::endl;
        } catch (...) { std::cerr << "Fail to load replica table" << std::endl; }

        try {
            if (fs::exists(edge_label_path)) {
                torch::load(edge_labels, edge_label_path);
                edge_labels = edge_labels.to(torch::kLong).to(torch::kCPU);
            }
        } catch (...) {}
    }
    // c10::impl::GenericList compute_route(Tensor gids, Tensor ts) {
    //     c10::impl::GenericList plans(c10::AnyType::get());
    //     if (!node_parts.defined() || gids.numel() == 0) return plans;
        
    //     if (gids.max().item<int64_t>() >= node_parts.size(0)) {
    //          plans.push_back(c10::IValue()); return plans;
    //     }

    //     // [Fix] Ensure input is on CPU
    //     if (gids.is_cuda()) gids = gids.cpu();

    //     auto owners = node_parts.index_select(0, gids);
    //     auto mask = (owners != pid);
        
    //     if (mask.any().item<bool>()) {
    //         auto raw_send_ranks = owners.masked_select(mask); // 目标 Rank
    //         auto raw_send_indices = torch::arange(gids.size(0), torch::kLong).to(torch::kCPU).masked_select(mask);
    //         auto raw_send_remote = gids.masked_select(mask); // 全局 Node ID

    //         auto sort_idx = torch::argsort(raw_send_ranks);
            
    //         auto sorted_ranks = raw_send_ranks.index_select(0, sort_idx);   
    //         auto sorted_indices = raw_send_indices.index_select(0, sort_idx);
    //         auto sorted_remote = raw_send_remote.index_select(0, sort_idx);
    //         auto send_sizes = torch::bincount(sorted_ranks, {}, NUM_PARTS);

    //         // C. 打包
    //         c10::impl::GenericDict plan(c10::StringType::get(), c10::AnyType::get());
    //         plan.insert("send_ranks", compress(sorted_ranks));
    //         plan.insert("send_indices", compress(sorted_indices));
    //         plan.insert("send_remote_indices", compress(sorted_remote));
    //         plan.insert("send_sizes", compress(send_sizes)); // [新增]
            
    //         plans.push_back(plan);
    //     } else {
    //         plans.push_back(c10::IValue()); 
    //     }
    //     return plans;
    // }
    void init_graph_from_tensors(Tensor full_src, Tensor full_dst, Tensor full_ts, Tensor full_eid, Tensor full_cluster) {
        
        // 确保输入都在 CPU
        full_src = full_src.to(torch::kCPU);
        full_dst = full_dst.to(torch::kCPU);
        full_ts = full_ts.to(torch::kCPU);
        full_eid = full_eid.to(torch::kCPU);
        full_cluster = full_cluster.to(torch::kCPU);

        dst_pool = std::get<0>(at::_unique(full_dst));

        std::cout << "  [Build] Sorting " << full_src.size(0) << " edges for CSR..." << std::endl;

        int64_t num_edges = full_src.size(0);
        auto sort_indices = torch::empty({num_edges}, torch::kLong).to(torch::kCPU);
        int64_t* sort_ptr = sort_indices.data_ptr<int64_t>();
        
        #pragma omp parallel for
        for(int64_t i=0; i<num_edges; ++i) sort_ptr[i] = i;

        auto dst_acc = full_dst.accessor<int64_t,1>();
        auto ts_acc = full_ts.accessor<int64_t,1>();

        // 这里的排序是为了建立 CSR (按 dst, ts 排序)
        // 注意：data.pt 原本是按 (slot, large_id, ts) 排序的，这里为了采样必须重排
        std::sort(sort_ptr, sort_ptr + num_edges, [&](int64_t i, int64_t j) {
            if (dst_acc[i] != dst_acc[j]) return dst_acc[i] < dst_acc[j];
            return ts_acc[i] < ts_acc[j];
        });

        sorted_src = full_src.index_select(0, sort_indices);
        sorted_eid = full_eid.index_select(0, sort_indices);
        sorted_cluster = full_cluster.index_select(0, sort_indices);
        sorted_ts = full_ts.index_select(0, sort_indices);
        auto sorted_dst_tmp = full_dst.index_select(0, sort_indices);

        // 构建 CSR indptr
        indptr = torch::zeros({MAX_NODES + 1}, torch::kLong).to(torch::kCPU);
        auto indptr_ptr = indptr.data_ptr<int64_t>();
        auto dst_ptr_raw = sorted_dst_tmp.data_ptr<int64_t>();

        for(int64_t i=0; i<num_edges; ++i) {
            if (dst_ptr_raw[i] < MAX_NODES) indptr_ptr[dst_ptr_raw[i] + 1]++;
        }
        for(int64_t i=0; i<MAX_NODES; ++i) indptr_ptr[i+1] += indptr_ptr[i];
        
        std::cout << "  [Build] Graph CSR built." << std::endl;
    }
    c10::impl::GenericList compute_route(Tensor gids, Tensor ts) {
        c10::impl::GenericList plans(c10::AnyType::get());
        
        if (!node_parts.defined() || !rep_indptr.defined() || gids.numel() == 0) {
            plans.push_back(c10::IValue()); // 对应 Python 的 None
            return plans;
        }

        if (gids.is_cuda()) gids = gids.cpu();
        if (ts.is_cuda()) ts = ts.cpu();

        // --- Step 1: 基于时间戳的 Master 节点去重 (对应 Python _jit_compute_route_cpu) ---
        // 目的：如果一个物理节点在 batch 中出现多次，只发时间戳最大的那个
        auto gids_ptr = gids.data_ptr<int64_t>();
        auto ts_ptr = ts.data_ptr<int64_t>();
        auto parts_ptr = node_parts.data_ptr<int64_t>();
        
        std::unordered_map<int64_t, std::pair<int64_t, int64_t>> master_latest; // gid -> {max_ts, batch_idx}
        
        for (int64_t i = 0; i < gids.size(0); ++i) {
            int64_t gid = gids_ptr[i];
            int64_t t = ts_ptr[i];
            
            // 只有当前分区是该节点的 Master (Owner) 时，才由我负责同步
            if (gid < node_parts.size(0) && parts_ptr[gid] == pid) {
                if (master_latest.find(gid) == master_latest.end() || t > master_latest[gid].first) {
                    master_latest[gid] = {t, i};
                }
            }
        }

        if (master_latest.empty()) {
            plans.push_back(c10::IValue());
            return plans;
        }

        // --- Step 2: 查表并生成多副本路由 (对应 Python CSRReplicaTable.lookup) ---
        std::vector<int64_t> s_ranks, s_indices, s_remote;
        auto r_indptr_ptr = rep_indptr.data_ptr<int64_t>();
        auto r_indices_ptr = rep_indices.data_ptr<int64_t>();
        auto r_locs_ptr = rep_locs.data_ptr<int64_t>();

        for (auto const& [gid, info] : master_latest) {
            int64_t batch_idx = info.second;
            int64_t start = r_indptr_ptr[gid];
            int64_t end = r_indptr_ptr[gid + 1];

            for (int64_t j = start; j < end; ++j) {
                int64_t target_rank = r_indices_ptr[j];
                if (target_rank != pid) { // 排除自己
                    s_ranks.push_back(target_rank);
                    s_indices.push_back(batch_idx);
                    s_remote.push_back(r_locs_ptr[j]); // 关键：发的是目标分区的 Local Offset
                }
            }
        }

        if (s_ranks.empty()) {
            plans.push_back(c10::IValue());
            return plans;
        }

        // --- Step 3: 排序、计算维度并封装 (严格对齐 Python) ---
        auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU);
        auto t_ranks = torch::from_blob(s_ranks.data(), {(int64_t)s_ranks.size()}, opts).clone();
        auto t_indices = torch::from_blob(s_indices.data(), {(int64_t)s_indices.size()}, opts).clone();
        auto t_remote = torch::from_blob(s_remote.data(), {(int64_t)s_remote.size()}, opts).clone();

        auto sort_idx = torch::argsort(t_ranks);
        auto sorted_ranks = t_ranks.index_select(0, sort_idx);

        c10::impl::GenericDict plan(c10::StringType::get(), c10::AnyType::get());
        plan.insert("send_ranks", compress(sorted_ranks));
        plan.insert("send_indices", compress(t_indices.index_select(0, sort_idx)));
        plan.insert("send_remote_indices", compress(t_remote.index_select(0, sort_idx)));
        

        plan.insert("send_sizes", compress(torch::bincount(sorted_ranks, {}, NUM_PARTS)));

        plans.push_back(plan); 
        return plans;

    }
    std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> sample_one_hop(Tensor nodes, Tensor times, const LayerConfig& config, int filter_cluster_id) {
        // [Fix] Ensure inputs are CPU
        if (nodes.is_cuda()) nodes = nodes.cpu();
        if (times.is_cuda()) times = times.cpu();

        int64_t num_targets = nodes.size(0);
        
        auto nodes_acc = nodes.accessor<int64_t,1>();
        auto times_acc = times.accessor<int64_t,1>();
        auto start_acc = indptr.accessor<int64_t,1>();
        
        auto g_src_ptr = sorted_src.data_ptr<int64_t>();
        auto g_ts_ptr = sorted_ts.data_ptr<int64_t>();
        auto g_eid_ptr = sorted_eid.data_ptr<int64_t>();
        auto g_cluster_ptr = sorted_cluster.data_ptr<int64_t>(); 

        std::vector<Tensor> s_vec, d_vec, t_vec, e_vec, dt_vec;

        #pragma omp parallel 
        {
            std::vector<int64_t> s_loc, d_loc, t_loc, e_loc;
            std::vector<float> dt_loc;
            
            #pragma omp for schedule(dynamic, 64)
            for(int64_t i=0; i<num_targets; ++i) {
                int64_t u = nodes_acc[i];
                int64_t t = times_acc[i];
                if (u >= MAX_NODES) continue;
                
                int64_t start = start_acc[u];
                int64_t end = start_acc[u+1];
                if (start >= end) continue;

                int64_t valid_start = -1, valid_end = -1;
                auto it_begin = g_ts_ptr + start;
                auto it_end = g_ts_ptr + end;

                if (config.type == CTDG_RECENT || config.type == CTDG_UNIFORM) {
                    auto it_cut = std::lower_bound(it_begin, it_end, t); 
                    valid_start = start;
                    valid_end = std::distance(g_ts_ptr, it_cut);
                } 
                else if (config.type == DTDG_FULL) {
                    auto it_lb = std::lower_bound(it_begin, it_end, t);
                    auto it_ub = std::upper_bound(it_begin, it_end, t);
                    valid_start = std::distance(g_ts_ptr, it_lb);
                    valid_end = std::distance(g_ts_ptr, it_ub);
                } 
                else if (config.type == DTDG_CLUSTER) {
                    auto it_lb = std::lower_bound(it_begin, it_end, t);
                    auto it_ub = std::upper_bound(it_begin, it_end, t);
                    valid_start = std::distance(g_ts_ptr, it_lb);
                    valid_end = std::distance(g_ts_ptr, it_ub);
                }

                int64_t valid_count = valid_end - valid_start;
                if (valid_count <= 0) continue;

                if (config.type == DTDG_CLUSTER) {
                    for(int64_t k=0; k<valid_count; ++k) {
                        int64_t idx = valid_start + k;
                        if (g_cluster_ptr[idx] == filter_cluster_id) {
                            s_loc.push_back(g_src_ptr[idx]);
                            d_loc.push_back(i);
                            t_loc.push_back(g_ts_ptr[idx]);
                            e_loc.push_back(g_eid_ptr[idx]);
                            dt_loc.push_back((float)(t - g_ts_ptr[idx]));
                        }
                    }
                } 
                else if (config.type == CTDG_RECENT || config.type == DTDG_FULL) {
                    int64_t count = (config.type == DTDG_FULL) ? valid_count : std::min((int64_t)config.fanout, valid_count);
                    int64_t read_start = valid_end - count;
                    for(int64_t k=0; k<count; ++k) {
                        int64_t idx = read_start + k;
                        s_loc.push_back(g_src_ptr[idx]);
                        d_loc.push_back(i);
                        t_loc.push_back(g_ts_ptr[idx]);
                        e_loc.push_back(g_eid_ptr[idx]);
                        dt_loc.push_back((float)(t - g_ts_ptr[idx]));
                    }
                }
                else if (config.type == CTDG_UNIFORM) {
                    if (valid_count <= config.fanout) {
                        for(int64_t k=0; k<valid_count; ++k) {
                            int64_t idx = valid_start + k;
                            s_loc.push_back(g_src_ptr[idx]);
                            d_loc.push_back(i);
                            t_loc.push_back(g_ts_ptr[idx]);
                            e_loc.push_back(g_eid_ptr[idx]);
                            dt_loc.push_back((float)(t - g_ts_ptr[idx]));
                        }
                    } else {
                        std::vector<int64_t> selected;
                        selected.reserve(config.fanout);
                        for(int k=0; k<config.fanout; ++k) {
                            bool unique = false;
                            int64_t rnd;
                            while(!unique) {
                                rnd = get_rand_int(0, valid_count - 1);
                                unique = true;
                                for(auto v : selected) if(v==rnd) unique=false;
                            }
                            selected.push_back(rnd);
                            int64_t idx = valid_start + rnd;
                            s_loc.push_back(g_src_ptr[idx]);
                            d_loc.push_back(i);
                            t_loc.push_back(g_ts_ptr[idx]);
                            e_loc.push_back(g_eid_ptr[idx]);
                            dt_loc.push_back((float)(t - g_ts_ptr[idx]));
                        }
                    }
                }
            }
            
            #pragma omp critical
            {
                if (!s_loc.empty()) {
                    auto opts_l = torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU); // Force CPU
                    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
                    s_vec.push_back(torch::from_blob(s_loc.data(), {(int64_t)s_loc.size()}, opts_l).clone());
                    d_vec.push_back(torch::from_blob(d_loc.data(), {(int64_t)d_loc.size()}, opts_l).clone());
                    t_vec.push_back(torch::from_blob(t_loc.data(), {(int64_t)t_loc.size()}, opts_l).clone());
                    e_vec.push_back(torch::from_blob(e_loc.data(), {(int64_t)e_loc.size()}, opts_l).clone());
                    dt_vec.push_back(torch::from_blob(dt_loc.data(), {(int64_t)dt_loc.size()}, opts_f).clone());
                }
            }
        }

        if (s_vec.empty()) return {Tensor(), Tensor(), Tensor(), Tensor(), Tensor()};
        return {torch::cat(s_vec), torch::cat(d_vec), torch::cat(t_vec), torch::cat(dt_vec), torch::cat(e_vec)};
    }

    struct NodeTimeKey {
        int64_t id;
        int64_t ts;
        bool operator==(const NodeTimeKey& other) const {
            return id == other.id && ts == other.ts;
        }
    };

    struct NodeTimeHasher {
        std::size_t operator()(const NodeTimeKey& k) const {
            std::size_t h1 = std::hash<int64_t>{}(k.id);
            std::size_t h2 = std::hash<int64_t>{}(k.ts);
            // 使用黄金分割比组合哈希值
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
        }
    };

    // =========================================================================
    // [Main Function] GraphSampler::build_batch_data
    // =========================================================================
    c10::impl::GenericList build_batch_data(std::string type, std::vector<Tensor> task_data, int num_neg, int tid, int cid) {
        c10::impl::GenericList batch_list(c10::AnyType::get());
        Tensor l0_nodes, l0_ts;
        Tensor task_src, task_dst, task_ts, task_eid, task_label;
        c10::impl::GenericDict task_dict(c10::StringType::get(), c10::AnyType::get());

        // ---------------------------------------------------------------------
        // 1. 构建 Task 层 (Layer 0)
        // ---------------------------------------------------------------------
        if (type == "link") {
            task_src = task_data[0]; task_dst = task_data[1]; task_ts = task_data[2]; 
            task_label = task_data[3]; task_eid = task_data[4];
            l0_nodes = torch::cat({task_src, task_dst});
            l0_ts = torch::cat({task_ts, task_ts});
            task_dict.insert("task_src", compress(task_src));
            task_dict.insert("task_dst", compress(task_dst));
            task_dict.insert("task_ts", compress(task_ts));
            task_dict.insert("task_label", compress(task_label));
            task_dict.insert("task_eid", compress(task_eid));
        } else if (type == "neg") {
            Tensor seed_ts = task_data[0];
            int64_t B = seed_ts.size(0);
            auto rand_idx = torch::randint(0, dst_pool.numel(), {B * num_neg}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
            l0_nodes = dst_pool.index_select(0, rand_idx);
            l0_ts = seed_ts.repeat_interleave(num_neg);
            task_dict.insert("task_neg_dst", compress(l0_nodes));
            task_dict.insert("task_ts", compress(seed_ts));
        }

        int64_t batch_size = (type == "link") ? task_src.size(0) : l0_nodes.size(0);
        Tensor slot_tensor = torch::full({batch_size}, cid, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
        task_dict.insert("task_slot", slot_tensor); 

        auto unique_res = at::_unique(l0_nodes, true, true);
        Tensor curr_gids = std::get<0>(unique_res);
        Tensor inv_map = std::get<1>(unique_res);
        Tensor curr_ts = torch::zeros_like(curr_gids); 
        curr_ts.index_put_({inv_map}, l0_ts);

        task_dict.insert("gids", compress(curr_gids));
        task_dict.insert("ts", compress(curr_ts));
        task_dict.insert("inv_map", compress(inv_map));
        batch_list.push_back(task_dict);

        c10::impl::GenericList routes(c10::AnyType::get());
        routes.push_back(compute_route(curr_gids, curr_ts));

        // ---------------------------------------------------------------------
        // 2. 采样状态初始化
        // ---------------------------------------------------------------------
        Tensor prev_layer_neighbors_mask = torch::ones({curr_gids.size(0)}, torch::kBool).to(torch::kCPU);
        Tensor dst_batch_indices = inv_map; 

        for(const auto& layer : LAYERS) {
            // [Step 1] 计算过滤掩码
            Tensor mask = (node_parts.index_select(0, curr_gids) == pid);
            if (prev_layer_neighbors_mask.defined()) {
                mask = mask & prev_layer_neighbors_mask;
            }
            Tensor valid_indices = torch::nonzero(mask).squeeze(1);

            // [Early Stop 1] 采样源为空，保证 indptr 长度对齐
            if (valid_indices.numel() == 0) {
                c10::impl::GenericDict empty_d(c10::StringType::get(), c10::AnyType::get());
                empty_d.insert("indptr", torch::zeros({curr_gids.size(0) + 1}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("indices", torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("eid",  torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("edge_dt",  torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("dst_batch_indices", compress(dst_batch_indices));
                empty_d.insert("gids", compress(curr_gids));
                empty_d.insert("ts", compress(curr_ts));
                batch_list.push_back(empty_d);
                routes.push_back(c10::IValue());
                break;
            }

            // [Step 2] 采样
            Tensor nodes_to_sample = curr_gids.index_select(0, valid_indices);
            Tensor ts_to_sample = curr_ts.index_select(0, valid_indices);
            auto [src, raw_dst_idx, res_ts, dt, eid] = sample_one_hop(nodes_to_sample, ts_to_sample, layer, cid);
            
            // [Early Stop 2] 采样邻居为空
            if (!src.defined() || src.numel() == 0) {
                c10::impl::GenericDict empty_d(c10::StringType::get(), c10::AnyType::get());
                empty_d.insert("indptr", torch::zeros({curr_gids.size(0) + 1}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("indices", torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("eid",  torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("edge_dt",  torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("dst_batch_indices", compress(dst_batch_indices));
                empty_d.insert("gids", compress(curr_gids)); 
                empty_d.insert("ts", compress(curr_ts));
                batch_list.push_back(empty_d);
                routes.push_back(c10::IValue());
                break;
            }

            Tensor dst_idx = valid_indices.index_select(0, raw_dst_idx);

            // -----------------------------------------------------------------
            // [Step 3] 联合去重 (ID + TS)
            // -----------------------------------------------------------------
            std::vector<int64_t> next_gids_vec;
            std::vector<int64_t> next_ts_vec;
            std::vector<int64_t> src_indices_vec;
            
            int64_t est_size = curr_gids.size(0) + src.size(0);
            next_gids_vec.reserve(est_size);
            next_ts_vec.reserve(est_size);
            src_indices_vec.reserve(src.size(0));
            
            std::unordered_map<NodeTimeKey, int64_t, NodeTimeHasher> pair_to_idx;
            pair_to_idx.reserve(est_size);

            // 3.1 固化当前层节点 (作为下一层采样目标的基准)
            auto curr_ptr = curr_gids.data_ptr<int64_t>();
            auto curr_ts_ptr = curr_ts.data_ptr<int64_t>();
            for(int64_t i = 0; i < curr_gids.size(0); ++i) {
                NodeTimeKey key = {curr_ptr[i], curr_ts_ptr[i]};
                if (pair_to_idx.find(key) == pair_to_idx.end()) {
                    pair_to_idx[key] = next_gids_vec.size();
                    next_gids_vec.push_back(key.id);
                    next_ts_vec.push_back(key.ts);
                }
            }

            // 3.2 映射采样邻居
            auto src_ptr = src.data_ptr<int64_t>();
            auto res_ts_ptr = res_ts.data_ptr<int64_t>();
            for(int64_t i = 0; i < src.size(0); ++i) {
                NodeTimeKey key = {src_ptr[i], res_ts_ptr[i]};
                auto it = pair_to_idx.find(key);
                if (it != pair_to_idx.end()) {
                    src_indices_vec.push_back(it->second);
                } else {
                    int64_t new_idx = next_gids_vec.size();
                    pair_to_idx[key] = new_idx;
                    next_gids_vec.push_back(key.id);
                    next_ts_vec.push_back(key.ts);
                    src_indices_vec.push_back(new_idx);
                }
            }

            Tensor next_gids = torch::from_blob(next_gids_vec.data(), {(int64_t)next_gids_vec.size()}, torch::kLong).to(torch::kCPU).clone();
            Tensor next_ts = torch::from_blob(next_ts_vec.data(), {(int64_t)next_ts_vec.size()}, torch::kLong).to(torch::kCPU).clone();
            Tensor src_new_indices = torch::from_blob(src_indices_vec.data(), {(int64_t)src_indices_vec.size()}, torch::kLong).to(torch::kCPU).clone();
            
            // -----------------------------------------------------------------
            // [Step 4] 构建 CSR
            // -----------------------------------------------------------------
            auto sort_perm = torch::argsort(dst_idx);
            Tensor sorted_indices = src_new_indices.index_select(0, sort_perm);
            Tensor sorted_eid_layer = eid.index_select(0, sort_perm);
            Tensor sorted_dt = dt.index_select(0, sort_perm);
            
            Tensor layer_indptr = torch::zeros({curr_gids.size(0) + 1}, torch::kLong).to(torch::kCPU);
            auto count_ptr = layer_indptr.data_ptr<int64_t>();
            auto dst_ptr_raw = dst_idx.data_ptr<int64_t>();
            for(int64_t i=0; i<dst_idx.numel(); ++i) count_ptr[dst_ptr_raw[i] + 1]++;
            for(int64_t i=0; i<curr_gids.size(0); ++i) count_ptr[i+1] += count_ptr[i];

            c10::impl::GenericDict layer_d(c10::StringType::get(), c10::AnyType::get());
            layer_d.insert("indptr", compress(layer_indptr));
            layer_d.insert("indices", compress(sorted_indices));
            layer_d.insert("eid", compress(sorted_eid_layer));
            layer_d.insert("edge_dt", compress(sorted_dt));
            layer_d.insert("gids", compress(next_gids)); 
            layer_d.insert("ts", compress(next_ts));
            layer_d.insert("dst_batch_indices", compress(dst_batch_indices));
            batch_list.push_back(layer_d);
            
            // -----------------------------------------------------------------
            // [Step 5] 更新循环状态
            // -----------------------------------------------------------------
            dst_batch_indices = src_new_indices; 
            prev_layer_neighbors_mask = torch::zeros({next_gids.size(0)}, torch::kBool).to(torch::kCPU);
            prev_layer_neighbors_mask.index_fill_(0, src_new_indices, true);

            curr_gids = next_gids;
            curr_ts = next_ts; 
            
            routes.push_back(compute_route(curr_gids,curr_ts));
        }
        
        auto first_elem = batch_list.get(0).toGenericDict();
        first_elem.insert("comm_plan", routes);
        return batch_list;
    }
};
// int main() {
//     torch::set_num_threads(1); 
//     omp_set_num_threads(NUM_THREADS);

//     std::cout << "=== StarryGL C++ Preprocessor (CPU Forced) ===" << std::endl;

//     for (int pid = 0; pid < NUM_PARTS; ++pid) {
//         std::string p_dir = RAW_DIR + "/part_" + std::to_string(pid);
//         if (!fs::exists(p_dir)) continue;
        
//         std::string save_dir = OUT_DIR + "/part_" + std::to_string(pid);
//         fs::create_directories(save_dir);

//         std::ofstream meta_file(save_dir + "/meta.txt");

//         GraphSampler sampler(pid);
        
//         std::vector<std::string> files;
//         for (const auto& entry : fs::directory_iterator(p_dir)) {
//             std::string name = entry.path().filename().string();
//             if (name.find("slot_") == 0 && entry.path().extension() == ".pt") {
//                 files.push_back(entry.path().string());
//             }
//         }
//         std::sort(files.begin(), files.end());

//         sampler.load_partition_data(
//             files, 
//             RAW_DIR + "/partition_book.pt", 
//             RAW_DIR + "/replica_table.pt", // 确保此路径存在
//             RAW_DIR + "/edge_label.pt"
//         );

//         int batch_idx = 0;
//         int64_t current_bytes = 0;
//         c10::impl::GenericList mega_batch(c10::AnyType::get());
        
//         struct MetaInfo { std::string fname; int start; int count; int tid; int cid; };
//         std::vector<MetaInfo> meta_buffer;

//         std::cout << "  [Sample] Processing " << files.size() << " slots..." << std::endl;

//         #pragma omp parallel for schedule(dynamic)
//         for(size_t i=0; i<files.size(); ++i) {
//             try {
//                 int tid, cid;
//                 parse_file_ids(fs::path(files[i]).filename().string(), tid, cid);

//                 auto bytes = read_bytes(files[i]);
//                 auto dict = torch::jit::pickle_load(bytes).toGenericDict();
                
//                 // [Fix] Force CPU
//                 auto src = dict.at("src").toTensor().to(torch::kLong).to(torch::kCPU);
//                 auto dst = dict.at("dst").toTensor().to(torch::kLong).to(torch::kCPU);
//                 auto ts = dict.at("ts").toTensor().to(torch::kLong).to(torch::kCPU);
//                 auto eid = dict.at("eid").toTensor().to(torch::kLong).to(torch::kCPU);
                
//                 Tensor lbl = torch::zeros_like(src);
//                 if (sampler.edge_labels.defined()) {
//                     // [Fix] sampler.edge_labels is forced CPU in load_partition_data
//                     // eid is forced CPU above
//                     lbl = sampler.edge_labels.index_select(0, eid);
//                 }

//                 auto res_link = sampler.build_batch_data("link", {src, dst, ts, lbl, eid}, 0, tid, cid);
//                 int64_t size_link = estimate_size(res_link.get(0).toGenericDict());

//                 int num_set = 8;
//                 std::vector<c10::impl::GenericList> res_negs;
//                 int64_t size_negs = 0;
//                 for(int k=0; k<num_set; ++k) {
//                     auto res = sampler.build_batch_data("neg", {ts}, 1, tid, cid);
//                     if (!res.empty()) {
//                         res_negs.push_back(res);
//                         size_negs += estimate_size(res.get(0).toGenericDict());
//                     }
//                 }

//                 #pragma omp critical
//                 {
//                     int start_index = mega_batch.size();
//                     mega_batch.push_back(res_link);
//                     for(auto& r : res_negs) mega_batch.push_back(r);
//                     int total_count = 1 + res_negs.size();
                    
//                     current_bytes += (size_link + size_negs);
//                     meta_buffer.push_back({"", start_index, total_count, tid, cid});

//                     if (current_bytes >= TARGET_FILE_SIZE_MB * 1024 * 1024 || mega_batch.size() >= MAX_PACK_COUNT) {
//                         std::string fname = "mega_batch_" + std::to_string(batch_idx++) + ".pt";
//                         std::string out_name = save_dir + "/" + fname;
                        
//                         auto out_bytes = torch::jit::pickle_save(mega_batch);
//                         write_bytes(out_name, out_bytes);
                        
//                         for(auto& m : meta_buffer) {
//                             meta_file << fname << " " << m.start << " " << m.count << " " << m.tid << " " << m.cid << "\n";
//                         }

//                         mega_batch = c10::impl::GenericList(c10::AnyType::get());
//                         meta_buffer.clear();
//                         current_bytes = 0;
//                         if (batch_idx % 10 == 0) std::cout << "." << std::flush;
//                     }
//                 }
//             } catch (const std::exception& e) {
//                 std::cerr << "\nErr processing " << files[i] << ": " << e.what() << std::endl;
//             }
//         }
        
//         if (!mega_batch.empty()) {
//             std::string fname = "mega_batch_" + std::to_string(batch_idx++) + ".pt";
//             std::string out_name = save_dir + "/" + fname;
//             auto out_bytes = torch::jit::pickle_save(mega_batch);
//             write_bytes(out_name, out_bytes);
//             for(auto& m : meta_buffer) {
//                 meta_file << fname << " " << m.start << " " << m.count << " " << m.tid << " " << m.cid << "\n";
//             }
//         }
        
//         meta_file.close();
//         std::cout << "\n  Done Partition " << pid << std::endl;
//     }
//     return 0;
// }

struct TaskSpec {
    int tid;
    int cid;
    int64_t start;
    int64_t count;
};

int main() {
    torch::set_num_threads(1); 
    omp_set_num_threads(NUM_THREADS);

    std::cout << "=== StarryGL C++ Preprocessor (CPU Forced, Monolithic Input) ===" << std::endl;

    for (int pid = 0; pid < NUM_PARTS; ++pid) {
        std::string p_dir = RAW_DIR + "/part_" + std::to_string(pid);
        std::string data_path = p_dir + "/data.pt";

        if (!fs::exists(data_path)) {
            std::cout << "Skip " << pid << " (no data.pt)" << std::endl;
            continue;
        }
        
        std::string save_dir = OUT_DIR + "/part_" + std::to_string(pid);
        fs::create_directories(save_dir);
        std::ofstream meta_file(save_dir + "/meta.txt");

        std::cout << "  [Load] Loading " << data_path << "..." << std::endl;

        // 1. 加载单个大文件 data.pt
        auto bytes = read_bytes(data_path);
        auto dict = torch::jit::pickle_load(bytes).toGenericDict();
        
        // [FIX 1] 显式转换为 kLong (int64)。这是解决 Runtime Error 的关键！
        // 即使 Python 端存的是 Int32，这里也会强制转为 Int64 以匹配 C++ 的 accessor<int64_t>
        auto full_src = dict.at("src").toTensor().to(torch::kLong);
        auto full_dst = dict.at("dst").toTensor().to(torch::kLong);
        auto full_ts = dict.at("ts").toTensor().to(torch::kLong);
        auto full_eid = dict.at("eid").toTensor().to(torch::kLong);
        
        Tensor full_cluster;
        if (dict.contains("cid")) {
            full_cluster = dict.at("cid").toTensor().to(torch::kLong);
        } else {
            // Fallback if cid is missing
            full_cluster = torch::zeros_like(full_src);
        }

        auto full_index = dict.at("index").toGenericDict(); // {tid: {cid: (start, count)}}

        // 2. 初始化 Sampler (一次性传入所有数据)
        GraphSampler sampler(pid);
        sampler.load_aux_tables(
            RAW_DIR + "/partition_book.pt", 
            RAW_DIR + "/replica_table.pt", 
            RAW_DIR + "/edge_label.pt"
        );
        sampler.init_graph_from_tensors(full_src, full_dst, full_ts, full_eid, full_cluster);

        // 3. 展平任务列表 (Flatten tasks for parallelism)
        std::vector<TaskSpec> tasks;
        
        // [FIX 2] 使用 const auto& 引用遍历，解决 DictEntryRef 编译错误
        for (const auto& item : full_index) {
            int tid = item.key().toInt();
            auto cid_map = item.value().toGenericDict();
            
            for (const auto& sub_item : cid_map) {
                int cid = sub_item.key().toInt();
                auto tuple = sub_item.value().toTuple()->elements();
                int64_t start = tuple[0].toInt();
                int64_t count = tuple[1].toInt();
                if (count > 0) {
                    tasks.push_back({tid, cid, start, count});
                }
            }
        }
        
        // 按 TID 排序任务，保持处理顺序
        std::sort(tasks.begin(), tasks.end(), [](const TaskSpec& a, const TaskSpec& b){
            if (a.tid != b.tid) return a.tid < b.tid;
            return a.cid < b.cid;
        });

        int batch_idx = 0;
        int64_t current_bytes = 0;
        c10::impl::GenericList mega_batch(c10::AnyType::get());
        struct MetaInfo { std::string fname; int start; int count; int tid; int cid; };
        std::vector<MetaInfo> meta_buffer;

        std::cout << "  [Sample] Processing " << tasks.size() << " task blocks..." << std::endl;

        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0; i<tasks.size(); ++i) {
            try {
                auto& task = tasks[i];
                int tid = task.tid;
                int cid = task.cid;
                
                // 4. 切片 (Slicing) 获取当前任务数据
                // 因为 full_src 已经是 Long 类型，这里切片出来的 src 也是 Long
                auto src = full_src.slice(0, task.start, task.start + task.count);
                auto dst = full_dst.slice(0, task.start, task.start + task.count);
                auto ts = full_ts.slice(0, task.start, task.start + task.count);
                auto eid = full_eid.slice(0, task.start, task.start + task.count);

                Tensor lbl = torch::zeros_like(src);
                if (sampler.edge_labels.defined()) {
                    lbl = sampler.edge_labels.index_select(0, eid);
                }

                auto res_link = sampler.build_batch_data("link", {src, dst, ts, lbl, eid}, 0, tid, cid);
                int64_t size_link = estimate_size(res_link.get(0).toGenericDict());

                int num_set = 8;
                std::vector<c10::impl::GenericList> res_negs;
                int64_t size_negs = 0;
                for(int k=0; k<num_set; ++k) {
                    auto res = sampler.build_batch_data("neg", {ts}, 1, tid, cid);
                    if (!res.empty()) {
                        res_negs.push_back(res);
                        size_negs += estimate_size(res.get(0).toGenericDict());
                    }
                }

                #pragma omp critical
                {
                    int start_index = mega_batch.size();
                    mega_batch.push_back(res_link);
                    for(auto& r : res_negs) mega_batch.push_back(r);
                    int total_count = 1 + res_negs.size();
                    
                    current_bytes += (size_link + size_negs);
                    meta_buffer.push_back({"", start_index, total_count, tid, cid});

                    if (current_bytes >= TARGET_FILE_SIZE_MB * 1024 * 1024 || mega_batch.size() >= MAX_PACK_COUNT) {
                        std::string fname = "mega_batch_" + std::to_string(batch_idx++) + ".pt";
                        std::string out_name = save_dir + "/" + fname;
                        
                        auto out_bytes = torch::jit::pickle_save(mega_batch);
                        write_bytes(out_name, out_bytes);
                        
                        for(auto& m : meta_buffer) {
                            // 修正 meta 格式，确保文件名正确写入
                            m.fname = fname; 
                            meta_file << m.fname << " " << m.start << " " << m.count << " " << m.tid << " " << m.cid << "\n";
                        }

                        mega_batch = c10::impl::GenericList(c10::AnyType::get());
                        meta_buffer.clear();
                        current_bytes = 0;
                        if (batch_idx % 10 == 0) std::cout << "." << std::flush;
                    }
                }
            } catch (const std::exception& e) {
                // 打印更详细的错误信息，帮助定位
                std::cerr << "\nErr processing TID " << tasks[i].tid << ": " << e.what() << std::endl;
            }
        }
        
        // 处理剩余的 batch
        if (!mega_batch.empty()) {
            std::string fname = "mega_batch_" + std::to_string(batch_idx++) + ".pt";
            std::string out_name = save_dir + "/" + fname;
            auto out_bytes = torch::jit::pickle_save(mega_batch);
            write_bytes(out_name, out_bytes);
            for(auto& m : meta_buffer) {
                m.fname = fname;
                meta_file << m.fname << " " << m.start << " " << m.count << " " << m.tid << " " << m.cid << "\n";
            }
        }
        
        meta_file.close();
        std::cout << "\n  Done Partition " << pid << std::endl;
    }
    return 0;
}