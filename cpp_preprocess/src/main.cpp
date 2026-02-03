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

namespace fs = std::filesystem;
using torch::Tensor;

// =========================================================================
// [Configuration]
// =========================================================================
const std::string RAW_DIR = "/mnt/data/zlj/starrygl-data/nparts/WIKI_004";
const std::string OUT_DIR = "/mnt/data/zlj/starrygl-data/processed_atomic/WIKI_004";
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

    GraphSampler(int partition_id) : pid(partition_id) {}

    void load_partition_data(const std::vector<std::string>& slot_files, 
                             const std::string& book_path, 
                             const std::string& edge_label_path) {
        std::cout << "  [Load] Loading history..." << std::endl;

        try {
            auto bytes = read_bytes(book_path);
            auto book = torch::jit::pickle_load(bytes).toTuple()->elements();
            // [Fix] Force CPU
            node_parts = book[1].toTensor().to(torch::kLong).to(torch::kCPU); 
        } catch (...) {}

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

    c10::impl::GenericList compute_route(Tensor gids) {
        c10::impl::GenericList plans(c10::AnyType::get());
        if (!node_parts.defined() || gids.numel() == 0) return plans;
        
        if (gids.max().item<int64_t>() >= node_parts.size(0)) {
             plans.push_back(c10::IValue()); return plans;
        }

        // [Fix] Ensure input is on CPU
        if (gids.is_cuda()) gids = gids.cpu();

        auto owners = node_parts.index_select(0, gids);
        auto mask = (owners != pid);
        
        if (mask.any().item<bool>()) {
            auto raw_send_ranks = owners.masked_select(mask); // 目标 Rank
            auto raw_send_indices = torch::arange(gids.size(0), torch::kLong).to(torch::kCPU).masked_select(mask);
            auto raw_send_remote = gids.masked_select(mask); // 全局 Node ID

            auto sort_idx = torch::argsort(raw_send_ranks);
            
            auto sorted_ranks = raw_send_ranks.index_select(0, sort_idx);
            auto sorted_indices = raw_send_indices.index_select(0, sort_idx);
            auto sorted_remote = raw_send_remote.index_select(0, sort_idx);
            auto send_sizes = torch::bincount(sorted_ranks, {}, NUM_PARTS);

            // C. 打包
            c10::impl::GenericDict plan(c10::StringType::get(), c10::AnyType::get());
            plan.insert("send_ranks", compress(sorted_ranks));
            plan.insert("send_indices", compress(sorted_indices));
            plan.insert("send_remote_indices", compress(sorted_remote));
            plan.insert("send_sizes", compress(send_sizes)); // [新增]
            
            plans.push_back(plan);
        } else {
            plans.push_back(c10::IValue()); 
        }
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

    c10::impl::GenericList build_batch_data(std::string type, std::vector<Tensor> task_data, int num_neg, int tid, int cid) {
        c10::impl::GenericList batch_list(c10::AnyType::get());
        Tensor l0_nodes, l0_ts;
        Tensor task_src, task_dst, task_ts, task_eid, task_label;
        c10::impl::GenericDict task_dict(c10::StringType::get(), c10::AnyType::get());

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
            
            // Generate Rand Indices on CPU
            auto rand_idx = torch::randint(0, dst_pool.numel(), {B * num_neg}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCPU));
            
            // [Fix] dst_pool is CPU, rand_idx is CPU. Safe.
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

        task_dict.insert("gids", compress(curr_gids));
        task_dict.insert("ts", compress(curr_ts));
        task_dict.insert("inv_map", compress(inv_map));
        batch_list.push_back(task_dict);

        c10::impl::GenericList routes(c10::AnyType::get());
        routes.push_back(compute_route(curr_gids));

        for(const auto& layer : LAYERS) {
            auto [src, dst_idx, ts, dt, eid] = sample_one_hop(curr_gids, curr_ts, layer, cid);
            
            if (!src.defined() || src.numel() == 0) {
                c10::impl::GenericDict empty_d(c10::StringType::get(), c10::AnyType::get());
                empty_d.insert("indptr", torch::zeros({1}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("indices", torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("eid",  torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("edge_dt",  torch::empty({0}, torch::kInt32).to(torch::kCPU));
                empty_d.insert("gids", compress(curr_gids)); 
                batch_list.push_back(empty_d);
                routes.push_back(c10::IValue());
                break;
            }

            // Incremental Merge Logic (Manual)
            std::vector<int64_t> next_gids_vec;
            std::vector<int64_t> src_indices_vec;
            int64_t estimated_size = curr_gids.size(0) + src.size(0);
            next_gids_vec.reserve(estimated_size);
            src_indices_vec.reserve(src.size(0));
            
            std::unordered_map<int64_t, int64_t> id_to_idx;
            id_to_idx.reserve(estimated_size);

            auto curr_ptr = curr_gids.data_ptr<int64_t>();
            for(int64_t i = 0; i < curr_gids.size(0); ++i) {
                int64_t node = curr_ptr[i];
                if (id_to_idx.find(node) == id_to_idx.end()) {
                    id_to_idx[node] = i;
                    next_gids_vec.push_back(node);
                }
            }
            int64_t current_count = next_gids_vec.size();

            auto src_ptr = src.data_ptr<int64_t>();
            for(int64_t i = 0; i < src.size(0); ++i) {
                int64_t node = src_ptr[i];
                auto it = id_to_idx.find(node);
                if (it != id_to_idx.end()) {
                    src_indices_vec.push_back(it->second);
                } else {
                    id_to_idx[node] = current_count;
                    next_gids_vec.push_back(node);
                    src_indices_vec.push_back(current_count);
                    current_count++;
                }
            }

            Tensor next_gids = torch::from_blob(next_gids_vec.data(), { (int64_t)next_gids_vec.size() }, torch::kLong).to(torch::kCPU).clone();
            Tensor src_new_indices = torch::from_blob(src_indices_vec.data(), { (int64_t)src_indices_vec.size() }, torch::kLong).to(torch::kCPU).clone();
            
            auto sort_perm = torch::argsort(dst_idx);
            Tensor sorted_indices = src_new_indices.index_select(0, sort_perm);
            Tensor sorted_eid_layer = eid.index_select(0, sort_perm);
            Tensor sorted_dt = dt.index_select(0, sort_perm);
            
            Tensor layer_indptr = torch::zeros({curr_gids.size(0) + 1}, torch::kLong).to(torch::kCPU);
            auto count_ptr = layer_indptr.data_ptr<int64_t>();
            auto dst_ptr = dst_idx.data_ptr<int64_t>();
            for(int64_t i=0; i<dst_idx.numel(); ++i) count_ptr[dst_ptr[i] + 1]++;
            for(int64_t i=0; i<curr_gids.size(0); ++i) count_ptr[i+1] += count_ptr[i];

            c10::impl::GenericDict layer_d(c10::StringType::get(), c10::AnyType::get());
            layer_d.insert("indptr", compress(layer_indptr));
            layer_d.insert("indices", compress(sorted_indices));
            layer_d.insert("eid", compress(sorted_eid_layer));
            layer_d.insert("edge_dt", compress(sorted_dt));
            layer_d.insert("gids", compress(next_gids)); 
            
            batch_list.push_back(layer_d);
            
            curr_gids = next_gids;
            curr_ts = torch::zeros_like(curr_gids); 
            routes.push_back(compute_route(curr_gids));
        }
        
        auto first_elem = batch_list.get(0).toGenericDict();
        first_elem.insert("comm_plan", routes);
        return batch_list;
    }
};

int main() {
    torch::set_num_threads(1); 
    omp_set_num_threads(NUM_THREADS);

    std::cout << "=== StarryGL C++ Preprocessor (CPU Forced) ===" << std::endl;

    for (int pid = 0; pid < NUM_PARTS; ++pid) {
        std::string p_dir = RAW_DIR + "/part_" + std::to_string(pid);
        if (!fs::exists(p_dir)) continue;
        
        std::string save_dir = OUT_DIR + "/part_" + std::to_string(pid);
        fs::create_directories(save_dir);

        std::ofstream meta_file(save_dir + "/meta.txt");

        GraphSampler sampler(pid);
        
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(p_dir)) {
            std::string name = entry.path().filename().string();
            if (name.find("slot_") == 0 && entry.path().extension() == ".pt") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());

        sampler.load_partition_data(files, RAW_DIR + "/partition_book.pt", RAW_DIR + "/edge_label.pt");

        int batch_idx = 0;
        int64_t current_bytes = 0;
        c10::impl::GenericList mega_batch(c10::AnyType::get());
        
        struct MetaInfo { std::string fname; int start; int count; int tid; int cid; };
        std::vector<MetaInfo> meta_buffer;

        std::cout << "  [Sample] Processing " << files.size() << " slots..." << std::endl;

        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0; i<files.size(); ++i) {
            try {
                int tid, cid;
                parse_file_ids(fs::path(files[i]).filename().string(), tid, cid);

                auto bytes = read_bytes(files[i]);
                auto dict = torch::jit::pickle_load(bytes).toGenericDict();
                
                // [Fix] Force CPU
                auto src = dict.at("src").toTensor().to(torch::kLong).to(torch::kCPU);
                auto dst = dict.at("dst").toTensor().to(torch::kLong).to(torch::kCPU);
                auto ts = dict.at("ts").toTensor().to(torch::kLong).to(torch::kCPU);
                auto eid = dict.at("eid").toTensor().to(torch::kLong).to(torch::kCPU);
                
                Tensor lbl = torch::zeros_like(src);
                if (sampler.edge_labels.defined()) {
                    // [Fix] sampler.edge_labels is forced CPU in load_partition_data
                    // eid is forced CPU above
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
                            meta_file << fname << " " << m.start << " " << m.count << " " << m.tid << " " << m.cid << "\n";
                        }

                        mega_batch = c10::impl::GenericList(c10::AnyType::get());
                        meta_buffer.clear();
                        current_bytes = 0;
                        if (batch_idx % 10 == 0) std::cout << "." << std::flush;
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "\nErr processing " << files[i] << ": " << e.what() << std::endl;
            }
        }
        
        if (!mega_batch.empty()) {
            std::string fname = "mega_batch_" + std::to_string(batch_idx++) + ".pt";
            std::string out_name = save_dir + "/" + fname;
            auto out_bytes = torch::jit::pickle_save(mega_batch);
            write_bytes(out_name, out_bytes);
            for(auto& m : meta_buffer) {
                meta_file << fname << " " << m.start << " " << m.count << " " << m.tid << " " << m.cid << "\n";
            }
        }
        
        meta_file.close();
        std::cout << "\n  Done Partition " << pid << std::endl;
    }
    return 0;
}