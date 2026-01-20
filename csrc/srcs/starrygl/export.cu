#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "sample.cu.hpp"
#include "commContext.cu.hpp"
#include "feature.cu.hpp"

namespace py = pybind11;

// [Fix 1]: 定义类型别名，确保 T 和 TS 可用
using T = int64_t;
using TS = int64_t;

// [Fix 2]: 将辅助函数移到模块定义之外。
// 这样它就变成了全局函数，Lambda 调用时不再需要捕获，彻底解决 capture list 报错。
torch::Tensor make_cuda_tensor(size_t size) {
    auto options = torch::dtype(torch::kInt64).device(torch::kCUDA);
    return torch::empty({(int64_t)size}, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    using Root_T = TemporalRoot<T, TS>;
    using Negative_T = NegativeRoot<T, TS>;
    using Neighbors_T = TemporalNeighbor<T, TS>;
    using Result_T = TemporalResult<T, TS>;

    // 绑定 TemporalRoot 类
    // 注意：Lambda 的捕获列表 [] 现在是空的，因为 make_cuda_tensor 是全局函数
    py::class_<Root_T>(m, "TemporalRoot")
        .def_property_readonly("roots", [](const Root_T& self) {
            if (self.roots.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.roots.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.roots.data()),
                       self.roots.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("ts_ptr", [](const Root_T& self) {
            if (self.ts_ptr.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.ts_ptr.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.ts_ptr.data()),
                       self.ts_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("ts", [](const Root_T& self) {
            if (self.ts.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.ts.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.ts.data()),
                       self.ts.size() * sizeof(TS),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("q_ptr", [](const Root_T& self) {
            if (self.q_ptr.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.q_ptr.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.q_ptr.data()),
                       self.q_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("q_eid", [](const Root_T& self) {
            if (self.q_eid.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.q_eid.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.q_eid.data()),
                       self.q_eid.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("root_num", [](const Root_T& self) { return self.root_num; })
        .def_property_readonly("root_ts_num", [](const Root_T& self) { return self.root_ts_num; })
        .def("__repr__", [](const Root_T& self) {
            return "<TemporalRoot with " + std::to_string(self.root_num) + " roots>";
        });
    
    // 绑定 NegativeRoot 类
    py::class_<Negative_T>(m, "NegativeRoot")
        .def_property_readonly("roots", [](const Negative_T& self) {
            if (self.roots.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.roots.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.roots.data()),
                       self.roots.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("ts", [](const Negative_T& self) {
            if (self.ts.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.ts.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.ts.data()),
                       self.ts.size() * sizeof(TS),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("root_num", [](const Negative_T& self) { return self.root_num; })
        .def("__repr__", [](const Negative_T& self) {
            return "<NegativeRoot with " + std::to_string(self.root_num) + " roots>";
        });
    
    // 绑定 TemporalNeighbor 类
    py::class_<Neighbors_T>(m, "TemporalNeighbor")
        .def_property_readonly("root_start_ptr", [](const Neighbors_T& self) {
            if (self.root_start_ptr.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.root_start_ptr.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.root_start_ptr.data()),
                       self.root_start_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("root_end_ptr", [](const Neighbors_T& self) {
            if (self.root_end_ptr.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.root_end_ptr.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.root_end_ptr.data()),
                       self.root_end_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbors", [](const Neighbors_T& self) {
            if (self.neighbors.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.neighbors.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.neighbors.data()),
                       self.neighbors.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbors_ts", [](const Neighbors_T& self) {
            if (self.neighbors_ts.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.neighbors_ts.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.neighbors_ts.data()),
                       self.neighbors_ts.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbors_eid", [](const Neighbors_T& self) {
            if (self.neighbors_eid.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.neighbors_eid.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.neighbors_eid.data()),
                       self.neighbors_eid.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbors_dt", [](const Neighbors_T& self) {
            if (self.neighbors_dt.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.neighbors_dt.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.neighbors_dt.data()),
                       self.neighbors_dt.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbor_num", [](const Neighbors_T& self) { return self.neighbor_num; })
        .def("__repr__", [](const Neighbors_T& self) {
            return "<TemporalNeighbor with " + std::to_string(self.neighbor_num) + " neighbors>";
        });
    
    // 绑定 TemporalResult 类
    py::class_<Result_T>(m, "TemporalResult")
        .def_readwrite("roots", &Result_T::roots)
        .def_readwrite("neg_roots", &Result_T::neg_roots)
        .def_readwrite("neighbors_list", &Result_T::neighbors_list)
        // [Fix 3]: 为 neighbors_list 创建别名 "neighbors"，解决 Python 端 AttributeError
        .def_readwrite("neighbors", &Result_T::neighbors_list)
        .def_property_readonly("nodes_remapper_id", [](const Result_T& self) {
            if (self.nodes_remapper_id.empty()) return make_cuda_tensor(0);
            torch::Tensor t = make_cuda_tensor(self.nodes_remapper_id.size());
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.nodes_remapper_id.data()),
                       self.nodes_remapper_id.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def("__repr__", [](const Result_T& self) {
            return "<TemporalResult with " + std::to_string(self.neighbors_list.size()) + " neighbor layers>";
        });
    
    // 绑定 CUDAGraph 类
    py::class_<CUDAGraph>(m, "CUDAGraph")
        .def("slice_by_chunk_ts", &CUDAGraph::slice_by_chunk_ts,
             py::arg("chunks"), 
             py::arg("time_begin"), py::arg("time_end"),
             py::arg("using_full_timestamp") = false,
             py::arg("stream") = 0,
             py::call_guard<py::gil_scoped_release>())
         .def("sample_src_in_chunks_khop", &CUDAGraph::sample_src_in_chunks_khop,
              py::arg("postive_root"), py::arg("negative_root"), 
              py::arg("k"), py::arg("layers"),
              py::arg("allowed_offset"),
              py::arg("equal_root_time"),
              py::arg("keep_root_time"),
              py::arg("sample_type"),
              py::call_guard<py::gil_scoped_release>())
         .def("get_seeds_in_chunks", &CUDAGraph::get_seeds_in_chunks,
            py::arg("chunks"),
            py::arg("time_begin"),
            py::arg("time_end"),
            py::arg("using_full_timestamp"),
            py::call_guard<py::gil_scoped_release>())
         .def("get_negative_root", &CUDAGraph::get_negative_root,
            py::arg("negative_root"),
            py::arg("negative_time"))
         .def("submit_query", &CUDAGraph::submit_query,
            py::arg("time_start"),
            py::arg("time_end"),
            py::arg("chunk_list"),
            py::arg("test_generate_samples"),
            py::arg("test_generate_samples_ts"),
            py::arg("sample_type"), 
            py::arg("layers"),
            py::arg("fanout"),
            py::arg("allowed_offset"), 
            py::arg("equal_root_time"),
            py::arg("keep_root_time"),
            py::arg("op"), 
            py::call_guard<py::gil_scoped_release>())
        .def("get", &CUDAGraph::get,
            py::call_guard<py::gil_scoped_release>())
    ;

    m.def("from_edge_index", [](T n, T chunk_size, 
                            const torch::Tensor& src,
                            const torch::Tensor& dst,
                            const torch::Tensor& ts,
                            const torch::Tensor& eid,
                            const torch::Tensor& row_chunk_mapper,
                            uint64_t stream_ptr,
                            int rank) -> CUDAGraph {
        try {
            return from_edge_index(n, chunk_size, src, dst, ts, eid,
                                    row_chunk_mapper, stream_ptr, rank);
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            throw py::error_already_set();
        }
    },
          py::arg("n"), py::arg("chunk_size"), 
          py::arg("src"), py::arg("dst"), py::arg("ts"), py::arg("eid"), py::arg("mapper"), 
          py::arg("stream") = 0,
          py::arg("rank") = 0,
          py::call_guard<py::gil_scoped_release>());
    
    register_nccl_comm(m);
    register_cuda_feature(m);
}