#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "sample.cu.hpp"
#include  "commContext.cu.hpp"
#include "feature.cu.hpp"
namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // using Block = TemporalBlock<int64_t,int64_t>;
    using Root_T = TemporalRoot<int64_t, int64_t>;
    using Negative_T = TemporalRoot<int64_t, int64_t>;
    using Neighbors_T = TemporalNeighbor<int64_t, int64_t>;
    using Result_T = TemporalResult<int64_t, int64_t>;
    py::class_<Root_T>(m, "TemporalRoot")
        .def_property_readonly("roots", [](const Root_T& self) {
            if (self.roots.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.roots.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.roots.data()),
                       self.roots.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("ts_ptr", [](const Root_T& self) {
            if (self.ts_ptr.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.ts_ptr.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.ts_ptr.data()),
                       self.ts_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("ts", [](const Root_T& self) {
            if (self.ts.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.ts.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.ts.data()),
                       self.ts.size() * sizeof(TS),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("q_ptr", [](const Root_T& self) {
            if (self.q_ptr.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.q_ptr.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.q_ptr.data()),
                       self.q_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("q_eid", [](const Root_T& self) {
            if (self.q_eid.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.q_eid.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.q_eid.data()),
                       self.q_eid.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("root_num", [](const Root_T& self) { return self.root_num; })
        .def_property_readonly("root_ts_num", [](const Root_T& self) { return self.root_ts_num; })
        .def("__repr__", [](const Root_T& self) {
            return "<TemporalRoot with " + std::to_string(self.root_num) + " roots and " + 
                   std::to_string(self.root_ts_num) + " timestamps>";
        });
    
    // 绑定 NegativeRoot 类
    py::class_<Negative_T>(m, "NegativeRoot")
        .def_property_readonly("roots", [](const Negative_T& self) {
            if (self.roots.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.roots.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.roots.data()),
                       self.roots.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("ts", [](const Negative_T& self) {
            if (self.ts.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.ts.size()}, torch::kInt64);
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
            if (self.root_start_ptr.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.root_start_ptr.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.root_start_ptr.data()),
                       self.root_start_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("root_end_ptr", [](const Neighbors_T& self) {
            if (self.root_end_ptr.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.root_end_ptr.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.root_end_ptr.data()),
                       self.root_end_ptr.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbors", [](const Neighbors_T& self) {
            if (self.neighbors.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.neighbors.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.neighbors.data()),
                       self.neighbors.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbors_ts", [](const Neighbors_T& self) {
            if (self.neighbors_ts.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.neighbors_ts.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.neighbors_ts.data()),
                       self.neighbors_ts.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def_property_readonly("neighbors_eid", [](const Neighbors_T& self) {
            if (self.neighbors_eid.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.neighbors_eid.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.neighbors_eid.data()),
                       self.neighbors_eid.size() * sizeof(T),
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
        .def_property_readonly("nodes_remapper_id", [](const Result_T& self) {
            if (self.nodes_remapper_id.empty()) return torch::empty({0}, torch::kInt64);
            torch::Tensor t = torch::empty({(int64_t)self.nodes_remapper_id.size()}, torch::kInt64);
            cudaMemcpy(t.data_ptr<int64_t>(),
                       thrust::raw_pointer_cast(self.nodes_remapper_id.data()),
                       self.nodes_remapper_id.size() * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            return t;
        })
        .def("__repr__", [](const Result_T& self) {
            return "<TemporalResult with " + std::to_string(self.neighbors_list.size()) + " neighbor layers>";
        });
    
    // py::class_<Block>(m, "TemporalBlock")
    //     .def_property_readonly("neighbors", [](const Block& self) {
    //         if (self.neighbors.empty()) return torch::empty({0}, torch::kInt64);
    //         torch::Tensor t = torch::empty(self.neighbors.size(), torch::kInt64);
    //         cudaMemcpy(t.data_ptr<int64_t>(),
    //                    thrust::raw_pointer_cast(self.neighbors.data()),
    //                    self.neighbors.size() * sizeof(int64_t),
    //                    cudaMemcpyDeviceToDevice);
    //         return t;
    //     })
    //     .def_property_readonly("neighbors_ts", [](const Block& self) {
    //         if (self.neighbors_ts.empty()) return torch::empty({0}, torch::kInt64);
    //         torch::Tensor t = torch::empty(self.neighbors_ts.size(), torch::kInt64);
    //         cudaMemcpy(t.data_ptr<int64_t>(),
    //                    thrust::raw_pointer_cast(self.neighbors_ts.data()),
    //                    self.neighbors_ts.size() * sizeof(int64_t),
    //                    cudaMemcpyDeviceToDevice);
    //         return t;
    //     })
    //     .def_property_readonly("neighbors_eid", [](const Block& self) {
    //         if (self.neighbors_eid.empty()) return torch::empty({0}, torch::kInt64);
    //         torch::Tensor t = torch::empty(self.neighbors_eid.size(), torch::kInt64);
    //         cudaMemcpy(t.data_ptr<int64_t>(),
    //                    thrust::raw_pointer_cast(self.neighbors_eid.data()),
    //                    self.neighbors_eid.size() * sizeof(int64_t),
    //                    cudaMemcpyDeviceToDevice);
    //         return t;
    //     })
    //     .def_property_readonly("row_ptr", [](const Block& self) {
    //         if (self.row_ptr.empty()) return torch::empty({0}, torch::kInt64);
    //         torch::Tensor t = torch::empty(self.row_ptr.size(), torch::kInt64);
    //         cudaMemcpy(t.data_ptr<int64_t>(),
    //                    thrust::raw_pointer_cast(self.row_ptr.data()),
    //                    self.row_ptr.size() * sizeof(int64_t),
    //                    cudaMemcpyDeviceToDevice);
    //         return t;
    //     })
    //     .def_property_readonly("row_idx", [](const Block& self) {
    //         if (self.row_idx.empty()) return torch::empty({0}, torch::kInt64);
    //         torch::Tensor t = torch::empty(self.row_idx.size(), torch::kInt64);
    //         cudaMemcpy(t.data_ptr<int64_t>(),
    //                    thrust::raw_pointer_cast(self.row_idx.data()),
    //                    self.row_idx.size() * sizeof(int64_t),
    //                    cudaMemcpyDeviceToDevice);
    //         return t;
    //     })
    //     .def_property_readonly("layer_ptr", [](const Block& self) {
    //         if (self.layer_ptr.empty()) return torch::empty({0}, torch::kInt64);
    //         torch::Tensor t = torch::empty(self.layer_ptr.size(), torch::kInt64);
    //         cudaMemcpy(t.data_ptr<int64_t>(),
    //                    thrust::raw_pointer_cast(self.layer_ptr.data()),
    //                    self.layer_ptr.size() * sizeof(int64_t),
    //                    cudaMemcpyDeviceToDevice);
    //         return t;
    //     })
    //     .def_property_readonly("layer_num", [](const Block& self) {
    //         return self.layer;
    //     });
    
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
            py::arg("negative_time"));
    m.def("from_edge_index", [](T n, T chunk_size,
                            const torch::Tensor& src,
                            const torch::Tensor& dst,
                            const torch::Tensor& ts,
                            const torch::Tensor& row_chunk_mapper,
                            uint64_t stream_ptr) -> CUDAGraph {
        try {
            return from_edge_index(n, chunk_size, src, dst, ts, row_chunk_mapper, stream_ptr);
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
            throw py::error_already_set();
        }
    },
          py::arg("n"), py::arg("chunk_size"),
          py::arg("src"), py::arg("dst"), py::arg("ts"), py::arg("mapper"), py::arg("stream") = 0,
          py::call_guard<py::gil_scoped_release>());
    register_nccl_comm(m);
    register_cuda_feature(m);
}