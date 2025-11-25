#pragma once
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <random>
#include <graph.cpu.hpp>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    std::cout << "Initializing libstarrygl_sampler" << std::endl;
    m.def("transform_from_edge_index", &starrygl::transform_from_edge_index);

    py::class_<starrygl::CPUGraph>(m, "Graph")
        .def("sample_neighbor", &starrygl::CPUGraph::sample_neighbor)
        .def("sample_chunk_slice", &starrygl::CPUGraph::sample_chunk_slice);

    py::class_<starrygl::TemporalBlock<int64_t, int64_t>>(m, "TemporalBlock")
        // 添加默认构造函数
        .def("get_col_idx_tensor", &starrygl::TemporalBlock<int64_t, int64_t>::get_col_idx_tensor)
        .def("get_row_ptr_tensor", &starrygl::TemporalBlock<int64_t, int64_t>::get_row_ptr_tensor)
        .def("get_neighbors_tensor", &starrygl::TemporalBlock<int64_t, int64_t>::get_neighbors_tensor)
        .def("get_neighbors_ts_tensor", &starrygl::TemporalBlock<int64_t, int64_t>::get_neighbors_ts_tensor);
}