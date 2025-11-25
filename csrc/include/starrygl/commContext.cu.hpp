#pragma once
#include <nccl.h>
#include <cuda_runtime.h>
#include <iostream>
#include <starrygl.hpp>
#include <thrust/device_vector.h>
//#include <pybind11/numpy.py>

namespace starrygl{

    py::bytes create_nccl_id(){
        ncclUniqueId Id;
        ncclGetUniqueId(&Id);
        std::string temp(reinterpret_cast<const char *>(Id.internal),
                     sizeof(Id.internal));
        return py::bytes(temp);
    }
    class __attribute__((visibility("default"))) DistContext{
        ncclComm_t comm;
        int size;
        int rank;
        int world_size;
        ncclUniqueId nccl_id;
        void ptr_type(torch::Tensor tensor, void **ptr, ncclDataType_t *type)
        {
            if (tensor.options().dtype() == torch::kFloat16) {
                *type = ncclFloat16;
                *ptr = (void *)tensor.data_ptr<at::Half>();
                //*size = tensor.numel();
            }
            if (tensor.options().dtype() == torch::kFloat32) {
                *type = ncclFloat32;
                *ptr = (void *)tensor.data_ptr<float>();
                ///*size = tensor.numel();
            }
            if (tensor.options().dtype() == torch::kInt64) {
                *type = ncclInt64;
                *ptr = (void *)tensor.data_ptr<int64_t>();
                //*size = tensor.numel();
            }
        }
        public:
            DistContext(){
            }
            DistContext(int rank_, int world_size_, int size_, py::bytes id)
                :rank(rank_), world_size(world_size_),size(size_)
            {
                std::string id_str = id;
                memcpy(nccl_id.internal, id_str.data(), sizeof(nccl_id.internal));
                ncclCommInitRank(&comm, world_size, nccl_id, rank);
            }
            ~DistContext(){
                ncclCommDestroy(comm);

            }
            int get_rank() { return rank; }

            int get_size() { return size; }

            int get_world_size(){
                return world_size;
            }

            int get_device()
            {
                int dev;
                ncclCommCuDevice(comm, &dev);
                return dev;
            }
            int64_t get_prefix_sum(thrust::device_vector<int64_t> &d_vec, int index)
            {
                 int64_t sum_of_first_i = thrust::reduce(d_vec.begin(), d_vec.begin() + index, 0, thrust::plus<int64_t>());
                return sum_of_first_i;
            }
            int64_t sum(thrust::device_vector<int64_t> &d_vec){
                int64_t sum = 0;
                thrust::reduce(d_vec.begin(), d_vec.end(), sum, thrust::plus<int64_t>());
                return sum;
            }
            void send_indices_to_all(
                int64_t *comm_ptr_buffer,
                thrust::device_vector<int64_t> &comm_size,
                int64_t *(&recv_index_buffer),
                thrust::device_vector<int64_t> &recv_size,
                cudaStream_t stream)
            {
                int64_t *send_size_buffer;
                int64_t *recv_size_buffer;
                CHECK(ncclMemAlloc((void **)&send_size_buffer, world_size * sizeof(int64_t)));
                cudaMemcpy(send_size_buffer, 
                           thrust::raw_pointer_cast(comm_size.data()),
                           world_size * sizeof(int64_t),
                            cudaMemcpyDeviceToDevice);
                CHECK(ncclMemAlloc((void **)&recv_size_buffer, world_size * sizeof(int64_t)));
                for(int i = 0; i < world_size; i++){
                    //如果位于同一台机器
                    if(i/size == rank/size){
                        recv_size_buffer[i] = comm_size[i];
                        continue;
                    }
                    ncclSend(
                        send_size_buffer + i, 1,
                        ncclInt64, i, comm, stream
                    );
                    ncclRecv(
                        recv_size_buffer + i, 1,
                        ncclInt64, i, comm, stream
                    );
                }
                cudaMemcpy(thrust::raw_pointer_cast(recv_size.data()),
                           recv_size_buffer,
                           world_size * sizeof(int64_t),
                           cudaMemcpyDeviceToDevice);
                CHECK(cudaFree(send_size_buffer));
                CHECK(cudaFree(recv_size_buffer));
                int64_t total_recv_size = sum(recv_size);
                CHECK(ncclMemAlloc(
                    (void**)&recv_index_buffer,
                    total_recv_size * sizeof(int64_t)
                ));
                for(int i = 0; i < world_size; i++){
                    //如果位于同一台机器
                    if(i/size == rank/size){
                        cudaMemcpy(
                            (int64_t*)recv_index_buffer + get_prefix_sum(recv_size, i),
                            (int64_t*)comm_ptr_buffer + get_prefix_sum(comm_size, i),
                            comm_size[i] * sizeof(int64_t),
                            cudaMemcpyDeviceToDevice
                        );
                        continue;
                    }
                    ncclSend(
                        (int64_t*)comm_ptr_buffer + get_prefix_sum(comm_size, i),
                        comm_size[i],
                        ncclInt64, i, comm, stream
                    );
                    ncclRecv(
                        (int64_t*)recv_index_buffer + get_prefix_sum(recv_size, i),
                        recv_size[i],
                        ncclInt64, i, comm, stream
                    );
                }
            }
            void send_data_to_all(void *send_data_buffer, 
                thrust::device_vector<int64_t> &comm_size,
                void *recv_data_buffer,
                thrust::device_vector<int64_t> &recv_size,
                int stride,
                cudaStream_t stream
            )
            {
                for(int i = 0 ; i < world_size; i++){
                    if(i/size == rank/size){
                        cudaMemcpy(
                            (float *)recv_data_buffer + get_prefix_sum(recv_size, i) * stride,
                            (float *)send_data_buffer + get_prefix_sum(comm_size, i)  * stride,
                            comm_size[i] * stride * sizeof(float),
                            cudaMemcpyDeviceToDevice
                        );
                        continue;
                    }
                    ncclSend(
                        (float *)send_data_buffer +  get_prefix_sum(comm_size, i) * stride,
                        comm_size[i] * stride,
                        ncclFloat32, i, comm, stream
                    );
                    ncclRecv(
                        (float *)recv_data_buffer + get_prefix_sum(recv_size, i) * stride,
                        recv_size[i] * stride,
                        ncclFloat32, i, comm, stream
                    );

                }
            }
            // template <typename T>
            // void padding(thrust::device_vector<T> &vec,
            //              thrust::device_vector<T> &padding_vec,
            //              int counts,
            //              thrust::device_vector<T> old_size,
            //              size_t new_size, 
            //              size_t stride,
            //              cudaStream_t stream)
            // {
            //     offset = 0;
            //     for(int i = 0; i < counts; i++){
            //         thrust::copy(vec.begin() + offset*stride, 
            //                     vec.begin() + (offset + old_size[i])* stride,
            //                     padding_vec.begin() + new_size * i * stride);
            //         offset += old_size[i];
            //     }
            // }
            // //recv_counts是发起的index请求
            // void get_padding_ind(thrust::device_vector<int64_t> &recv_ind,
            //                 thrust::device_vector<int64_t> &send_ind,
            //                 size_t *recv_counts, size_t *send_counts, 
            //                 size_t *counts, cudaStream_t stream)
            // {
            //     //ncclAllToAll(
            //     //    recv_counts, send_counts, world_size,
            //     //    ncclInt64, comm, stream
            //     //);
            //     for(int i = 0;i<world_size;i++){

            //     }
            //     *counts = 0;
            //     for(int i = 0; i < world_size; i++){
            //         *counts = max(*counts, static_cast<size_t>(send_counts[i]));
            //     }
            //     ncclAllReduce(counts, counts, 1, ncclInt64, ncclMax, comm, stream);
            //     recv_ind.resize(*counts * world_size, -1);
            //     thrust::device_vector<int64_t> padding_recv_ind((*counts) * world_size, 0); 
            //     send_ind.resize(*counts * world_size, -1);
            //     padding<int64_t>(send_ind, padding_recv_ind, world_size, send_counts, *counts, 1, stream);
            //     ncclAllToAll(
            //         send_ind.data().get(), padding_recv_ind.data().get(),
            //         (*counts), ncclInt64, comm, stream
            //     );
                
            // }
            // template <typename T>
            // float* data2all(  th::Tensor data, size_t counts,
            //                 void *recv_counts, void *send_counts,
            //                 int stride, cudaStream_t stream)
            // {
            //     thrust::device_vector<float> recv_data(world_size * counts, 0);
            //     ncclAllToAll(
            //         data.data_ptr<float>(), recv_data.data().get(), 
            //         counts * stride, ncclInt64, comm, stream
            //     );
            //     size_t total_recv_counts= 0;
            //     //for(int i = 0; i< world_size; i++)total_recv_counts += recv_counts[i];
            //     //thrust::device_vector<float> data(total_recv_counts, 0);
            //     for(int i = 0; i < world_size; i++){
            //         thrust::copy(recv_data.begin() + i*counts*stride, 
            //                     recv_data.begin() + (i*counts + total_recv_counts) *stride,
            //                     recv_data.begin() + total_recv_counts * stride);
            //         total_recv_counts += recv_counts[i];
            //     }
            //     return recv_data.data().get();
            // }

    // Compute maximum count across all ranks

    };
};

#define cudaCheckError()                                                       \
    {                                                                          \
        cudaError_t e = cudaGetLastError();                                    \
        if (e != cudaSuccess) {                                                \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,           \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }



void register_nccl_comm(pybind11::module &m)
{
    m.def("create_nccl_id", &starrygl::create_nccl_id);
    py::class_<starrygl::DistContext>(m, "DistContext")
        .def(py::init<int, int, int, py::bytes>())
        .def("rank", &starrygl::DistContext::get_rank)
        .def("size", &starrygl::DistContext::get_size)
        .def("device", &starrygl::DistContext::get_device);
        //.def("all2all", &starrygl::DistContext::all2all);
}