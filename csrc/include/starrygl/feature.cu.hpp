#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <atomic>
#include <iostream>
#include <string>
#include <torch/csrc/utils/python_numbers.h>
#include <unordered_map>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include  <commContext.cu.hpp>

constexpr const int64_t part_param = (0xFFFF ^ (1<<14));
__device__ __host__
int get_partition(int64_t x){
    return (x>>48) & part_param;
}
__device__ __host__
int64_t get_loc(int64_t x){
    return x & 0x0000FFFFFFFFFFFFL;
}
__device__ __host__
bool is_shared(int64_t x){
    return (bool)(x>>62);
}

#define quiverRegister(ptr, size, flag)                                        \
    ;                                                                          \
    {                                                                          \
        size_t BLOCK = 1000000000;                                             \
        void *register_ptr = (void *)ptr;                                      \
        for (size_t pos = 0; pos < size; pos += BLOCK) {                       \
            size_t s = BLOCK;                                                  \
            if (size - pos < BLOCK) { s = size - pos; }                        \
            cudaHostRegister(register_ptr + pos, s, flag);                     \
        }                                                                      \
    }
    
__global__ void tensor_gather(char **dev_ptrs, const int64_t *offsets,
                                     const int device_count,
                                     const int64_t *indices, int indice_length,
                                     char *res, const int stride,
                                     const int *access_book,
                                     const int ignore_access_book,
                                     const int gpu_size
                                )
{
    //
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;
    // each warp take charge of one-feature copy
    unsigned int warp_id = tid / warpSize;
    unsigned int warp_step = step / warpSize;
    unsigned int warp_start = warp_id;
    unsigned int thread_start = tid % warpSize;
    int64_t dev_index = 0;
    int64_t dev_offset = 0;
    char *dev_ptr;
    int64_t src_copy_start = 0;
    int64_t dst_copy_start = 0;
    unsigned int local_start = thread_start;
    while (warp_start < indice_length) {
        if(indices[warp_start] == -1){
            continue;
        }
        local_start = thread_start;
        if (dev_index != -1 && (ignore_access_book || access_book[dev_index] == 1)) {
            dev_index = 1-is_shared(indices[warp_start]);
            dev_offset = get_loc(indices[warp_start]);
            if(dev_index > 0){
                dev_index += get_partition(indices[warp_start])%gpu_size;
                dev_offset -= offsets[0];
            }
            dev_ptr = dev_ptrs[dev_index];
            //dev_offset = get_loc(indices[warp_start])//indices[warp_start] - offsets[dev_index];
            src_copy_start = dev_offset * stride;
            dst_copy_start = warp_start * stride;
            for (; local_start < stride; local_start += warpSize) {
                res[dst_copy_start + local_start] =
                    dev_ptr[src_copy_start + local_start];
            }
        }
        warp_start += warp_step;
    }
}

namespace starrygl
{


#define CHECK_CPU(x) AT_ASSERTM(!x.device().is_cuda(), #x " must be CPU tensor")
class ShardTensorItem
{
  public:
    int rank;
    int group;
    int device;
    cudaIpcMemHandle_t mem_handle; 
    cudaIpcMemHandle_t ts_handle;
    std::vector<int> shape;
    // for now we assume it is all float
    int element_size;
    ShardTensorItem(int device_, cudaIpcMemHandle_t mem_handle_,
                    std::vector<int> shape_)
        : device(device_), mem_handle(mem_handle_), shape(shape_)
    {
        rank = 0;
        group = 0;
    }
    ShardTensorItem(int rank, int device_, cudaIpcMemHandle_t mem_handle_,
                    std::vector<int> shape_)
        : group(group), rank(rank), device(device_), mem_handle(mem_handle_), shape(shape_)
    {

    }
    ShardTensorItem(){

    };
    std::tuple<int, int, py::bytes, std::vector<int>> share_ipc()
    {
        auto _handle = PyBytes_FromStringAndSize((char *)&mem_handle,
                                                 CUDA_IPC_HANDLE_SIZE);
        auto bytes_obj = py::reinterpret_steal<py::object>((PyObject *)_handle);
        return std::make_tuple(device, element_size, bytes_obj, shape);
    }
    void from_ipc(std::tuple<int, int, std::string, std::vector<int>> ipc_data)
    {

        device = std::get<0>(ipc_data);
        element_size = std::get<1>(ipc_data);
        shape = std::get<3>(ipc_data);
        auto handle = std::get<2>(ipc_data);
        auto ipc_handle =
            reinterpret_cast<const cudaIpcMemHandle_t *>(handle.c_str());

        mem_handle = *ipc_handle;
    }
};
class ShardTensor
{
  public:
    ShardTensor(int device, DistContext context_) : device_(device), inited_(false), device_count_(0)
    {

        context = context_;
        offset_list_.push_back(0);
    }

    size_t get_tensor_bytes(torch::Tensor tensor)
    {
        // assume it's float
        int dim = tensor.dim();
        size_t total_bytes = element_size;
        for (int index = 0; index < dim; index++) {
            total_bytes *= tensor.sizes()[index];
        }
        return total_bytes;
    }
    std::vector<int> get_tensor_shape(torch::Tensor tensor)
    {
        std::vector<int> shape;
        int dim = tensor.dim();
        for (int index = 0; index < dim; index++) {
            shape.push_back(tensor.sizes()[index]);
        }
        return shape;
    }

    void append(ShardTensorItem item)
    {
        cudaSetDevice(device_);
        if (!inited_) {
            shape_.resize(item.shape.size());
            shape_[0] = 0;
            auto tensor_sizes = item.shape;
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
        }
        offset_list_.push_back(offset_list_[offset_list_.size() - 1] +
                               item.shape[0]);

        // Check accessbility
        if(item.group == group_){
            if (item.device >= 0) {
                // TODO
                int access_i_j, access_j_i;
                cudaDeviceCanAccessPeer(&access_i_j, device_, item.device);
                cudaDeviceCanAccessPeer(&access_j_i, item.device, device_);
                if ((access_i_j && access_j_i) || device_ == item.device) {
                    access_book.push_back(1);
                    // printf("%d <-> %d support peer access \n", device_,
                    // item.device);
                } else {
                    access_book.push_back(0);
                    // printf("%d <-> %d dont support peer access \n", device_,
                    // item.device);
                }

            } else {
                access_book.push_back(1);
                // printf("%d <-> CPU support peer access \n", device_);
            }
            access_rank.push_back(item.rank);
        }
        else{
            access_book.push_back(0);
            access_rank.push_back(item.rank);
        }
        // get dev_ptr that can be accessed from this process
        void *ptr = NULL;
        tensor_devices_.push_back(item.device);
        if (!access_book[access_book.size() - 1]) {
            cudaSetDevice(item.device);
            cudaIpcOpenMemHandle(&ptr, item.mem_handle,
                                 cudaIpcMemLazyEnablePeerAccess);
            cudaSetDevice(device_);
            // printf("WARNING: Tensor from device %d can NOT be accessed in
            // kernel launched on device %d \n", item.device, device_);
        } else {
            cudaIpcOpenMemHandle(&ptr, item.mem_handle,
                                 cudaIpcMemLazyEnablePeerAccess);
        }

        //
        dev_ptrs_.push_back(ptr);
        element_size = item.element_size;
        shape_[0] += item.shape[0];
        device_count_ += 1;
        cudaCheckError();
    }
        void append(torch::Tensor &tensor, int target_device)
    {
        CHECK_CPU(tensor);
        // for now, we assume tensor is added ordered
        if (!inited_) {
            shape_.resize(tensor.dim());
            shape_[0] = 0;
            auto tensor_sizes = tensor.sizes();
            for (int index = 1; index < shape_.size(); index++) {
                shape_[index] = tensor_sizes[index];
            }
            inited_ = true;
        }
        element_size = tensor.element_size();
        tensor_shapes_.push_back(get_tensor_shape(tensor));

        offset_list_.push_back(offset_list_[offset_list_.size() - 1] +
                               tensor.sizes()[0]);

        void *ptr = NULL;
        size_t data_size = get_tensor_bytes(tensor);
        tensor_devices_.push_back(target_device);
        if (target_device >= 0) {
            // if target_device >= 0, it means we use p2p
            // printf("LOG >>> Malloc Data On Device %d With %ulld Bytes\n",
            // target_device, data_size);
            cudaSetDevice(target_device);
            cudaMalloc(&ptr, data_size);
            cudaMemcpy(ptr, tensor.data_ptr(), data_size,
                       cudaMemcpyHostToDevice);
            cudaSetDevice(device_);

            // decide access book

            int access_i_j, access_j_i;
            cudaDeviceCanAccessPeer(&access_i_j, device_, target_device);
            cudaDeviceCanAccessPeer(&access_j_i, target_device, device_);
            if ((access_i_j && access_j_i) || device_ == target_device) {
                access_book.push_back(1);
                // printf("%d <-> %d support peer access \n", device_,
                // target_device);
            } else {
                access_book.push_back(0);
                // printf("%d <-> %d dont support peer access \n", device_,
                // target_device);
            }

        } else {
            cudaSetDevice(device_);
            // if target_device < 0, it means we use Zero-Copy
            quiverRegister(tensor.data_ptr(), data_size,
                           cudaHostRegisterMapped);
            cudaHostGetDevicePointer(&ptr, (void *)tensor.data_ptr(), 0);
            access_book.push_back(1);
            // printf("%d <-> CPU support peer access \n", device_);
        }

        dev_ptrs_.push_back(ptr);

        shape_[0] += tensor.size(0);
        device_count_ += 1;
    }
    torch::Tensor operator[](torch::Tensor &indices)
    {
        /*
        __global__ void starrygl_tensor_gather(const int64_t** dev_ptrs, const
        int64_t* offsets, const int device_count, const int64_t* indices, int
        indice_length, const float* res, const int item_byte_size){
        torch::zeros((100,100),torch::KF32);
        */
        int current_device = 0;
        cudaGetDevice(&current_device);
        auto stream = at::cuda::getCurrentCUDAStream();

        std::vector<int64_t> res_shape(shape_);
        //res_shape[0] = indices.numel();
        // decide Tensor

        auto options = torch::TensorOptions();
        if(element_size == 2){
            options = options.dtype(torch::kFloat16).device(torch::kCUDA, current_device);
        }else if(element_size == 4){
            options = options.dtype(torch::kFloat32).device(torch::kCUDA, current_device);
        }

                    
        
        //cudaCheckError();

        // Device Data
        // for(int index = 0; index < offset_list_.size(); index++){
        //    std::cout<<"offset " << offset_list_[index]<<std::endl;
        //    std::cout<<"access_book[index] " << access_book[index]<<std::endl;
        //}

        char **buffers_device;
        int64_t *offset_device;
        int *access_book_device;

        auto val = get_device_pointers(current_device);
        buffers_device = std::get<0>(val);
        offset_device = std::get<1>(val);
        access_book_device = std::get<2>(val);


        auto counting_iter = thrust::make_counting_iterator(0);
        auto *data_ptr = indices.data_ptr<float>();
        int64_t indices_length = indices.numel();
        thrust::device_vector<int64_t> thrust_vec(data_ptr, data_ptr + indices_length);
        //和每个机器通信的指针起始位置
        thrust::device_vector<int64_t> comm_ptr(indices_length, 0);
        thrust::device_vector<int64_t> comm_size(context.get_world_size(), 0);
        thrust::device_vector<int64_t> recv_size(context.get_world_size(), 0);
        //原始索引位置
        thrust::device_vector<int64_t> res_ptr(indices_length, 0);
       
        auto comm_ptr_begin = 0;
        auto res_ptr_begin = 0;
        for(int i = 0; i < context.get_world_size() ; i++){
            auto end = thrust::copy_if(thrust::device, thrust_vec.begin(), thrust_vec.end(), comm_ptr.begin()+comm_ptr_begin, is_partition(i));
            thrust::copy_if(thrust::device, counting_iter, counting_iter + indices_length, thrust_vec.begin(), res_ptr.begin()+res_ptr_begin, is_partition(i));
            comm_size[i] = end-(comm_ptr.begin()+comm_ptr_begin);
        }
        ncclGroupStart();
        //计算提取特征的位置
        int64_t *comm_ptr_buffer;
        int64_t *recv_index_buffer;
        CHECK(ncclMemAlloc((void **)&comm_ptr_buffer, indices_length * sizeof(int64_t)));
        //拷贝赋值 
        cudaMemcpy(comm_ptr_buffer, thrust::raw_pointer_cast(comm_ptr.data()), indices_length * sizeof(int64_t), cudaMemcpyDeviceToDevice);
        //cudaMemcopy(comm_ptr_buffer, comm_ptr.data().get(), indice_length * sizeof(int64_t), cudaMemcpyDeviceToDevice);
        context.send_indices_to_all(comm_ptr_buffer, comm_size, recv_index_buffer, recv_size, stream);
        //merge
        int64_t total_recv_size = context.sum(recv_size);
        auto selected_indx = torch::empty({total_recv_size}, torch::kInt64).to(torch::kCUDA);
        auto selected_tensor = torch::empty({total_recv_size, shape_[1]}, options);
        int blockSize = 0;
        int numBlocks = 0;
        cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
                                           tensor_gather);
        tensor_gather<<<numBlocks, blockSize, 0, stream>>>(
            buffers_device, offset_device, offset_list_.size(),
            selected_indx.data_ptr<int64_t>(), total_recv_size, (char*)selected_tensor.data_ptr(),
            stride_in_bytes(0), access_book_device, 0, context.get_size());
        float *send_data_buffer;
        float *recv_data_buffer;
        int stride = shape_[1];
        CHECK(ncclMemAlloc((void **)&send_data_buffer, total_recv_size * element_size * shape_[1]));
        CHECK(ncclMemAlloc((void **)&recv_data_buffer, indices_length * element_size * shape_[1]));
        cudaMemcpy(send_data_buffer, selected_tensor.data_ptr(), total_recv_size * element_size * shape_[1], cudaMemcpyDeviceToDevice);
        context.send_data_to_all(send_data_buffer, recv_size, recv_data_buffer, comm_size, stride, stream);
        //scatter to res
        auto recv_data = torch::from_blob(recv_data_buffer, {indices_length, shape_[1]}, options);
        auto res = torch::empty({indices_length, shape_[1]}, options);
        auto res_indices = torch::empty({indices_length}, torch::kInt64).to(torch::kCUDA);
        cudaMemcpy(res_indices.data_ptr(), res_ptr.data().get(), indices_length * sizeof(int64_t), cudaMemcpyDeviceToDevice);
        res[res_indices] = recv_data;
        ncclGroupEnd();

        // int blockSize = 0;
        // int numBlocks = 0;
        // cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
        //                                    starrygl_tensor_gather);
        // // std::cout<<"LOG >>> "<<" numBlocks "<< numBlocks <<" blockSize
        // // "<<blockSize<<std::endl;
        // int ignore_access_book = 0;
        // if (current_device != device_) { ignore_access_book = 1; }
        // starrygl_tensor_gather<<<numBlocks, blockSize, 0, stream>>>(
        //     buffers_device, offset_device, offset_list_.size(),
        //     indices.data_ptr<int64_t>(), indices.numel(), (char*)res.data_ptr(),
        //     stride_in_bytes(0), access_book_device, ignore_access_book);
        // cudaCheckError();
        // //if(is_local){
        // auto dist_res = dist_tensor_gather(buffers_device, 
        //                         offset_device, 
        //                         offset_list_.size(),
        //                         indices.data_ptr<int64_t>(),
        //                         indices.numel(),
        //                         (char *)res.data_ptr(),
        //                         const int stride,
        //                         const int *access_book,
        //                         )
        // res[dist_res.second] = dist_res.first;
            
        
        return res;
    }
    void update(torch::Tensor &indices, torch::Tensor &values){

    }
    std::vector<int64_t> shape() const { return shape_; }

    int device() const { return device_; }

    int size(int dim) const
    {
        if (shape_.size() == 0) return 0;
        return shape_[dim];
    }

    int64_t stride(int dim) const
    {
        int64_t res = 1;
        for (int index = dim + 1; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }

    int64_t stride_in_bytes(int dim) const{
        return stride(dim) * element_size;
    }

    int64_t numel() const
    {
        int64_t res = 1;
        for (int index = 0; index < shape_.size(); index++) {
            res *= shape_[index];
        }
        return res;
    }
    std::vector<ShardTensorItem> share_ipc()
    {
        std::vector<ShardTensorItem> res;
        for (int index = 0; index < dev_ptrs_.size(); index++) {
            if (tensor_devices_[index] >= 0) {
                cudaSetDevice(tensor_devices_[index]);
                ShardTensorItem *item = new ShardTensorItem();
                item->device = tensor_devices_[index];
                item->shape = tensor_shapes_[index];
                item->element_size = element_size;
                cudaIpcGetMemHandle(&(item->mem_handle), dev_ptrs_[index]);
                res.push_back(*item);
            }
        }
        return res;
    }

    int device_count() const { return device_count_; }

    void unregister(torch::Tensor &cpu_tensor)
    {

        std::cout << "begin unregister" << std::endl;
        cudaHostUnregister((void *)cpu_tensor.data_ptr<float>());
        std::cout << "end unregister" << std::endl;
    }
    struct is_partition
    {
        int p;
        is_partition(int value) : p(value) {}
        __host__ __device__ bool operator()(int x) const {
            return get_partition(x) == p;
        }
    };
  private:
    std::vector<int64_t> offset_list_;
    std::vector<void *> dev_ptrs_;
    std::vector<int> tensor_devices_;
    std::vector<int> access_book;
    std::vector<int> access_rank;
    std::vector<std::vector<int>> tensor_shapes_;
    std::vector<int64_t> shape_;
    std::unordered_map<int, std::tuple<char **, int64_t *, int *>>
        device_pointers_map;
    int group_;
    int world_size_;
    int size_;
    int rank_;
    int machine_rank_;
    int device_;
    int device_count_;
    bool inited_;
    int element_size;
    DistContext context;
    
    std::tuple<char **, int64_t *, int *> get_device_pointers(int device)
    {
        auto iter = device_pointers_map.find(device);
        if (iter == device_pointers_map.end()) {
            char **buffers_device;
            int64_t *offset_device;
            int *access_book_device;

            // Copy buffers Device
            cudaMalloc((void ***)&buffers_device,
                       sizeof(float *) * device_count_);
            cudaMemcpy(buffers_device, &dev_ptrs_[0],
                       sizeof(float *) * dev_ptrs_.size(),
                       cudaMemcpyHostToDevice);
            cudaCheckError();

            // copy offset
            cudaMalloc((void **)&offset_device,
                       sizeof(int64_t) * offset_list_.size());
            cudaMemcpy(offset_device, &offset_list_[0],
                       sizeof(int64_t) * offset_list_.size(),
                       cudaMemcpyHostToDevice);
            cudaCheckError();

            cudaMalloc((void **)&access_book_device,
                       sizeof(int) * access_book.size());
            cudaMemcpy(access_book_device, &access_book[0],
                       sizeof(int) * access_book.size(),
                       cudaMemcpyHostToDevice);
            cudaCheckError();
            device_pointers_map.emplace(
                device, std::make_tuple(buffers_device, offset_device,
                                        access_book_device));
            iter = device_pointers_map.find(device);
        }
        return iter->second;
    }
    // void dist_tensor_gather(char **dev_ptrs, const int64_t *offsets,
    //                                  const int device_count,
    //                                  const int64_t *indices, int indice_length,
    //                                  char *res, const int stride,
    //                                  const int *access_book,
    //                                  const int ignore_access_book){
        
    //     auto counting_iter = thrust::make_counting_iterator(0);
    //     thrust::device_vector<int64_t> thrust_vec(indices.begin<int64_t>(), indices.end<int64_t>());
    //     thrust::device_vector<int64_t> comm_ptr(indice_length, 0);
    //     thrust::device_vector<int64_t> res_ptr(indice_length, 0);
    //     thrust::device_vector<int64_t> comm_ptr_size(context.get_world_size(), 0);
    //     auto comm_ptr_begin = 0;
    //     auto res_ptr_begin = 0;
    //     for(int i = 0; i < context.get_world_size ; i++){
    //         if(i/context.get_size() == context.get_rank()/context.get_size())continue;
    //         auto end = thrust::copy_if(thrust::device, thrust_vec.begin(), thrust_vec.end(), comm_ptr.begin()+comm_ptr_begin, is_partition(i));
    //         thrust::copy_if(thrust::device, counting_iter, counting_iter + indices_length, thrust_vec.begin(), res_ptr.begin()+res_ptr_begin, is_partition(i));
    //         comm_ptr_size[i] = end-(comm_ptr.begin()+comm_ptr_begin);
    //     }
    //     thrust::device_vector<int> send_counts(context.get_world_size(), 0);
    //     thrust::device_vector<int> send_ind;
    //     int counts = 0;
    //     context.get_padding_ind(comm_ptr, send_ind, 
    //                             comm_ptr_size.data().get(),
    //                             send_counts.data().get(),counts,at::cuda::getCurrentCUDAStream());
    //     int stride = stride(0);
    //     thrust::device_vector<float> send_data = thrust::device_vector(float)(context.get_world_size() * counts * stride, 0);
    //     int blockSize = 0;
    //     int numBlocks = 0;
    //     cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize,
    //                                        starrygl_tensor_gather);
    //     char **buffers_device;
    //     int64_t *offset_device;
    //     auto val = get_device_pointers(current_device);
    //     buffers_device = std::get<0>(val);
    //     offset_device = std::get<1>(val);
    //     starrygl_tensor_gather<<<numBlocks, blockSize, 0, stream>>>(
    //         buffers_device, offset_device, offset_list_.size(),
    //         send_ind.data_ptr<int64_t>(), indices.numel(), (char*)send_data.data().get()
    //         stride_in_bytes(0)
    //     );
    //     auto recv_data = context.data2all(
    //         send_data.data().get(), send_counts.data().get(), counts * stride, at::cuda::getCurrentCUDAStream()
    //     )
    //     res_ptr_tensor = th::from_blob(res_ptr.data().get(), {counts}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)，
    //                                 [](void *ptr){ cudaFree(ptr); });
    //     data_tensor = th::from_blob(recv_data.data().get(), {counts, stride}, 
    //                             torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA),
    //                            [](void *ptr) { cudaFree(ptr); });
    //     return make_pair(data_tensor, res_ptr);
    // }
};

void init_p2p(std::vector<int> devices)
{
    std::cout << "LOG>>> P2P Access Initilization" << std::endl;

    for (int i = 0; i < devices.size(); i++) {
        int src = devices[i];
        cudaSetDevice(src);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, src);

        // CUDA IPC is only supported on devices with unified addressing
        if (!prop.unifiedAddressing) {
            printf(
                "Device %d does not support unified addressing, skipping...\n",
                i);
            continue;
        }
        // This sample requires two processes accessing each device, so we need
        // to ensure exclusive or prohibited mode is not set
        if (prop.computeMode != cudaComputeModeDefault) {
            printf(
                "Device %d is in an unsupported compute mode for this sample\n",
                i);
            continue;
        }

        for (int j = i + 1; j < devices.size(); j++) {
            int dst = devices[j];
            int access_i_j = 0;
            int access_j_i = 0;
            cudaDeviceCanAccessPeer(&access_i_j, src, dst);
            cudaDeviceCanAccessPeer(&access_j_i, dst, src);
            if (access_i_j && access_j_i) {
                printf("Enable P2P Access Between %d <---> %d \n", src, dst);
                cudaSetDevice(src);
                cudaDeviceEnablePeerAccess(dst, 0);
                cudaCheckError();
                cudaSetDevice(dst);
                cudaDeviceEnablePeerAccess(src, 0);
                cudaCheckError();
            }
        }
    }
}
bool can_device_access_peer(int src_device_index, int dst_device_index)
{
    int access_i_j = 0, access_j_i = 0;
    cudaDeviceCanAccessPeer(&access_i_j, src_device_index, dst_device_index);
    cudaDeviceCanAccessPeer(&access_j_i, dst_device_index, src_device_index);
    return (access_i_j == 1) && (access_j_i == 1);
}

}  // namespace starrygl

    
    //TODO Remote update.    
    //     __global__ void tensor_update(char **dev_ptrs, const int64_t *offsets,
    //                                  const int device_count,
    //                                  const int64_t *indices, int indice_length,
    //                                  char *value, const int stride,
    //                                  const int *access_book,
    //                                  const int ignore_access_book,
    //                                  const int machine_rank,
    //                                  const int rank,
    //                                  const int size,
    //                                  const int world_size)
    // {

    //     //
    //     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //     unsigned int step = gridDim.x * blockDim.x;

    //     // each warp take charge of one-feature copy
    //     unsigned int warp_id = tid / WARP_SIZE;
    //     unsigned int warp_step = step / WARP_SIZE;

    //     unsigned int warp_start = warp_id;
    //     unsigned int thread_start = tid % WARP_SIZE;

    //     int64_t dev_index = 0;
    //     int64_t dev_offset = 0;
    //     char *dev_ptr;
    //     int64_t src_copy_start = 0;
    //     int64_t dst_copy_start = 0;

    //     unsigned int local_start = thread_start;
    //     while (warp_start < indice_length) {
    //         if(indices[warp_start] == -1){
    //             continue;
    //         }
    //         local_start = thread_start;
    //         dev_index = 1-is_shared(indices[warp_start]) ;
    //         dev_offset = get_loc(indices[warp_start]) //indices[warp_start] - offsets[dev_index];
    //         if(dev_index > 0){
    //             dev_index += get_partition(indices[warp_start])%size;
    //             dev_offset -= offsets[0];
    //         }
    //         if (dev_index != -1 && (ignore_access_book || access_book[dev_index] == 1)) {
    //             dev_ptr = dev_ptrs[dev_index];
    //             src_copy_start = dev_offset * stride;
    //             dst_copy_start = warp_start * stride;
    //             for (; local_start < stride; local_start += WARP_SIZE) {
    //                 dev_ptr[src_copy_start + local_start] = value[dst_copy_start + local_start];
    //             }
    //         }
    //         warp_start += warp_step;
    //     }
    // }


void register_cuda_feature(pybind11::module &m)
{
    m.def("init_p2p", &starrygl::init_p2p,
          py::call_guard<py::gil_scoped_release>());

    //m.def("can_device_access_peer", &::can_device_access_peer,
    //      py::call_guard<py::gil_scoped_release>());

    py::class_<starrygl::ShardTensorItem>(m, "ShardTensorItem")
        .def(py::init<>())
        .def("share_ipc", &starrygl::ShardTensorItem::share_ipc)
        .def("from_ipc", &starrygl::ShardTensorItem::from_ipc);

    py::class_<starrygl::ShardTensor>(m, "ShardTensor")
        .def(py::init<int,starrygl::DistContext>())
        .def("__getitem__", &starrygl::ShardTensor::operator[],
             py::call_guard<py::gil_scoped_release>())
        .def("unregister", &starrygl::ShardTensor::unregister,
             py::call_guard<py::gil_scoped_release>())
        .def("shape", &starrygl::ShardTensor::shape,
             py::call_guard<py::gil_scoped_release>())
        .def("numel", &starrygl::ShardTensor::numel,
             py::call_guard<py::gil_scoped_release>())
        .def("device", &starrygl::ShardTensor::device,
             py::call_guard<py::gil_scoped_release>())
        .def("stride", &starrygl::ShardTensor::stride,
             py::call_guard<py::gil_scoped_release>())
        .def("size", &starrygl::ShardTensor::size,
             py::call_guard<py::gil_scoped_release>())
        .def("device_count", &starrygl::ShardTensor::device_count,
             py::call_guard<py::gil_scoped_release>())
        .def("append",
             py::overload_cast<starrygl::ShardTensorItem>(
                 &starrygl::ShardTensor::append),
             py::call_guard<py::gil_scoped_release>())
        .def("share_ipc", &starrygl::ShardTensor::share_ipc,
             py::call_guard<py::gil_scoped_release>());
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     register_nccl_comm(m);
//     register_cuda_feature(m);
// }