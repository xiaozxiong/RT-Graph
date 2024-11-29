#include "common.h"
#include "timer.h"
#include "util.h"

#include <cxxopts.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define MAX_BLOCKS 65535U
#define MAX_BLOCK_SIZE 1024U

//* Sets must be sorted in ascending order for low collision.
__global__ void BlockBasedHI(tile_t A, tile_t B, const uint *elements, const uint *set_offsets,
                             uint *bucket, uint buckets, uint max_bucket_size, uint *counts);

__device__ int linear_search(uint *bucket, uint bucket_start, uint buckets, uint bucket_id,
                             uint offset_in_bucket, const uint *bucket_size, uint target);

void HashBasedIntersection(const Dataset &dataset, uint blocks, uint block_size, uint buckets,
                           uint max_bucket_size, std::vector<uint> &results);

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], "Hash based Set Intesection");
        options.add_options()("device", "chose GPU", cxxopts::value<uint>())(
            "dataset", "path of dataset", cxxopts::value<std::string>())(
            "buckets", "set the number of buckets", cxxopts::value<uint>())(
            "bucket_size", "set the max size of bucket", cxxopts::value<uint>())
            // ("blocks","number of blocks",cxxopts::value<uint>()->default_value("1"))
            ("threads", "threads per block",
             cxxopts::value<uint>()->default_value("512"))("h,help", "print help");
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        // arguments
        uint device_id = result["device"].as<uint>();
        std::string path = result["dataset"].as<std::string>();
        uint buckets = result["buckets"].as<uint>();
        uint max_bucket_size = result["bucket_size"].as<uint>();
        uint block_size = std::min(result["threads"].as<uint>(), MAX_BLOCK_SIZE);

        // query device attributes
        uint device_count;
        cuDeviceGetCount((int *)&device_count);
        if (device_id >= device_count) {
            std::cerr << "Device id (" << device_id << ") is larger than device count ("
                      << device_count << ")" << std::endl;
            exit(1);
        }
        cudaSetDevice(device_id);
        // uint max_shared_memory_per_block;
        // cuDeviceGetAttribute((int*)(&max_shared_memory_per_block),CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,device_id);
        // uint max_thread_per_block;
        // cuDeviceGetAttribute((int*)&max_thread_per_block,CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,device_id);
        // std::cout<<"Max shared memory per block = "<<max_shared_memory_per_block<<", max thread
        // per block = "<<max_thread_per_block<<std::endl;
        // TODO: execution
        //* read dataset
        Dataset dataset;
        ReadData(path, dataset);
        PrintDataInfo(dataset);
        //* run and check
        uint blocks = std::min(dataset.num_of_sets_a, MAX_BLOCKS); //* a block for a set A
        std::vector<uint> count_results;
        HashBasedIntersection(dataset, blocks, block_size, buckets, max_bucket_size, count_results);
        Check(dataset, count_results);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

void HashBasedIntersection(const Dataset &dataset, uint blocks, uint block_size, uint buckets,
                           uint max_bucket_size, std::vector<uint> &results) {
    tile_t A = {0, dataset.num_of_sets_a, dataset.num_of_sets_a}; // first
    tile_t B = {dataset.num_of_sets_a, dataset.num_of_sets_a + dataset.num_of_sets_b,
                dataset.num_of_sets_b}; // last
    size_t combinations = (size_t)A.size * B.size;
    std::cout << "blocks = " << blocks << ", block size = " << block_size << std::endl;
    std::cout << "combinations = " << combinations << std::endl;

    uint *d_elements, *d_set_offsets, *d_counts, *d_bucket;
    CUDA_CHECK(cudaMalloc((void **)&d_elements, sizeof(uint) * dataset.elements.size()));
    CUDA_CHECK(cudaMemcpy(d_elements, dataset.elements.data(),
                          sizeof(uint) * dataset.elements.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_set_offsets, sizeof(uint) * dataset.set_offsets.size()));
    CUDA_CHECK(cudaMemcpy(d_set_offsets, dataset.set_offsets.data(),
                          sizeof(uint) * dataset.set_offsets.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_counts, sizeof(uint) * combinations));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(uint) * combinations));
    size_t total_bucket_size = blocks * buckets * max_bucket_size; //
    CUDA_CHECK(cudaMalloc((void **)&d_bucket, sizeof(uint) * total_bucket_size));
    uint shared_memory_size = sizeof(uint) * buckets; // count size of each bucket in a block

    warm_up_gpu<<<blocks, block_size>>>();
    cudaDeviceSynchronize();

    GPUTimer gpu_timer;
    gpu_timer.StartTiming();
    BlockBasedHI<<<blocks, block_size, shared_memory_size>>>(
        A, B, d_elements, d_set_offsets, d_bucket, buckets, max_bucket_size, d_counts);
    // CUDA_SYNC_CHECK();
    gpu_timer.StopTiming();

    std::cout << "Hash based Set Intertsection: kernel time = " << gpu_timer.GetElapsedTime()
              << " ms\n";

    results.resize(combinations);
    CUDA_CHECK(
        cudaMemcpy(results.data(), d_counts, sizeof(uint) * combinations, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_bucket));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_set_offsets));
    CUDA_CHECK(cudaFree(d_elements));
}
// Create a bucket for each set A
__global__ void BlockBasedHI(tile_t A, tile_t B, const uint *elements, const uint *set_offsets,
                             uint *bucket, uint buckets, uint max_bucket_size, uint *counts) {

    extern __shared__ uint bucket_size[];
    uint bucket_start = blockIdx.x * (max_bucket_size * buckets);

    for (int a_id = A.start + blockIdx.x; a_id < A.end; a_id += gridDim.x) {
        uint a_set_start = set_offsets[a_id];
        uint a_set_end = set_offsets[a_id + 1];
        uint count_offset = (a_id - A.start) * B.size;
        for (int i = threadIdx.x; i < buckets; i += blockDim.x)
            bucket_size[i] = 0;
        __syncthreads();
        // hash shorter set, coalesce memory access, minimize collision
        for (int i = a_set_start + threadIdx.x; i < a_set_end; i += blockDim.x) {
            uint e = elements[i];
            uint bucket_id = e % buckets;
            uint index = atomicAdd(bucket_size + bucket_id, 1U);
            bucket[bucket_start + (index * buckets + bucket_id)] = e;
        }
        __syncthreads();
        // probe larger set
        uint b_set_id = B.start;
        uint id_in_set = threadIdx.x;
        bool flag = true;
        while (b_set_id < B.end) {
            uint tmp_size = set_offsets[b_set_id + 1] - set_offsets[b_set_id];
            while (id_in_set >= tmp_size) {
                id_in_set -= tmp_size;
                b_set_id += 1;
                if (b_set_id >= B.end) {
                    flag = false;
                    break;
                }
                tmp_size = set_offsets[b_set_id + 1] - set_offsets[b_set_id];
            }
            if (!flag)
                break;
            uint target = elements[set_offsets[b_set_id] + id_in_set];
            uint bucket_id = target % buckets;
            uint count =
                linear_search(bucket, bucket_start, buckets, bucket_id, 0, bucket_size, target);
            if (count > 0)
                atomicAdd(counts + count_offset + (b_set_id - B.start), 1U);
            id_in_set += blockDim.x;
        }
    }
}

__device__ int linear_search(uint *bucket, uint bucket_start, uint buckets, uint bucket_id,
                             uint offset_in_bucket, const uint *bucket_size, uint target) {
    uint len = bucket_size[offset_in_bucket + bucket_id]; // size of bucket
    uint index = bucket_start + bucket_id;                //
    for (int i = 0; i < len; i += 1) {
        uint e = bucket[index];
        if (e == target)
            return 1;
        else
            index += buckets;
    }
    return 0;
}
