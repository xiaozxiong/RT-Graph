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

__global__ void BlockBasedOBS(tile_t A, tile_t B, const uint *elements, const uint *set_offsets,
                              uint *counts);

__global__ void BlockBasedOBS_1(tile_t A, tile_t B, const uint *elements, const uint *set_offsets,
                                uint *counts);

void BinarySearchBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
                                   std::vector<uint> &results);

// TODO: RT -> binary search
void ThreadBinarySearch(const Dataset &dataset, uint blocks, uint block_size,
                        std::vector<uint> &results);

__global__ void ThreadBasedBS(tile_t B, int a_nums, const uint *a_sets, const uint *elements,
                              const uint *set_offsets, uint *counts);

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], "Binary Search based Set Intesection");
        options.add_options()("device", "chose GPU", cxxopts::value<uint>())(
            "dataset", "path of dataset", cxxopts::value<std::string>())
            // ("blocks","number of blocks",cxxopts::value<uint>()->default_value("1"))
            ("threads", "threads per block",
             cxxopts::value<uint>()->default_value("512"))("h,help", "print help");
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        //* arguments
        uint device_id = result["device"].as<uint>();
        std::string path = result["dataset"].as<std::string>();
        uint block_size = std::min(result["threads"].as<uint>(), MAX_BLOCK_SIZE);
        //* device query
        uint device_count;
        cuDeviceGetCount((int *)&device_count);
        if (device_id >= device_count) {
            std::cerr << "Device id (" << device_id << ") is larger than device count ("
                      << device_count << ")" << std::endl;
            exit(1);
        }
        //* set device
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
        BinarySearchBasedIntersection(dataset, blocks, block_size, count_results);

        // ThreadBinarySearch(dataset,blocks,block_size,count_results); // for test
        Check(dataset, count_results);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}

void BinarySearchBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
                                   std::vector<uint> &results) {
    tile_t A = {0, dataset.num_of_sets_a, dataset.num_of_sets_a}; // first
    tile_t B = {dataset.num_of_sets_a, dataset.num_of_sets_a + dataset.num_of_sets_b,
                dataset.num_of_sets_b}; // last
    size_t combinations = (size_t)A.size * B.size;
    std::cout << "blocks = " << blocks << ", block size = " << block_size << std::endl;
    std::cout << "combinations = " << combinations << std::endl;

    uint *d_elements, *d_set_offsets, *d_counts;
    CUDA_CHECK(cudaMalloc((void **)&d_elements, sizeof(uint) * dataset.elements.size()));
    CUDA_CHECK(cudaMemcpy(d_elements, dataset.elements.data(),
                          sizeof(uint) * dataset.elements.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_set_offsets, sizeof(uint) * dataset.set_offsets.size()));
    CUDA_CHECK(cudaMemcpy(d_set_offsets, dataset.set_offsets.data(),
                          sizeof(uint) * dataset.set_offsets.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_counts, sizeof(uint) * combinations));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(uint) * combinations));

    warm_up_gpu<<<blocks, block_size>>>();
    cudaDeviceSynchronize();

    // std::cout<<"Start binary search kernel\n";
    uint shared_memory_size = sizeof(uint) * block_size;

    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    BlockBasedOBS<<<blocks, block_size, shared_memory_size>>>(A, B, d_elements, d_set_offsets,
                                                              d_counts);
    // BlockBasedOBS_1<<<blocks,block_size,shared_memory_size>>>(A,B,d_elements,d_set_offsets,d_counts);
    // // for test
    cudaDeviceSynchronize();
    cpu_timer.StopTiming();
    std::cout << "Binary Search based Set Intertsection: kernel time = "
              << cpu_timer.GetElapsedTime() << " ms\n";

    results.resize(combinations);
    CUDA_CHECK(
        cudaMemcpy(results.data(), d_counts, sizeof(uint) * combinations, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_set_offsets));
    CUDA_CHECK(cudaFree(d_elements));
}

// skip list in shared memory can also be useful
__global__ void BlockBasedOBS(tile_t A, tile_t B, const uint *elements, const uint *set_offsets,
                              uint *counts) {

    extern __shared__ uint shared_array[]; // = block size
    // for each set in tile A
    for (int a_id = A.start + blockIdx.x; a_id < A.end; a_id += gridDim.x) {
        uint a_set_start = set_offsets[a_id];
        uint a_set_end = set_offsets[a_id + 1];
        uint a_set_size = a_set_end - a_set_start;

        uint count_offset = (a_id - A.start) * B.size;
        // place top levels of larger set into shared memory
        uint eid = a_set_start + (threadIdx.x * a_set_size / blockDim.x);
        shared_array[threadIdx.x] = elements[eid];
        __syncthreads();
        // enumerate set b
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
            // lookup
            uint target = elements[set_offsets[b_set_id] + id_in_set];
            uint count = 0;
            // phase1: search in shared memory
            int left = 0, right = blockDim.x - 1;
            bool finished = false;
            while (left <= right) {
                int mid = (left + right) >> 1;
                int x = shared_array[mid];
                if (x == target) {
                    count += 1;
                    finished = true;
                    break;
                } else if (x < target)
                    left = mid + 1;
                else
                    right = mid - 1;
            }
            // phase2: search in global memory
            if (!finished && right >= 0) {
                left = right * a_set_size / blockDim.x;
                right = (right + 1) * a_set_size / blockDim.x;
                while (left <= right) {
                    int mid = (left + right) >> 1;
                    int x = elements[a_set_start + mid];
                    if (x == target) {
                        count += 1;
                        break;
                    } else if (x < target)
                        left = mid + 1;
                    else
                        right = mid - 1;
                }
            }
            if (count > 0U)
                atomicAdd(counts + count_offset + (b_set_id - B.start), 1U);
            id_in_set += blockDim.x;
        }
    }
}

__global__ void BlockBasedOBS_1(tile_t A, tile_t B, const uint *elements, const uint *set_offsets,
                                uint *counts) {
    extern unsigned int __shared__ s[]; // block size
    unsigned int *cache = s;

    for (unsigned int a = blockIdx.x + A.start; a < A.end; a += gridDim.x) {
        unsigned int aStart = set_offsets[a]; // set offset
        unsigned int aEnd = set_offsets[a + 1];
        unsigned int aSize = aEnd - aStart;
        unsigned int count_offset = a * (B.end - B.start);

        // cache first levels of the binary tree of the larger set
        cache[threadIdx.x] = elements[aStart + (threadIdx.x * aSize / blockDim.x)];
        __syncthreads();

        for (unsigned int b = B.start; b < B.end; b++) {

            unsigned int bStart = set_offsets[b];
            unsigned int bEnd = set_offsets[b + 1];

            unsigned int count = 0;

            // search smaller set
            for (unsigned int i = threadIdx.x + bStart; i < bEnd; i += blockDim.x) {
                unsigned int x = elements[i];
                unsigned int y;

                // phase 1: cache
                int bottom = 0;
                int top = blockDim.x;
                int mid;
                while (top > bottom + 1) {
                    mid = (top + bottom) / 2;
                    y = cache[mid];
                    if (x == y) {
                        count++;
                        bottom = top + blockDim.x;
                    }
                    if (x < y)
                        top = mid;
                    if (x > y)
                        bottom = mid;
                }

                // phase 2
                bottom = bottom * aSize / blockDim.x;
                top = top * aSize / blockDim.x - 1;
                while (top >= bottom) {
                    mid = (top + bottom) / 2;
                    y = elements[aStart + mid];
                    if (x == y)
                        count++;
                    if (x <= y)
                        top = mid - 1;
                    if (x >= y)
                        bottom = mid + 1;
                }
            }
            atomicAdd(counts + count_offset + (b - B.start), count);
            __syncthreads();
        }
    }
}

//! ==
void ThreadBinarySearch(const Dataset &dataset, uint blocks, uint block_size,
                        std::vector<uint> &results) {
    printf("-----> thread binary search\n");
    tile_t A = {0, dataset.num_of_sets_a, dataset.num_of_sets_a}; // first
    tile_t B = {dataset.num_of_sets_a, dataset.num_of_sets_a + dataset.num_of_sets_b,
                dataset.num_of_sets_b}; // last
    size_t combinations = (size_t)A.size * B.size;
    std::cout << "combinations = " << combinations << std::endl;
    uint a_nums = dataset.set_offsets[A.end];
    std::cout << "total number of elements in set A = " << a_nums << std::endl;

    //* record set id of each elements in set A
    uint *a_sets = (uint *)malloc(sizeof(int) * a_nums);
    for (int i = A.start; i < A.end; i++) {
        int start = dataset.set_offsets[i];
        int end = dataset.set_offsets[i + 1];
        for (int j = start; j < end; j++)
            a_sets[j] = i;
    }
    uint *d_a_sets;
    CUDA_CHECK(cudaMalloc((void **)&d_a_sets, sizeof(uint) * a_nums));
    CUDA_CHECK(cudaMemcpy(d_a_sets, a_sets, sizeof(uint) * a_nums, cudaMemcpyHostToDevice));

    uint *d_elements, *d_set_offsets, *d_counts;
    CUDA_CHECK(cudaMalloc((void **)&d_elements, sizeof(uint) * dataset.elements.size()));
    CUDA_CHECK(cudaMemcpy(d_elements, dataset.elements.data(),
                          sizeof(uint) * dataset.elements.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_set_offsets, sizeof(uint) * dataset.set_offsets.size()));
    CUDA_CHECK(cudaMemcpy(d_set_offsets, dataset.set_offsets.data(),
                          sizeof(uint) * dataset.set_offsets.size(), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void **)&d_counts, sizeof(uint) * combinations));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(uint) * combinations));

    // a_nums /= 1000;
    printf("a_nums = %d\n", a_nums);
    int new_block_size = 256;
    int new_blocks = (a_nums + new_block_size - 1) / new_block_size;

    warm_up_gpu<<<new_blocks, new_block_size>>>();
    cudaDeviceSynchronize();

    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    ThreadBasedBS<<<new_blocks, new_block_size>>>(B, a_nums, d_a_sets, d_elements, d_set_offsets,
                                                  d_counts);
    cudaDeviceSynchronize();
    cpu_timer.StopTiming();
    std::cout << "Binary Search based Set Intertsection: kernel time = "
              << cpu_timer.GetElapsedTime() << " ms\n";

    results.resize(combinations);
    CUDA_CHECK(
        cudaMemcpy(results.data(), d_counts, sizeof(uint) * combinations, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_set_offsets));
    CUDA_CHECK(cudaFree(d_elements));

    CUDA_CHECK(cudaFree(d_a_sets));
    free(a_sets);
}

//* binary search mimicing RT
// a_nums: the total number of element in set A
__global__ void ThreadBasedBS(tile_t B, int a_nums, const uint *a_sets, const uint *elements,
                              const uint *set_offsets, uint *counts) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // int a = tid;
    for (int a = tid; a < a_nums; a += gridDim.x * blockDim.x) {
        // if(a < a_nums){
        int a_set_id = a_sets[a];
        int target = elements[a]; //
        int count_offset = a_set_id * (B.end - B.start);
        for (int b = B.start; b < B.end; b++) {
            int b_set_start = set_offsets[b];
            int b_set_end = set_offsets[b + 1];
            uint count = 0;
            int l = b_set_start, r = b_set_end - 1;
            while (l <= r) {
                int mid = l + (r - l) / 2;
                if (elements[mid] == target) {
                    count += 1;
                    break;
                } else if (elements[mid] > target)
                    r = mid - 1;
                else
                    l = mid + 1;
            }
            //  printf("access = %d\n", access);
            if (counts > 0)
                atomicAdd(counts + count_offset + (b - B.start), count);
        }
    }
}