#include "common.h"
#include "timer.h"
#include "util.h"

#include <cxxopts.hpp>
#include <iostream>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#define MAX_BLOCKS 65535U
#define MAX_BLOCK_SIZE 1024U

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

__forceinline__ __device__ int BinarySearchDiagonal(uint *a, uint a_size, uint *b, uint b_size,
                                                    uint diag);

__device__ uint SerialIntersection(uint *a, uint a_begin, uint a_end, uint *b, uint b_begin,
                                   uint b_end, uint vt);

__global__ void FindDiagonals(tile_t A, tile_t B, uint *elements, uint *set_offsets,
                              uint *global_diagonals, uint *counts);

__global__ void IntersectionKernel(tile_t A, tile_t B, uint *elements, uint *set_offsets,
                                   uint *global_diagonals, uint *counts);

void IntersectPathBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
                                    std::vector<uint> &results);

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], "Intersect Path based Set Intesection");
        options.add_options()("device", "chose GPU", cxxopts::value<uint>()->default_value("0"))(
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
        // uint device_count;
        // cuDeviceGetCount((int*)&device_count);
        // if(device_id>=device_count){
        //     std::cerr<<"Device id ("<<device_id<<") is larger than device count
        //     ("<<device_count<<")"<<std::endl; exit(1);
        // }
        //* set device
        cudaSetDevice(device_id);
        int multiprocessorCount;
        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, device_id);
        // TODO: execution
        //* read dataset
        Dataset dataset;
        ReadData(path, dataset);
        PrintDataInfo(dataset);
        //* run and check
        uint total_length = (dataset.set_offsets[1] - dataset.set_offsets[0]) +
                            (dataset.set_offsets[dataset.num_of_sets] -
                             dataset.set_offsets[dataset.num_of_sets - 1]);
        uint blocks = std::min((total_length + block_size - 1) / block_size,
                               MAX_BLOCKS); //* a grid for a combination
        std::vector<uint> count_results;
        IntersectPathBasedIntersection(dataset, blocks, block_size, count_results);
        Check(dataset, count_results);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}

void IntersectPathBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
                                    std::vector<uint> &results) {
    tile_t A = {0, dataset.num_of_sets_a, dataset.num_of_sets_a};                   // first
    tile_t B = {dataset.num_of_sets_a, dataset.num_of_sets, dataset.num_of_sets_b}; // last
    size_t combinations = (size_t)A.size * B.size;
    std::cout << "blocks = " << blocks << ", block size = " << block_size << std::endl;
    std::cout << "combinations = " << combinations << std::endl;

    uint *d_elements, *d_set_offsets;
    CUDA_CHECK(cudaMalloc((void **)&d_elements, sizeof(uint) * dataset.elements.size()));
    CUDA_CHECK(cudaMemcpy(d_elements, dataset.elements.data(),
                          sizeof(uint) * dataset.elements.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_set_offsets, sizeof(uint) * dataset.set_offsets.size()));
    CUDA_CHECK(cudaMemcpy(d_set_offsets, dataset.set_offsets.data(),
                          sizeof(uint) * dataset.set_offsets.size(), cudaMemcpyHostToDevice));

    uint *d_global_diagonals, *d_counts;
    CUDA_CHECK(
        cudaMalloc((void **)&d_global_diagonals, sizeof(uint) * 2 * (blocks + 1) * combinations));
    CUDA_CHECK(cudaMemset(d_global_diagonals, 0, sizeof(uint) * 2 * (blocks + 1) * combinations));
    CUDA_CHECK(cudaMalloc((void **)&d_counts, sizeof(uint) * combinations));
    CUDA_CHECK(cudaMemset(d_counts, 0, sizeof(uint) * combinations));

    warm_up_gpu<<<blocks, block_size>>>();
    cudaDeviceSynchronize();

    GPUTimer gpu_timer;
    gpu_timer.StartTiming();
    FindDiagonals<<<blocks, 32>>>(A, B, d_elements, d_set_offsets, d_global_diagonals, d_counts);
    IntersectionKernel<<<blocks, block_size>>>(A, B, d_elements, d_set_offsets, d_global_diagonals,
                                               d_counts);
    gpu_timer.StopTiming();
    std::cout << "Intersect Path based Set Intertsection: kernel time = "
              << gpu_timer.GetElapsedTime() << " ms\n";

    results.resize(combinations);
    CUDA_CHECK(
        cudaMemcpy(results.data(), d_counts, sizeof(uint) * combinations, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_global_diagonals));
    CUDA_CHECK(cudaFree(d_set_offsets));
    CUDA_CHECK(cudaFree(d_elements));
}

// a grid for a intersection, because set size is large
__global__ void IntersectionKernel(tile_t A, tile_t B, uint *elements, uint *set_offsets,
                                   uint *global_diagonals, uint *counts) {
    for (int a = A.start; a < A.end; a += 1) {
        uint *a_set = elements + set_offsets[a];
        for (int b = B.start; b < B.end; b += 1) {
            uint *b_set = elements + set_offsets[b];

            uint offset = (a - A.start) * B.size + (b - B.start); // index of combinations
            uint *diagonals = global_diagonals + (2 * (gridDim.x + 1)) * offset; // current matrix

            uint a_start = diagonals[blockIdx.x];
            uint a_end = diagonals[blockIdx.x + 1];
            uint a_size = a_end - a_start;
            uint b_start = diagonals[(gridDim.x + 1) + blockIdx.x];
            uint b_end = diagonals[(gridDim.x + 1) + blockIdx.x + 1];
            uint b_size = b_end - b_start;

            uint vt = ((a_size + b_size) / blockDim.x) + 1;
            uint diag = threadIdx.x * vt;
            int mp = BinarySearchDiagonal(a_set + a_start, a_size, b_set + b_start, b_size, diag);
            uint intersection = SerialIntersection(a_set + a_start, mp, a_size, b_set + b_start,
                                                   diag - mp, b_size, vt);
            atomicAdd(counts + offset, intersection);
        }
    }
}

// https://github.com/ogreen/MergePathGPU/blob/master/main.cu
// get the diagonal of each block
__global__ void FindDiagonals(tile_t A, tile_t B, uint *elements, uint *set_offsets,
                              uint *global_diagonals, uint *counts) {
    for (int a = A.start; a < A.end; a += 1) {
        uint a_set_size = set_offsets[a + 1] - set_offsets[a];
        uint *a_set = elements + set_offsets[a];
        for (int b = B.start; b < B.end; b += 1) {
            uint b_set_size = set_offsets[b + 1] - set_offsets[b];
            uint *b_set = elements + set_offsets[b];

            uint count_offset = (a - A.start) * B.size + (b - B.start); // index of combinations
            // get the start and end position of diagonal of this block
            uint *diagonals =
                global_diagonals + (2 * (gridDim.x + 1)) * count_offset; // (start..., end...)
            // the start index of intersect path of this block
            uint combined_index = (uint64_t)blockIdx.x *
                                  ((uint64_t)a_set_size + (uint64_t)b_set_size) /
                                  (uint64_t)gridDim.x;

            __shared__ int x_top, y_top, x_bottom, y_bottom, found;
            __shared__ uint one_or_zero[32];
            __shared__ uint increment;

            if (threadIdx.x == 0)
                increment = 0;
            __syncthreads();

            uint thread_offset = threadIdx.x - 16;

            x_top = MIN(combined_index, a_set_size);
            y_top = combined_index > a_set_size ? combined_index - a_set_size : 0;
            x_bottom = y_top;
            y_bottom = x_top;
            found = 0;
            // Search the diagonal
            int cnt = 0;
            while (!found) {
                int current_x = x_top - ((x_top - x_bottom) >> 1) - thread_offset;
                int current_y = y_top + ((y_bottom - y_top) >> 1) + thread_offset;

                if (current_x >= a_set_size || current_y < 0)
                    one_or_zero[threadIdx.x] = 0;
                else if (current_y >= b_set_size || current_x < 1)
                    one_or_zero[threadIdx.x] = 1;
                else {
                    one_or_zero[threadIdx.x] = (a_set[current_x - 1] <= b_set[current_y]) ? 1 : 0;
                    // count argument
                    if (a_set[current_x - 1] == b_set[current_y] && increment == 0) {
                        atomicAdd(counts + count_offset, 1U);
                        atomicAdd(&increment, 1U);
                    }
                }
                __syncthreads();
                if (threadIdx.x > 0 && (one_or_zero[threadIdx.x] != one_or_zero[threadIdx.x - 1]) &&
                    current_x >= 0 && current_y >= 0) {
                    found = 1;
                    diagonals[blockIdx.x] = current_x;
                    diagonals[blockIdx.x + gridDim.x + 1] = current_y;
                }
                __syncthreads();
                if (threadIdx.x == 16) {
                    if (one_or_zero[31] != 0) {
                        x_bottom = current_x;
                        y_bottom = current_y;
                    } else {
                        x_top = current_x;
                        y_top = current_y;
                    }
                }
                __syncthreads();

                cnt += 1;
            }

            if (threadIdx.x == 0 && blockIdx.x == 0) {
                diagonals[0] = 0;
                diagonals[gridDim.x + 1] = 0;
                diagonals[gridDim.x] = a_set_size;
                diagonals[gridDim.x + gridDim.x + 1] = b_set_size;
            }
            one_or_zero[threadIdx.x] = 0;
            __syncthreads();
        }
    }
}

__forceinline__ __device__ int BinarySearchDiagonal(uint *a, uint a_size, uint *b, uint b_size,
                                                    uint diag) {
    int begin = MAX(0, (int)(diag - b_size)); //!
    int end = MIN(diag, a_size);
    while (begin < end) {
        int mid = (begin + end) / 2;
        if (a[mid] < b[diag - 1 - mid])
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}

__device__ uint SerialIntersection(uint *a, uint a_begin, uint a_end, uint *b, uint b_begin,
                                   uint b_end, uint vt) {
    uint count = 0;
    for (int i = 0; i < vt; i += 1) {
        bool flag = false;
        if (a_begin >= a_end)
            flag = false;
        else if (b_begin >= b_end)
            flag = true;
        else {
            if (a[a_begin] < b[b_begin])
                flag = true;
            if (a[a_begin] == b[b_begin])
                count += 1;
        }
        if (flag)
            a_begin += 1;
        else
            b_begin += 1;
    }
    return count;
}