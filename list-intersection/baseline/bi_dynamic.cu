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

using word_t = unsigned long long;

#define WORD_BITS (sizeof(word_t) * CHAR_BIT)
#define MAX_BLOCKS 65535U
#define MAX_BLOCK_SIZE 1024U

__global__ void IntersectionKernel(tile_t A, tile_t B, const uint *elements,
                                   const uint *set_offsets, word_t *bitmaps, uint words,
                                   uint *counts);

void BitmapDynamicBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
                                    std::vector<uint> &results);

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], "Bitmap(dynamic) based Set Intesection");
        // clang-format off
        options.add_options()
            ("device","chose GPU",cxxopts::value<uint>()->default_value("0"))
            ("dataset","path of dataset",cxxopts::value<std::string>())
            ("blocks","number of blocks",cxxopts::value<uint>()->default_value("0"))
            ("threads","threads per block",cxxopts::value<uint>()->default_value("512"))
            ("help","print help")
        ;
        // clang-format on
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        //* arguments
        uint device_id = result["device"].as<uint>();
        std::string path = result["dataset"].as<std::string>();
        uint block_size = std::min(result["threads"].as<uint>(), MAX_BLOCK_SIZE);
        uint blocks = result["blocks"].as<uint>();
        //* device query
        // uint device_count;
        // cuDeviceGetCount((int*)&device_count);
        // if(device_id>=device_count){
        //     std::cerr<<"Device id ("<<device_id<<") is larger than device count
        //     ("<<device_count<<")"<<std::endl; exit(1);
        // }
        //* set device
        cudaSetDevice(device_id);
        // TODO: execution
        //* read dataset
        Dataset dataset;
        ReadData(path, dataset);
        PrintDataInfo(dataset);
        //* run and check
        if (blocks == 0U)
            blocks = std::min(dataset.num_of_sets_a, MAX_BLOCKS); //* a block for a set A
        std::vector<uint> count_results;
        BitmapDynamicBasedIntersection(dataset, blocks, block_size, count_results);
        Check(dataset, count_results);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

void BitmapDynamicBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
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

    //* setup bitmap
    uint words = (dataset.max_element + WORD_BITS - 1) / WORD_BITS; // words of a set
    word_t *d_bitmaps;
    std::cout << "memory size of bitmap = "
              << 1.0 * sizeof(word_t) * words * blocks / 1024 / 1024 / 1024 << " GB\n";
    CUDA_CHECK(cudaMalloc((void **)&d_bitmaps, sizeof(word_t) * words * blocks)); //
    CUDA_CHECK(cudaMemset(d_bitmaps, 0, sizeof(word_t) * words * blocks));        //

    warm_up_gpu<<<blocks, block_size>>>();
    cudaDeviceSynchronize();

    GPUTimer gpu_timer;
    gpu_timer.StartTiming();
    IntersectionKernel<<<blocks, block_size>>>(A, B, d_elements, d_set_offsets, d_bitmaps, words,
                                               d_counts);
    gpu_timer.StopTiming();
    std::cout << "Bitmap(dynamic) based Set Intertsection: kernel time = "
              << gpu_timer.GetElapsedTime() << " ms\n";

    results.resize(combinations);
    CUDA_CHECK(
        cudaMemcpy(results.data(), d_counts, sizeof(uint) * combinations, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_bitmaps));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_set_offsets));
    CUDA_CHECK(cudaFree(d_elements));
}

// only need a bitmap
__global__ void IntersectionKernel(tile_t A, tile_t B, const uint *elements,
                                   const uint *set_offsets, word_t *bitmaps, uint words,
                                   uint *counts) {
    for (int a = A.start + blockIdx.x; a < A.end; a += gridDim.x) {
        uint a_set_start = set_offsets[a];
        uint a_set_end = set_offsets[a + 1];
        uint count_offset = (a - A.start) * B.size;
        // create bitmap of set a
        for (int i = a_set_start + threadIdx.x; i < a_set_end; i += blockDim.x) {
            uint a_e = elements[i];
            atomicOr(bitmaps + (blockIdx.x * words) + (a_e / WORD_BITS),
                     (word_t)1 << (a_e % WORD_BITS));
        }
        __syncthreads();
        for (int b = B.start; b < B.end; b += 1) { // iterate over all sets B
            uint b_set_start = set_offsets[b];
            uint b_set_end = set_offsets[b + 1];
            uint count = 0;
            for (int i = b_set_start + threadIdx.x; i < b_set_end; i += blockDim.x) {
                uint b_e = elements[i];
                // if the bit is set
                if (bitmaps[(blockIdx.x * words) + (b_e / WORD_BITS)] &
                    ((word_t)1 << (b_e % WORD_BITS)))
                    count += 1;
            }
            atomicAdd(counts + count_offset + (b - B.start), count);
        }
        __syncthreads();
        // clear bitmap
        for (int i = a_set_start + threadIdx.x; i < a_set_end; i += blockDim.x) {
            uint a_e = elements[i];
            bitmaps[(blockIdx.x * words) + (a_e / WORD_BITS)] = 0;
        }
        __syncthreads();
    }
}