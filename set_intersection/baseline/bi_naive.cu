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

__global__ void ConstructBitmapKernel(const uint *elements, const uint *set_offsets,
                                      uint num_of_sets, word_t *bitmaps, uint words);

__global__ void IntersectionKernel(tile_t A, tile_t B, const word_t *bitmaps, uint words,
                                   uint *counts);

void BitmapNaiveBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
                                  std::vector<uint> &results);

__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], "Bitmap(naive) based Set Intesection");
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
        // TODO: execution
        //* read dataset
        Dataset dataset;
        ReadData(path, dataset);
        PrintDataInfo(dataset);
        uint total_length = (dataset.set_offsets[1] - dataset.set_offsets[0]) +
                            (dataset.set_offsets[dataset.num_of_sets] -
                             dataset.set_offsets[dataset.num_of_sets - 1]);
        uint blocks = std::min((total_length + block_size - 1) / block_size,
                               MAX_BLOCKS); //* a thread for a intersection,
        //* run and check
        std::vector<uint> count_results;
        BitmapNaiveBasedIntersection(dataset, blocks, block_size, count_results);
        Check(dataset, count_results);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}

void BitmapNaiveBasedIntersection(const Dataset &dataset, uint blocks, uint block_size,
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
    std::cout << "memory size of bitmap = "
              << 1.0 * sizeof(word_t) * words * dataset.num_of_sets / 1024 / 1024 / 1024 << " GB\n";
    word_t *d_bitmaps;
    CUDA_CHECK(cudaMalloc((void **)&d_bitmaps, sizeof(word_t) * words * dataset.num_of_sets));
    CUDA_CHECK(cudaMemset(d_bitmaps, 0, sizeof(word_t) * words * dataset.num_of_sets));

    warm_up_gpu<<<blocks, block_size>>>();
    cudaDeviceSynchronize();

    GPUTimer *gpu_timer = new GPUTimer();
    gpu_timer->StartTiming();
    ConstructBitmapKernel<<<blocks, block_size>>>(d_elements, d_set_offsets, dataset.num_of_sets,
                                                  d_bitmaps, words);
    gpu_timer->StopTiming();
    double construction_time = gpu_timer->GetElapsedTime();
    delete gpu_timer;

    gpu_timer = new GPUTimer();
    gpu_timer->StartTiming();
    IntersectionKernel<<<blocks, block_size>>>(A, B, d_bitmaps, words, d_counts);
    gpu_timer->StopTiming();
    double counting_time = gpu_timer->GetElapsedTime();
    delete gpu_timer;
    // gpu_timer.StopTiming();
    // std::cout<<"Bitmap(naive) based Set Intertsection: kernel time =
    // "<<gpu_timer.GetElapsedTime()<<" ms\n";
    std::cout << "Bitmap(naive) based Set Intertsection:\nconstruction time = " << construction_time
              << " ms";
    std::cout << ", counting time = " << counting_time << " ms ";
    std::cout << ", kernel time = " << construction_time + counting_time << " ms\n";

    results.resize(combinations);
    CUDA_CHECK(
        cudaMemcpy(results.data(), d_counts, sizeof(uint) * combinations, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_bitmaps));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_set_offsets));
    CUDA_CHECK(cudaFree(d_elements));
}

__global__ void ConstructBitmapKernel(const uint *elements, const uint *set_offsets,
                                      uint num_of_sets, word_t *bitmaps, uint words) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < num_of_sets; i += 1) {
        uint set_start = set_offsets[i];
        uint set_end = set_offsets[i + 1];
        uint bitmap_start = i * words;
        for (int j = set_start + tid; j < set_end; j += gridDim.x * blockDim.x) {
            uint e = elements[j];
            atomicOr(bitmaps + bitmap_start + (e / WORD_BITS), ((word_t)1 << (e % WORD_BITS)));
        }
    }
}

__global__ void IntersectionKernel(tile_t A, tile_t B, const word_t *bitmaps, uint words,
                                   uint *counts) {
    uint tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint count_offset = 0;
    for (int a = A.start; a < A.end; a += 1) {
        uint a_bitmap_start = a * words;
        for (int b = B.start; b < B.end; b += 1) {
            uint b_bitmap_start = b * words;
            uint count = 0;
            for (int i = tid; i < words; i += gridDim.x * blockDim.x) {
                count += __popcll(bitmaps[a_bitmap_start + i] & bitmaps[b_bitmap_start + i]);
            }
            atomicAdd(counts + count_offset + (b - B.start), count);
        }
        count_offset += B.size;
    }
}
