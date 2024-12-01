#include "common.h"
#include "timer.h"
#include "util.hpp"

#include <cxxopts.hpp>
#include <iostream>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

__forceinline__ __device__ int myMax(int a, int b) { return (a < b) ? b : a; }

__forceinline__ __device__ int myMin(int a, int b) { return (a > b) ? b : a; }

__device__ inline unsigned int serialIntersect(unsigned int *a, unsigned int aBegin,
                                               unsigned int aEnd, unsigned int *b,
                                               unsigned int bBegin, unsigned int bEnd,
                                               unsigned int vt) {
    unsigned int count = 0;

    // vt parameter must be odd integer
    for (int i = 0; i < vt; i++) {
        bool p = false;
        if (aBegin >= aEnd)
            p = false; // a, out of bounds
        else if (bBegin >= bEnd)
            p = true; // b, out of bounds
        else {
            if (a[aBegin] < b[bBegin])
                p = true;
            if (a[aBegin] == b[bBegin]) {
                count++;
            }
        }
        if (p)
            aBegin++;
        else
            bBegin++;
    }
    return count;
}

__forceinline__ __device__ int binarySearch(unsigned int *a, unsigned int aSize, unsigned int *b,
                                            unsigned int bSize, unsigned int diag) {
    int begin = myMax(0, diag - bSize);
    int end = myMin(diag, aSize);

    while (begin < end) {
        int mid = (begin + end) / 2;

        if (a[mid] < b[diag - 1 - mid])
            begin = mid + 1;
        else
            end = mid;
    }
    return begin;
}
template <bool selfJoin>
__global__ void findDiagonals(tile_t A, tile_t B, unsigned int numOfSets, unsigned int *sets,
                              const unsigned int *sizes, unsigned int *offsets,
                              unsigned int *globalDiagonals, unsigned int *counts) {

    for (unsigned int a = A.start; a < A.end; a++) {
        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < B.end;
             b++) { // iterate every combination
            unsigned int offset =
                selfJoin ? triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets)
                         : quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets);
            unsigned int aSize = sizes[a];
            unsigned int bSize = sizes[b];
            unsigned int *aSet = sets + offsets[a];
            unsigned int *bSet = sets + offsets[b];
            unsigned int *diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * offset;

            unsigned int combinedIndex = (uint64_t)blockIdx.x *
                                         ((uint64_t)sizes[a] + (uint64_t)sizes[b]) /
                                         (uint64_t)gridDim.x;
            __shared__ int xTop, yTop, xBottom, yBottom, found;
            __shared__ unsigned int
                oneOrZero[32]; // array size must be equal to number of block threads
            __shared__ unsigned int
                increment; // use this as flag to ensure single increment, find a more elegant way

            increment = 0;

            unsigned int threadOffset = threadIdx.x - 16;

            xTop = myMin(combinedIndex, aSize);
            yTop = combinedIndex > aSize ? combinedIndex - aSize : 0;
            xBottom = yTop;
            yBottom = xTop;

            found = 0;

            // Search the diagonal
            while (!found) {
                // Update our coordinates within the 32-wide section of the diagonal
                int currentX = xTop - ((xTop - xBottom) >> 1) - threadOffset;
                int currentY = yTop + ((yBottom - yTop) >> 1) + threadOffset;

                // Are we a '1' or '0' with respect to A[x] <= B[x]
                if (currentX >= aSize || currentY < 0) {
                    oneOrZero[threadIdx.x] = 0;
                } else if (currentY >= bSize || currentX < 1) {
                    oneOrZero[threadIdx.x] = 1;
                } else {
                    oneOrZero[threadIdx.x] = (aSet[currentX - 1] <= bSet[currentY]) ? 1 : 0;
                    if (aSet[currentX - 1] == bSet[currentY] &&
                        increment == 0) { // count augmentation
                        atomicAdd(counts + offset, 1);
                        atomicAdd(&increment, 1);
                    }
                }

                __syncthreads();

                // If we find the meeting of the '1's and '0's, we found the
                // intersection of the path and diagonal
                if (threadIdx.x > 0 && (oneOrZero[threadIdx.x] != oneOrZero[threadIdx.x - 1]) &&
                    currentY >= 0 && currentX >= 0) {
                    found = 1;
                    diagonals[blockIdx.x] = currentX;
                    diagonals[blockIdx.x + gridDim.x + 1] = currentY;
                }

                __syncthreads();

                // Adjust the search window on the diagonal
                if (threadIdx.x == 16) {
                    if (oneOrZero[31] != 0) {
                        xBottom = currentX;
                        yBottom = currentY;
                    } else {
                        xTop = currentX;
                        yTop = currentY;
                    }
                }
                __syncthreads();
            }

            // Set the boundary diagonals (through 0,0 and A_length,B_length)
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                diagonals[0] = 0;
                diagonals[gridDim.x + 1] = 0;
                diagonals[gridDim.x] = aSize;
                diagonals[gridDim.x + gridDim.x + 1] = bSize;
            }

            oneOrZero[threadIdx.x] = 0;

            __syncthreads();
        }
    }
}

template <bool selfJoin>
__global__ void intersectPath(tile_t A, tile_t B, unsigned int numOfSets, unsigned int *sets,
                              unsigned int *offsets, unsigned int *globalDiagonals,
                              unsigned int *counts);

int main(int argc, char **argv) {
    try {

        int multiprocessorCount;
        int maxThreadsPerBlock;

        cudaDeviceGetAttribute(&multiprocessorCount, cudaDevAttrMultiProcessorCount, 0);
        cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

        // arguments
        std::string input;
        // std::string output;
        unsigned int blocks = multiprocessorCount * 16;
        unsigned int blockSize = maxThreadsPerBlock / 2;
        unsigned int partition = 10000;

        cxxopts::Options options(argv[0], "Help");

        options.add_options()("input", "Input dataset path", cxxopts::value<std::string>(input))
            // ("output", "Output result path", cxxopts::value<std::string>(output))
            ("blocks", "Number of blocks (default: " + std::to_string(blocks) + ")",
             cxxopts::value<unsigned int>(blocks))(
                "threads", "Threads per block (default: " + std::to_string(blockSize) + ")",
                cxxopts::value<unsigned int>(blockSize))
            // ("partition", "Number of sets to be processed per GPU invocation",
            // cxxopts::value<unsigned int>(partition))
            ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return 0;
        }

        if (!result.count("input")) {
            fmt::print("{}\n", "No input dataset given! Exiting...");
            return 1;
        }
        cudaSetDevice(3);
        Dataset dataset;
        ReadData(path, dataset);
        tile_t A = {0, dataset.num_of_sets_a, dataset.num_of_sets_a};                   // first
        tile_t B = {dataset.num_of_sets_a, dataset.num_of_sets, dataset.num_of_sets_b}; // last

        size_t combinations = A.size * B.size;
        std::vector<unsigned int> counts(combinations);

        // DeviceTimer deviceTimer;

        // allocate device memory space
        unsigned int *deviceOffsets;
        // unsigned int* deviceSizes;
        unsigned int *deviceElements;
        unsigned int *deviceCounts;
        unsigned int *deviceDiagonals; //?

        // EventPair *devMemAlloc = deviceTimer.add("Device memory allocation");
        errorCheck(
            cudaMalloc((void **)&deviceOffsets, sizeof(unsigned int) * dataset.set_offsets.size()))
            // errorCheck(cudaMalloc((void**)&deviceSizes, sizeof(unsigned int) * d->cardinality))
            errorCheck(cudaMalloc((void **)&deviceElements,
                                  sizeof(unsigned int) * dataset.elements.size()))
                errorCheck(
                    cudaMalloc((void **)&deviceCounts, sizeof(unsigned int) * combinations + 1))
                    errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations + 1))
                        errorCheck(
                            cudaMalloc((void **)&deviceDiagonals,
                                       sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
                            errorCheck(
                                cudaMemset(deviceDiagonals, 0,
                                           sizeof(unsigned int) * 2 * (blocks + 1) * combinations))
            // DeviceTimer::finish(devMemAlloc);

            // EventPair *dataTransfer = deviceTimer.add("Transfer to device");
            // errorCheck(cudaMemcpy(deviceSizes, d->sizes, sizeof(unsigned int) * d->cardinality,
            // cudaMemcpyHostToDevice))
            errorCheck(cudaMemcpy(deviceOffsets, dataset.set_offsets.data(),
                                  sizeof(unsigned int) * dataset.set_offsets.size(),
                                  cudaMemcpyDeviceToDevice))
                errorCheck(cudaMemcpy(deviceElements, dataset.elements.data(),
                                      sizeof(unsigned int) * dataset.elements.size(),
                                      cudaMemcpyHostToDevice))
            // DeviceTimer::finish(dataTransfer);

            // EventPair *setOffsets = deviceTimer.add("Compute set offsets");
            // thrust::exclusive_scan(thrust::device, deviceOffsets, deviceOffsets + d->cardinality,
            // deviceOffsets,
            //                        0); // in-place scan
            // DeviceTimer::finish(setOffsets);

            // unsigned int iter = 0;
            // for (auto& run : runs) {
            // tile &A = run.first;
            // tile &B = run.second;
            // bool selfJoin = A.id == B.id;
            // unsigned int numOfSets = selfJoin && runs.size() == 1 ? d->cardinality : partition;

            // EventPair *findDiags = deviceTimer.add("Find diagonals");
            // if (selfJoin) {
            //     findDiagonals<true><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes,
            //                                   deviceOffsets, deviceDiagonals, deviceCounts);
            // } else {//!
            findDiagonals<false><<<blocks, 32>>>(A, B, numOfSets, deviceElements, deviceSizes,
                                                 deviceOffsets, deviceDiagonals, deviceCounts);
        // }
        // DeviceTimer::finish(findDiags);

        // EventPair *computeIntersections = deviceTimer.add("Intersect path");
        // if (selfJoin) {
        //     intersectPath<true><<<blocks, blockSize, blockSize * sizeof(unsigned int)>>>
        //             (A, B, numOfSets, deviceElements, deviceOffsets, deviceDiagonals,
        //             deviceCounts);
        // } else {
        intersectPath<false><<<blocks, blockSize, blockSize * sizeof(unsigned int)>>>(
            A, B, numOfSets, deviceElements, deviceOffsets, deviceDiagonals, deviceCounts);
        // }
        // DeviceTimer::finish(computeIntersections);

        // EventPair *countTransfer = deviceTimer.add("Transfer result");
        errorCheck(cudaMemcpy(&counts[0] + combinations, deviceCounts,
                              sizeof(unsigned int) * combinations, cudaMemcpyDeviceToHost));
        // DeviceTimer::finish(countTransfer);

        // EventPair* clearMemory = deviceTimer.add("Clear memory");
        errorCheck(
            cudaMemset(deviceDiagonals, 0, sizeof(unsigned int) * 2 * (blocks + 1) * combinations));
        errorCheck(cudaMemset(deviceCounts, 0, sizeof(unsigned int) * combinations));
        // DeviceTimer::finish(clearMemory);
        // iter++;
        // }

        // EventPair *freeDevMem = deviceTimer.add("Free device memory");
        errorCheck(cudaFree(deviceOffsets));
        // errorCheck(cudaFree(deviceSizes))
        errorCheck(cudaFree(deviceElements));
        errorCheck(cudaFree(deviceCounts));
        // DeviceTimer::finish(freeDevMem);

        cudaDeviceSynchronize();

        // deviceTimer.print();

        // if (!output.empty()) {
        //     fmt::print("Writing result to file {}\n", output);
        //     if (runs.size() == 1) {
        //         writeResult(d->cardinality, counts, output);
        //     } else {
        //         writeResult(runs, partition, counts, output);
        //     }
        //     fmt::print("Finished\n");
        // }

    } catch (const cxxopts::OptionException &e) {
        // fmt::print("{}\n", e.what());

        std::cout << e.what() << std::endl;
        return 1;
    }
    return 0;
}

template <bool selfJoin>
__global__ void intersectPath(tile_t A, tile_t B, unsigned int numOfSets, unsigned int *sets,
                              unsigned int *offsets, unsigned int *globalDiagonals,
                              unsigned int *counts) {

    for (unsigned int a = A.start; a < A.end; a++) {
        for (unsigned int b = (selfJoin ? a + 1 : B.start); b < B.end;
             b++) { // iterate every combination
            unsigned int offset =
                (selfJoin ? triangular_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets)
                          : quadratic_idx(numOfSets, a - A.id * numOfSets, b - B.id * numOfSets));
            unsigned int *aSet = sets + offsets[a]; // start of set a
            unsigned int *bSet = sets + offsets[b]; // start of set b
            unsigned int *diagonals = globalDiagonals + (2 * (gridDim.x + 1)) * offset;
            // in a grid
            unsigned int aStart = diagonals[blockIdx.x];
            unsigned int aEnd = diagonals[blockIdx.x + 1];

            unsigned int bStart = diagonals[(gridDim.x + 1) + blockIdx.x];
            unsigned int bEnd = diagonals[(gridDim.x + 1) + blockIdx.x + 1];

            unsigned int aSize = aEnd - aStart;
            unsigned int bSize = bEnd - bStart;
            // in a block
            unsigned int vt = ((aSize + bSize) / blockDim.x) + 1;

            // local diagonal
            unsigned int diag = threadIdx.x * vt;

            int mp = binarySearch(aSet + aStart, aSize, bSet + bStart, bSize, diag);

            unsigned int intersection =
                serialIntersect(aSet + aStart, mp, aSize, bSet + bStart, diag - mp, bSize, vt);
            atomicAdd(counts + offset, intersection);
        }
    }
}