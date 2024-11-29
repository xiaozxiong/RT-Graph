#include "simple_bs.h"
// #include "Timing.h"
#include <algorithm>
#include <chrono>
#include <ctime>

#define EPSILON 1e-6;

#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

// *float32
// binary search for index of the first number greater than or equal to the query
__global__ void BinarySearchKernel(float *data, int data_size, float *query, int query_size,
                                   int *results) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < query_size; i += gridDim.x * blockDim.x) {
        int l = 0, r = data_size - 1;
        int target = query[i];
        while (l < r) {
            int mid = l + (r - l) / 2;
            if (data[mid] < target)
                l = mid + 1;
            else
                r = mid;
        }
        results[i] = (data[l] >= target ? l : -1);
    }
}

//* int32, just find the index of the first number equal to target
__global__ void BSAccessCount(int *data, int data_size, int *query, int query_size,
                              int *results) { // ,int *access_count
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < query_size; i += gridDim.x * blockDim.x) {
        int l = 0, r = data_size - 1;
        int target = query[i];
        // int access=0;
        while (l < r) {
            int mid = l + (r - l) / 2;
            // access+=1;
            if (data[mid] < target)
                l = mid + 1;
            else
                r = mid;
        }
        results[i] = (data[l] == target ? l : -1);
        // access_count[i]=access;
    }
}

//* int32, just check the existence of target
__global__ void BSAccessCount_Check(int *data, int data_size, int *query, int query_size,
                                    int *results, int *access_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < query_size; i += gridDim.x * blockDim.x) {
        int l = 0, r = data_size - 1;
        int target = query[i];
        int access = 0;
        int x = -1;
        while (l <= r) {
            int mid = l + (r - l) / 2;
            access += 1;
            if (data[mid] == target) {
                x = mid;
                break;
            } else if (data[mid] < target)
                l = mid + 1;
            else
                r = mid + 1;
        }
        results[i] = x;
        access_count[i] = access;
    }
}
// TODO: measure access cost, int
void BSAccessCountFunction(int *data, int data_size, int *query, int query_size, int *bs_results) {
    int *d_data, *d_query;
    cudaMalloc((void **)&d_data, sizeof(int) * data_size);
    cudaMalloc((void **)&d_query, sizeof(int) * query_size);
    int *d_results;
    cudaMalloc((void **)&d_results, sizeof(int) * query_size);

    cudaMemcpy(d_data, data, sizeof(int) * data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, sizeof(int) * query_size, cudaMemcpyHostToDevice);
    printf("data size = %d\n", data_size);

    std::clock_t begin = clock();
    std::sort(data, data + data_size);
    std::clock_t end = clock();
    double cpu_sort_time = double(end - begin) / CLOCKS_PER_SEC * 1000.0; // ms
    printf("cpu sort time = %.3f ms\n", cpu_sort_time);

    //* sorting
    thrust::device_ptr<int> d_ptr(d_data);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    // void* d_temp_storage = nullptr;
    // size_t temp_storage_bytes = 0;
    thrust::sort(d_ptr, d_ptr + data_size); // so slow!
    // cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data,
    // data_size); cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_data, d_data,
    // data_size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    double sort_time = duration.count(); // ms

    // cudaFree(d_temp_storage);

    //* access count
    // int *d_access_count;
    // cudaMalloc((void**)&d_access_count,sizeof(int)*query_size);
    // cudaMemset(d_access_count,0,sizeof(int)*query_size);

    int block_size = 256;
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, BSAccessCount, block_size, 0);
    int blocks = blocks_per_sm * 100;

    float bs_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    BSAccessCount<<<blocks, block_size>>>(d_data, data_size, d_query, query_size,
                                          d_results); // d_access_count
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&bs_time, start, stop);

    cudaMemcpy(bs_results, d_results, sizeof(int) * query_size, cudaMemcpyDeviceToHost);

    // double
    // avg_access=1.0*thrust::reduce(thrust::device,d_access_count,d_access_count+query_size)/query_size;
    // printf("Avg access = %f\n",avg_access);
    printf("Sorting time = %f ms, Binary search time = %f ms, Total time = %f ms\n", sort_time,
           bs_time, sort_time + bs_time);
    // cudaFree(d_access_count);
}
// TODO: just binary search
void BSFunction(float *data, int data_size, float *query, int query_size, int *bs_results) {
    float *d_data, *d_query;
    cudaMalloc((void **)&d_data, sizeof(float) * data_size);
    cudaMalloc((void **)&d_query, sizeof(float) * query_size);
    int *d_results;
    cudaMalloc((void **)&d_results, sizeof(int) * query_size);

    cudaMemcpy(d_data, data, sizeof(float) * data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, query, sizeof(float) * query_size, cudaMemcpyHostToDevice);

    //* sorting
    auto cpu_start = std::chrono::high_resolution_clock::now();
    thrust::sort(thrust::device, d_data, d_data + data_size);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    double sort_time = duration.count(); // ms

    int block_size = 256;
    int blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, BinarySearchKernel, block_size,
                                                  0);
    int blocks = blocks_per_sm * 100;

    float bs_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    BinarySearchKernel<<<blocks, block_size>>>(d_data, data_size, d_query, query_size, d_results);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&bs_time, start, stop);

    cudaMemcpy(bs_results, d_results, sizeof(int) * query_size, cudaMemcpyDeviceToHost);
    printf("BS: sorting time = %.3f ms, searching time = %f ms\n", sort_time, bs_time);
}