#include "../rt_tc/include/config.h"
#include "bs_tc.h"
#include "common.h"
#include "timer.h"

#include <omp.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 256
#define MAX_BLOCKS 65535

__device__ bool BinaryCheck(int *data, int l, int r, int val) {
    while (l <= r) {
        int mid = (l + r) >> 1;
        if (data[mid] == val)
            return true;
        else if (data[mid] < val)
            l = mid + 1;
        else
            r = mid - 1;
    }
    return false;
}

#if RTTC_METHOD == 0
#ifdef LIST_OPTIMIZATION
void UniqueandCount(int *adjs, int *offsets, int nodes, std::vector<int> &unique_two_hops,
                    std::vector<size_t> &two_hops_offsets, std::vector<int> &add_vals) {
    uint max_two_hop = 0;
#pragma omp parallel for reduction(max : max_two_hop)
    for (int i = 0; i < nodes; i += 1) {
        int start = offsets[i];
        int end = offsets[i + 1];
        uint tmp_size = 0;
        for (int j = start; j < end; j += 1) {
            int adj = adjs[j];
            tmp_size += (offsets[adj + 1] - offsets[adj]);
        }
        max_two_hop = std::max(max_two_hop, tmp_size);
    }
    // printf("max_two_hop = %d\n", max_two_hop);

    uint total_num = 0; // total two-hops neighbors
    int *temp_two_hop_adjs = (int *)malloc(sizeof(int) * max_two_hop);
    size_t unique_offset = 0;
    for (int i = 0; i < nodes; i += 1) {
        int start = offsets[i];
        int end = offsets[i + 1];
        int offset = 0;
        for (int j = start; j < end; j += 1) {
            int adj = adjs[j];
            int len = offsets[adj + 1] - offsets[adj];
            memcpy(temp_two_hop_adjs + offset, adjs + offsets[adj], sizeof(int) * len);
            offset += len;
        }
        uint two_adj_size = offset;
        total_num += two_adj_size;

        std::sort(temp_two_hop_adjs, temp_two_hop_adjs + two_adj_size);

        int last_node = -1, cnt = 0;
        uint unique_cnt = 0;
        two_hops_offsets[i] = unique_offset;
        for (int j = 0; j < two_adj_size; j += 1) {
            int cur_node = temp_two_hop_adjs[j];
            if (last_node == cur_node)
                cnt += 1;
            else {
                if (last_node != -1) {
                    unique_cnt += 1;
                    unique_two_hops.push_back(last_node);
                    add_vals.push_back(cnt);
                }
                cnt = 1;
            }
            last_node = cur_node;
        }
        if (last_node != -1) {
            unique_cnt += 1;
            unique_two_hops.push_back(last_node);
            add_vals.push_back(cnt);
        }
        unique_offset += unique_cnt;
    }
    assert(unique_offset == add_vals.size());
    assert(unique_two_hops.size() == add_vals.size());
    two_hops_offsets[nodes] = unique_offset;
    free(temp_two_hop_adjs);
}

__device__ int BinaryCheckOpt(int *data, size_t l, size_t r, int target, int *add_vals) {
    while (l <= r) {
        size_t mid = l + (r - l) / 2;
        if (data[mid] == target)
            return add_vals[mid];
        else if (data[mid] > target)
            r = mid - 1;
        else
            l = mid + 1;
    }
    return 0;
}

__global__ void ListIntersectionOpt(int *ray_start, int *ray_adjs, int num, int *two_hops,
                                    size_t *two_hop_offsets, int *add_vals, uint *count) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < num) {
        int node = ray_start[tid];
        int target = ray_adjs[tid];
        int res = BinaryCheckOpt(two_hops, two_hop_offsets[node], two_hop_offsets[node + 1] - 1,
                                 target, add_vals);
        if (res > 0)
            atomicAdd(count, res);
    }
}

#else
__global__ void ListIntersectionKernel(int *adjs, int *offsets, int nodes, int threads,
                                       uint *count) {
    uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint cur_id = tid;
    while (cur_id < threads) {
        uint val = adjs[cur_id];
        uint l = 0, r = nodes;
        while (l < r) {
            int mid = (l + r + 1) >> 1;
            if (offsets[mid] <= cur_id)
                l = mid;
            else
                r = mid - 1;
        }
        uint node = l;
        int start = offsets[node];
        int end = offsets[node + 1];
        uint cnt = 0;
        for (int i = start; i < end; i++) {
            uint adj = adjs[i];
            uint two_adj_size = offsets[adj + 1] - offsets[adj];
            if (two_adj_size == 0)
                continue;
            cnt += BinaryCheck(adjs + offsets[adj], 0, two_adj_size - 1, val);
        }
        atomicAdd(count, cnt);
        cur_id += gridDim.x * blockDim.x;
    }
}
#endif
#endif

#if RTTC_METHOD == 1
void Preprocessing(int *adjs, int *offsets, uint nodes, std::vector<uint> &ray_origins_vals) {
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();

    uint max_two_hop = 0;
#pragma omp parallel for reduction(max : max_two_hop)
    for (int i = 0; i < nodes; i += 1) {
        int start = offsets[i];
        int end = offsets[i + 1];
        uint tmp_size = 0;
        for (int j = start; j < end; j += 1) {
            int adj = adjs[j];
            tmp_size += (offsets[adj + 1] - offsets[adj]);
        }
        max_two_hop = std::max(max_two_hop, tmp_size);
    }

    int *two_hop_adjs = (int *)malloc(sizeof(int) * max_two_hop);
    for (int i = 0; i < nodes; i += 1) {
        int start = offsets[i];
        int end = offsets[i + 1];
        int offset = 0;
        for (int j = start; j < end; j += 1) {
            int adj = adjs[j]; // one-hop
            int len = offsets[adj + 1] - offsets[adj];
            memcpy(two_hop_adjs + offset, adjs + offsets[adj], sizeof(int) * len);
            offset += len;
        }
        uint two_adj_size = offset;
        if (two_adj_size == 0)
            continue;
        // ThrustSort(two_hop_adjs,two_adj_size);
        std::sort(two_hop_adjs, two_hop_adjs + two_adj_size);
        int last_node = -1, cnt = 0;
        for (int j = 0; j < two_adj_size; j += 1) {
            int cur_node = two_hop_adjs[j];
            if (last_node == cur_node)
                cnt += 1;
            else {
                if (last_node != -1) {
                    ray_origins_vals.push_back(i);
                    ray_origins_vals.push_back(last_node);
                    ray_origins_vals.push_back(cnt);
                }
                cnt = 1;
            }
            last_node = cur_node;
        }
        if (last_node != -1) {
            ray_origins_vals.push_back(i);
            ray_origins_vals.push_back(last_node);
            ray_origins_vals.push_back(cnt);
        }
    }
    free(two_hop_adjs);

    cpu_timer.StopTiming();
    printf("BSTC: preprocessing time = %f ms\n", cpu_timer.GetElapsedTime());
}

__global__ void HashmapKernel(uint *ray_origins_vals, int *adjs, int *offsets, uint nodes,
                              uint total_load, uint *count) {
    uint tid = threadIdx.x + blockDim.x * blockIdx.x;
    uint cur_id = tid;
    while (cur_id < total_load) {
        uint id = cur_id * 3;
        uint node = ray_origins_vals[id];
        uint two_hop_adj = ray_origins_vals[id + 1];
        uint val = ray_origins_vals[id + 2];
        uint adj_size = offsets[node + 1] - offsets[node];
        if (adj_size == 0)
            return;
        if (BinaryCheck(adjs + offsets[node], 0, adj_size - 1, two_hop_adj))
            atomicAdd(count, val);
        cur_id += gridDim.x * blockDim.x;
    }
}
#endif

void CountOnGPU(int *adjs, int *offsets, int nodes, int edges, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));

    int *d_adjs, *d_offsets;
    CUDA_CHECK(cudaMalloc((void **)&d_adjs, sizeof(int) * edges));
    CUDA_CHECK(cudaMemcpy(d_adjs, adjs, sizeof(int) * edges, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&d_offsets, sizeof(int) * (nodes + 1)));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets, sizeof(int) * (nodes + 1), cudaMemcpyHostToDevice));
    uint *d_count;
    CUDA_CHECK(cudaMalloc((void **)&d_count, sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    double count_time = 0.0;

#if RTTC_METHOD == 0
    printf("BSTC: using list intersection based method(1A2)\n");
#ifdef LIST_OPTIMIZATION
    printf("BSTC: using list optimization\n");
    std::vector<int> unique_two_hops;
    std::vector<size_t> two_hops_offsets(nodes + 1);
    std::vector<int> add_vals;
    UniqueandCount(adjs, offsets, nodes, unique_two_hops, two_hops_offsets, add_vals);
    // int maxn = 0, max_val = 0;
    // for(int i=0;i < nodes; i++){
    //     if(maxn < two_hops_offsets[i+1] - two_hops_offsets[i]){
    //         maxn = two_hops_offsets[i+1] - two_hops_offsets[i];
    //     }
    //     for(int j = two_hops_offsets[i]; j < two_hops_offsets[i+1]; j++){
    //         if(max_val < add_vals[j]){
    //             max_val = add_vals[j];
    //         }
    //     }
    // }
    // printf("max unique two hops = %d, max val = %d\n", maxn, max_val);
    // printf("unique two-hops adjs = %u\n", unique_two_hops.size());
    std::vector<int> ray_nodes(edges);
    for (int i = 0; i < nodes; i++) {
        int start = offsets[i];
        int end = offsets[i + 1];
        for (int j = start; j < end; j++) {
            ray_nodes[j] = i;
        }
    }
    int *d_ray_nodes;
    CUDA_CHECK(cudaMalloc((void **)&d_ray_nodes, sizeof(int) * ray_nodes.size()));
    CUDA_CHECK(cudaMemcpy(d_ray_nodes, ray_nodes.data(), sizeof(int) * ray_nodes.size(),
                          cudaMemcpyHostToDevice));
    int *d_unique_two_hops;
    CUDA_CHECK(cudaMalloc((void **)&d_unique_two_hops, sizeof(int) * unique_two_hops.size()));
    CUDA_CHECK(cudaMemcpy(d_unique_two_hops, unique_two_hops.data(),
                          sizeof(int) * unique_two_hops.size(), cudaMemcpyHostToDevice));
    size_t *d_two_hops_offsets;
    CUDA_CHECK(cudaMalloc((void **)&d_two_hops_offsets, sizeof(size_t) * two_hops_offsets.size()));
    CUDA_CHECK(cudaMemcpy(d_two_hops_offsets, two_hops_offsets.data(),
                          sizeof(size_t) * two_hops_offsets.size(), cudaMemcpyHostToDevice));
    int *d_add_vals;
    CUDA_CHECK(cudaMalloc((void **)&d_add_vals, sizeof(int) * add_vals.size()));
    CUDA_CHECK(cudaMemcpy(d_add_vals, add_vals.data(), sizeof(int) * add_vals.size(),
                          cudaMemcpyHostToDevice));

    int block_size = BLOCK_SIZE;
    int blocks = (edges + block_size - 1) / block_size;

    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    ListIntersectionOpt<<<blocks, block_size>>>(d_ray_nodes, d_adjs, edges, d_unique_two_hops,
                                                d_two_hops_offsets, d_add_vals, d_count);
    CUDA_SYNC_CHECK();
    cpu_timer.StopTiming();
    count_time = cpu_timer.GetElapsedTime();

    CUDA_CHECK(cudaFree(d_add_vals));
    CUDA_CHECK(cudaFree(d_two_hops_offsets));
    CUDA_CHECK(cudaFree(d_unique_two_hops));
    CUDA_CHECK(cudaFree(d_ray_nodes));
#else
    printf("BSTC: no list optimization\n");
    int block_size = BLOCK_SIZE;
    int blocks = (edges + block_size - 1) / block_size;
    blocks = (blocks <= MAX_BLOCKS ? blocks : MAX_BLOCKS);

    GPUTimer gpu_timer;
    // CPUTimer cpu_timer;
    // cpu_timer.StartTiming();
    gpu_timer.StartTiming();
    ListIntersectionKernel<<<blocks, block_size>>>(d_adjs, d_offsets, nodes, edges, d_count);
    gpu_timer.StopTiming();
    // cpu_timer.StopTiming();
    // printf("cpu time = %f ms\n", cpu_timer.GetElapsedTime());
    count_time = gpu_timer.GetElapsedTime();
#endif
#elif RTTC_METHOD == 1
    printf("BSTC: using hashmap based method (2A1)\n");
    std::vector<uint> ray_origins_vals;
    Preprocessing(adjs, offsets, nodes, ray_origins_vals);

    uint *d_ray_origins_vals;
    CUDA_CHECK(cudaMalloc((void **)&d_ray_origins_vals, sizeof(uint) * ray_origins_vals.size()));
    CUDA_CHECK(cudaMemcpy(d_ray_origins_vals, ray_origins_vals.data(),
                          sizeof(uint) * ray_origins_vals.size(), cudaMemcpyHostToDevice));

    uint total_load = ray_origins_vals.size() / 3;
    uint block_size = BLOCK_SIZE;
    uint blocks = (total_load + block_size - 1) / block_size;
    blocks = (blocks <= MAX_BLOCKS ? blocks : MAX_BLOCKS);

    GPUTimer gpu_timer;
    gpu_timer.StartTiming();
    HashmapKernel<<<blocks, block_size>>>(d_ray_origins_vals, d_adjs, d_offsets, nodes, total_load,
                                          d_count);
    gpu_timer.StopTiming();
    count_time = gpu_timer.GetElapsedTime();
    CUDA_CHECK(cudaFree(d_ray_origins_vals));
#endif
    uint count;
    CUDA_CHECK(cudaMemcpy(&count, d_count, sizeof(uint), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_adjs));

    printf("BSTC: counting time = %f ms, counting result = %u\n", count_time, count);
}
