#pragma once

#include <stdint.h>

#define OPTIX_CHECK(call)                                                                          \
    {                                                                                              \
        OptixResult res = call;                                                                    \
        if (res != OPTIX_SUCCESS) {                                                                \
            fprintf(stderr, "Optix call (%s) failed with code %d (line %d)\n", #call, res,         \
                    __LINE__);                                                                     \
            exit(2);                                                                               \
        }                                                                                          \
    }

#define CUDA_CHECK(call)                                                                           \
    {                                                                                              \
        const cudaError_t res = call;                                                              \
        if (res != cudaSuccess) {                                                                  \
            printf("CUDA error: %s:%d, ", __FILE__, __LINE__);                                     \
            printf("code:%d, reason: %s\n", res, cudaGetErrorString(res));                         \
            exit(2);                                                                               \
        }                                                                                          \
    }

#define CUDA_SYNC_CHECK()                                                                          \
    {                                                                                              \
        cudaDeviceSynchronize();                                                                   \
        cudaError_t res = cudaGetLastError();                                                      \
        if (res != cudaSuccess) {                                                                  \
            fprintf(stderr, "error (%s: line %d): %s\n", __FILE__, __LINE__,                       \
                    cudaGetErrorString(res));                                                      \
            exit(2);                                                                               \
        }                                                                                          \
    }

typedef uint32_t idx_t; // node id
// typedef long long unsigned int res_t; // result type
typedef uint32_t num_t; // node number and edge number

typedef struct Edge {
    idx_t src;
    idx_t dst;
} edge_t;

typedef struct GraphInfo {
    int node_num;
    int edge_num;
    int min_degree;
    int max_degree;
    double avg_degree;
} graph_info_t;
