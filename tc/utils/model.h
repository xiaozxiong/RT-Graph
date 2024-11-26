#pragma once

#include <cuda_runtime.h>
#include <vector>

typedef struct Model {
    std::vector<float3> vertices;
} model_t;

typedef struct GraphInfo {
    int nodes;
    int edges;

} graph_info_t;

typedef struct edge {
    int src;
    int dst;
    __host__ __device__ bool operator<(const edge &x) const {
        if (this->src == x.src)
            return this->dst < x.dst;
        else
            return this->src < x.src;
    }
    // __host__ __device__ bool operator > (const edge& x) const{
    //     if(this->src==x.src) return this->dst>x.dst;
    //     else return this->src>x.src;
    // }
} edge_t;

// enum RTMethods{
//     ListIntersecton = 0,
//     Hashmap =1
// };