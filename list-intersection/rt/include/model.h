#pragma once

#include <vector>
#include <cuda_runtime.h>
typedef struct Model{
    uint num_of_triangles=0;
    uint num_of_vertices=0;
    std::vector<float3> vertices;
} model_t;