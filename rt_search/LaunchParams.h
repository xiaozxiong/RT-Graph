#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include "helper.h"
#include "config.h"
struct LaunchParams{
    OptixTraversableHandle handle;
    // float *ray_lengths;
    // float3 *ray_origins;
    Ray *rays;
    int *results;
#if PRIMITIVE != 0
    float3 *centers;
    float side; // radius for sphere, side for aabb
#endif
};
