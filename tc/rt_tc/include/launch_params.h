#pragma once

#include "config.h"
#include <optix.h>
#include <cuda_runtime.h>

struct LaunchParams {
    OptixTraversableHandle handle;
    float3 axis_offset;
    uint *count;
};

//* method basing on list intersection
struct ListIntersectionParams : public LaunchParams {
#ifdef LIST_OPTIMIZATION
    uint *triangle_vals; // increment for an intersection
    float2 *ray_origin;
#else
    uint nodes;
    uint *ray_offset;
    uint *ray_length;
    uint *ray_origin;
#endif
};

//* method basing on hashmap
struct HashmapParams : public LaunchParams {
    uint *ray_origin_and_val;
    uint total_load;
    uint rays;
    uint *miss_record;
};