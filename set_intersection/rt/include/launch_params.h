#pragma once

#include <optix.h>
#include <cuda_runtime.h>

struct LaunchParams{
    OptixTraversableHandle handle;
    uint *ray_set_ids; // ray -> set
    uint *triangle_set_ids; // primitive -> set
    uint *ray_origins;
    uint ray_length;
    uint *counts;
    float3 axis_offset;
    uint chunk_length; // chunk length in z axis
};
