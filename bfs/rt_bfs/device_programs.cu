#include "launch_params.h"
#include <cuda_runtime.h>
#include <math.h>
#include <optix_device.h>
#include <stdint.h>

extern "C" {
__constant__ LaunchParams params;
}

__forceinline__ __device__ void ray_trace(float3 &ray_origin, float3 &ray_direction, float &tmin,
                                          float &tmax) {
    // unsigned int p0=ray_id;
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, //|OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0, 0, 0);
    // if(p0!=params.neighbor_size[curr_vertex]){
    //     printf("triangle error: %d , p0 = %d, neighbor size = %d, tmin = %f,
    //     tmax =
    //     %f\n",params.vir_real[curr_vertex],p0,params.neighbor_size[curr_vertex],tmin,tmax);
    // }
}

__forceinline__ __device__ void get_origin_coordinate(int origin, float3 &ray_origin) {
    ray_origin.y = -0.5f;
    if (origin >= params.adjust)
        origin -= params.adjust; //*
    if (origin == 0) {
        ray_origin.x = 0.0f;
        ray_origin.z = 0.0f;
    } else {
        int side = (int)sqrt(1.0 * origin);
        side += (side % 2 == 0 ? 1 : 2);
        int res = side * side - 1 - origin;
        int temp = res / (side - 1);
        int axis = (int)(side / 2);
        if (temp == 0) {
            ray_origin.x = 1.0f * (axis - res);
            ray_origin.z = 1.0f * (-axis);
        } else if (temp == 1) {
            ray_origin.x = 1.0f * (-axis);
            ray_origin.z = 1.0f * (-axis + res - (side - 1));
        } else if (temp == 2) {
            ray_origin.x = 1.0f * (-axis + res - 2 * (side - 1));
            ray_origin.z = 1.0f * (axis);
        } else if (temp == 3) {
            ray_origin.x = 1.0f * axis;
            ray_origin.z = 1.0f * (axis - (res - 3 * (side - 1)));
        }
    }
    ray_origin.x *= params.zoom;
    ray_origin.z *= params.zoom;
}

// get the ray_origin, ray_direction, tmin and tmax
__forceinline__ __device__ void init_ray(int origin) {
    float3 ray_origin;
    get_origin_coordinate(origin, ray_origin);
    float3 ray_direction = make_float3(0.0f, 1.0f, 0.0f);
    ray_direction.y = (origin < params.adjust ? 1.f : -1.f);
    float tmin = 0.0f, tmax = 1.0f * params.ray_length[origin];
    ray_trace(ray_origin, ray_direction, tmin, tmax);
}

extern "C" __global__ void __raygen__bfs() {
    const unsigned int ix = optixGetLaunchIndex().x; // ray id
    int cur_origin = params.origins[ix];             // ray id -> origin id in spiral curve
    init_ray(cur_origin);
}

// duplicated nodes in same level, need a filter
extern "C" __global__ void __anyhit__ah() {
    const uint32_t primitive_id = optixGetPrimitiveIndex(); // triangle id
    int node_id = params.triangle_id[primitive_id];         // triangle id -> node id
    int ray_id = optixGetPayload_0();

    if (params.levels[node_id] == -1) {
        // printf("level = %d, node = %d\n", params.current_level, node_id);
        params.queue[atomicAdd(params.queue_size, 1)] = node_id;
        params.levels[node_id] = params.current_level;
    }

    optixIgnoreIntersection();
}

// extern "C" __global__ void __closesthit__ch(){}

extern "C" __global__ void __miss__ms() {}