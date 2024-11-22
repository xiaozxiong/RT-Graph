#include "launch_params.h"
#include <cuda_runtime.h>
#include <math.h>
#include <optix_device.h>
#include <stdint.h>

extern "C" {
__constant__ LaunchParamsV2 params;
}

__forceinline__ __device__ void ray_trace(float3 &ray_origin, float3 &ray_direction, float &tmin,
                                          float &tmax) {

    // origin of ray = triangle center
    unsigned int rx = __float_as_uint(ray_origin.x);
    unsigned int ry = __float_as_uint(ray_origin.y);
    unsigned int rz = __float_as_uint(ray_origin.z);

    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE, //|OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
               0, 0, 0, rx, ry, rz);
    // if(p0!=params.neighbor_size[curr_vertex]){
    //     printf("triangle error: %d , p0 = %d, neighbor size = %d, tmin = %f,
    //     tmax =
    //     %f\n",params.vir_real[curr_vertex],p0,params.neighbor_size[curr_vertex],tmin,tmax);
    // }
}

__forceinline__ __device__ void get_origin_coordinate(int origin, float3 &ray_origin) {
    // int t = origin;
    ray_origin.y = 0.f; //!
    if (origin >= params.adjust) {
        origin -= params.adjust;
        ray_origin.y = -2.f; //!
    }
    if (origin == 0) {
        ray_origin.x = 0.0f;
        ray_origin.z = 0.0f;
        // if(t == 22541572) printf("#1 => (%f, %f, %f)\n", ray_origin.x,
        // ray_origin.y, ray_origin.z);
    } else {
        int side = (int)sqrt(1.0 * origin); //! if origin is too large, float is wrong
        side += (side % 2 == 0 ? 1 : 2);
        int res = side * side - 1 - origin;
        // if(t == 22541572) printf("side = %d, res = %d\n", side, res);
        int temp = res / (side - 1);
        int axis = (int)(side / 2);
        if (temp == 0) {
            ray_origin.x = (axis - res);
            ray_origin.z = (-axis);
        } else if (temp == 1) {
            ray_origin.x = (-axis);
            ray_origin.z = (-axis + res - (side - 1));
        } else if (temp == 2) {
            ray_origin.x = (-axis + res - 2 * (side - 1));
            ray_origin.z = (axis);
        } else if (temp == 3) {
            ray_origin.x = axis;
            ray_origin.z = (axis - (res - 3 * (side - 1)));
        }
        // if(t == 22541572) printf("#2 => (%f, %f, %f), temp = %d\n", ray_origin.x,
        // ray_origin.y, ray_origin.z, temp);
    }
}

// get the ray_origin, ray_direction, tmin and tmax
__forceinline__ __device__ void init_ray(int origin) {
    float3 ray_origin;
    get_origin_coordinate(origin, ray_origin);
    // if(origin == 22541572) {
    //     printf("no zoom origin: (%f, %f, %f)\n", ray_origin.x, ray_origin.y,
    //     ray_origin.z);
    // }
    ray_origin.x *= params.zoom.x;
    ray_origin.y *= params.zoom.y;
    ray_origin.z *= params.zoom.z;

    float3 ray_direction = make_float3(0.0f, 1.0f, 0.0f);
    ray_direction.y = 1.f;                           //(origin < params.adjust ? 1.f : -1.f); //!
    float tmin = 0.0f, tmax = params.max_ray_length; //!
    // if(origin == 22541572) {
    //     printf("origin id = %d, ray origin = (%f, %f, %f), length = %f, adjust
    //     = %d\n", origin, ray_origin.x, ray_origin.y, ray_origin.z, tmax,
    //     params.adjust);
    // }
    ray_trace(ray_origin, ray_direction, tmin, tmax);
}

extern "C" __global__ void __raygen__bfs() {
    unsigned int ix = optixGetLaunchIndex().x; // ray id
    int origin_id = params.origins[ix];        // ray id -> id in spiral curve
    // printf("ix = %d: origin id = %d\n", ix, origin_id);
    init_ray(origin_id);
}

// TODO: diff -> node id
__forceinline__ __device__ int decode(float diff) {
    int res = round(diff * params.encode_mod);
    return res;
}

// TODO: diff -> node id
__forceinline__ __device__ void visit(int node) {
    if ((node >= 0 && node < params.nodes) && params.levels[node] == -1) {
        // if(params.current_level == 1490) printf("1490: node = %d\n", node);
        // if(params.current_level == 1491) printf("1491: node = %d\n", node);
        // if(params.current_level == 1492) printf("1492: node = %d\n", node);
        params.queue[atomicAdd(params.queue_size, 1)] = node;
        params.levels[node] = params.current_level;
    }
}

// duplicated nodes in same level, need a filter
extern "C" __global__ void __anyhit__ah() {
    OptixTraversableHandle gas = optixGetGASTraversableHandle();
    unsigned int prim_idx = optixGetPrimitiveIndex();
    unsigned int sbt_idx = optixGetSbtGASIndex();
    float time = optixGetRayTime();

    float3 data[3];
    optixGetTriangleVertexData(gas, prim_idx, sbt_idx, time, data); //* read

    // float3 center = params.triangle_center[prim_idx]; //* read
    float3 center;
    center.x = __uint_as_float(optixGetPayload_0());
    center.y = __uint_as_float(optixGetPayload_1());
    center.z = __uint_as_float(optixGetPayload_2());

    float adjs[8];
    //* ===== decode =====
    adjs[0] = fabsf(data[0].y - center.y);
    adjs[1] = fabsf(data[0].z - center.z);
    adjs[2] = fabsf(data[1].x - center.x);
    adjs[3] = fabsf(data[1].y - center.y);
    adjs[4] = fabsf(data[1].z - center.z);
    adjs[5] = fabsf(data[2].x - center.x);
    adjs[6] = fabsf(data[2].y - center.y);
    adjs[7] = fabsf(data[2].z - center.z);

    // printf("%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f\n", adjs[0],
    // adjs[1], adjs[2], adjs[3], adjs[4], adjs[5], adjs[6], adjs[7]); printf("%d,
    // %d, %d, %d, %d, %d, %d, %d\n", decode(adjs[0]), decode(adjs[1]),
    // decode(adjs[2]), decode(adjs[3]), decode(adjs[4]), decode(adjs[5]),
    // decode(adjs[6]), decode(adjs[7]));

    int id_mod = params.encode_mod / 10;
    int node = -1, last_flag = -1;
    for (int i = 0; i < 8; i++) {
        // null: 1.0, must be in the end
        if (adjs[i] >= 1.0f) {
            if (node != -1)
                visit(node);
            break;
        }

        int tmp = decode(adjs[i]);
        // obtain flag
        int flag = tmp / id_mod; // >= 0
        int id = tmp % id_mod;
        if (flag == 0) {
            // if(node != -1) visit(node);
            visit(id);
        } else {
            if (flag == last_flag) {
                node = node * id_mod + id;
            } else {
                if (node != -1)
                    visit(node);
                node = id;
                last_flag = flag;
            }
            // the last digit may be null
            if (i == 7 && node != -1)
                visit(node);
        }
    }

    optixIgnoreIntersection();
}

// extern "C" __global__ void __closesthit__ch(){}

extern "C" __global__ void __miss__ms() {}