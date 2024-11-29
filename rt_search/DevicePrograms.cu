// #include <optix.h>
#include <cstdint>
#include <optix_device.h>

#include "LaunchParams.h"
#include "config.h"

extern "C" {
__constant__ LaunchParams params;
}

static __forceinline__ __device__ void *unpack_pointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
}

static __forceinline__ __device__ void pack_pointer(void *ptr, uint32_t &i0, uint32_t &i1) {
    const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void compute_ray(uint idx, float3 &origin, float3 &direction,
                                                   float &tmax) {
    origin = params.rays[idx].origin;
    direction = make_float3(0.0f, 1.0f, 0.0f); //
    tmax = params.rays[idx].length;
}

static __forceinline__ __device__ void trace_ray(OptixTraversableHandle handle, float3 ray_origin,
                                                 float3 ray_direction, float tmin, float tmax,
                                                 uint32_t u0, uint32_t u1) {
    optixTrace(handle, ray_origin, ray_direction, tmin, tmax, 0.0f, OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, // OPTIX_RAY_FLAG_NONE,//
               0, 0, 0,                               // SBT offset, SBT stride, missSBTIndex
               u0, u1);
}

extern "C" __global__ void __raygen__rg() {
    const uint idx = optixGetLaunchIndex().x;
    float3 ray_origin, ray_direction;
    float tmin = 0.0f, tmax = 0.0f;
    compute_ray(idx, ray_origin, ray_direction, tmax);
    uint32_t u0, u1;
    pack_pointer(params.results + idx, u0, u1);
    trace_ray(params.handle, ray_origin, ray_direction, tmin, tmax, u0, u1);
    // params.counts[idx]=count;
    // ray_direction.y*=-1.0f;
    // trace_ray(params.handle,ray_origin,ray_direction,tmin,tmax,count);
    // params.counts[idx]+=count;
}

// extern "C" __global__ void __anyhit__ah(){
//     printf("hit\n");
// }

extern "C" __global__ void __miss__ms() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    int *res = reinterpret_cast<int *>(unpack_pointer(u0, u1));
    *res = -1;
}

extern "C" __global__ void __closesthit__ch() {
    unsigned int prim_id = optixGetPrimitiveIndex();
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    int *res = reinterpret_cast<int *>(unpack_pointer(u0, u1));
    *res = (int)prim_id;
}

// TODO: aabb intersection program
__forceinline__ __device__ bool operator>(const float3 a, const float3 b) {
    return (a.x > b.x && a.y > b.y && a.z > b.z);
}

__forceinline__ __device__ bool operator<(const float3 a, const float3 b) {
    return (a.x < b.x && a.y < b.y && a.z < b.z);
}

__forceinline__ __device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ float3 operator+(const float3 &a, const float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__forceinline__ __device__ float3 operator-(const float3 &a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}

#if PRIMITIVE == 1
/** dot product */
__forceinline__ __device__ float dot(const float3 &a, const float3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

extern "C" __global__ void __intersection__sphere() {
    unsigned int prim_id = optixGetPrimitiveIndex();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 center = params.centers[prim_id];
    const float3 dist = ray_origin - center;
    float sq_dist = dot(dist, dist);

    if (sq_dist < params.side * params.side) {
        optixReportIntersection(0, 0); //
    }
}
#elif PRIMITIVE == 2
extern "C" __global__ void __intersection__aabb() {
    unsigned int prim_id = optixGetPrimitiveIndex();
    const float3 ray_origin = optixGetWorldRayOrigin();
    const float3 center = params.centers[prim_id];
    float3 top_right = center + params.side;
    float3 bottom_left = center - params.side;
    /*
    bool optixReportIntersection (float hitT, unsigned int hitKind);
    If optixGetRayTmin() <= hitT <= optixGetRayTmax(),
    the any hit program associated with this intersection program (via the SBT entry) is called.
    */
    if (ray_origin > bottom_left && ray_origin < top_right) {
        optixReportIntersection(0, 0); //
    }
}
#endif