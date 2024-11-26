#include <optix_device.h>
#include "launch_params.h"
#include "config.h"

extern "C" {
    __constant__ 
#if RTTC_METHOD == 0 
    ListIntersectionParams
#elif RTTC_METHOD == 1
    HashmapParams
#else  
    LaunchParams
#endif
    params;
}

#if RTTC_METHOD == 0
static __forceinline__ __device__ void trace_ray(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax,
    uint &cnt
){
    cnt=0;
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,0,0, // SBT offset, SBT stride, missSBTIndex
        cnt
    );

}
#elif RTTC_METHOD == 1
static __forceinline__ __device__ void trace_ray(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax,
    uint add_val=1
){
    optixTrace(
        handle,
        ray_origin,
        ray_direction,
        tmin,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,0,0, // SBT offset, SBT stride, missSBTIndex
        add_val
    );

}
#endif

extern "C" __global__ void __raygen__tc(){    
    const uint ray_id=optixGetLaunchIndex().x;
#if RTTC_METHOD == 0
//* ==================================
    float3 ray_origin;
    float3 ray_direction=make_float3(1.0f,0.0f,0.0f);
    float tmin=0.0f;
#ifdef LIST_OPTIMIZATION
// =====
    // printf("%f, %f, %f\n", params.axis_offset.x, params.axis_offset.y, params.axis_offset.z);
    ray_origin.x = -0.5f - params.axis_offset.x;
    ray_origin.y = params.ray_origin[ray_id].x - params.axis_offset.y;
    ray_origin.z = params.ray_origin[ray_id].y - params.axis_offset.z;
    float tmax=1.0f;
#else
// =====
    ray_origin.x=-0.5f-params.axis_offset.x;
    ray_origin.z=1.0f*params.ray_origin[ray_id]-params.axis_offset.z;
    uint l=0,r=params.nodes;
    while(l<r){
        int mid=(l+r+1)>>1;
        if(params.ray_offset[mid]<=ray_id) l=mid;
        else r=mid-1;
    }
    uint node=l;
    ray_origin.y=1.0f*l-params.axis_offset.y;
    float tmax=1.0f*params.ray_length[node];
#endif
// ====
    uint cnt=0;
    trace_ray(params.handle,ray_origin,ray_direction,tmin,tmax,cnt);
#if REDUCE == 1
    params.count[ray_id]=cnt;
#endif

#elif RTTC_METHOD == 1
//* ==================================
    float3 ray_direction=make_float3(0.0f,1.0f,0.0f),ray_origin;
    float tmin=0.0f,tmax=0.2f;
    for(int i=ray_id;i<params.total_load;i+=params.rays){
        uint id=i*3;
        float x=1.0f*params.ray_origin_and_val[id]-params.axis_offset.x;
        float y=-0.1f-params.axis_offset.y;
        float z=1.0f*params.ray_origin_and_val[id+1]-params.axis_offset.z;
        uint add_val=params.ray_origin_and_val[id+2];
        ray_origin=make_float3(x,y,z);
        trace_ray(params.handle,ray_origin,ray_direction,tmin,tmax,add_val);
    }
#endif
}

extern "C" __global__ void __anyhit__ah(){

#if RTTC_METHOD == 0
#ifdef LIST_OPTIMIZATION
// =====
    const uint primitive_id=optixGetPrimitiveIndex();
    uint add_val = params.triangle_vals[primitive_id];
    // printf("add_val = %u\n", add_val);
#if REDUCE == 1
    uint cnt=optixGetPayload_0();
    optixSetPayload_0(cnt+add_val);
#elif REDUCE == 0
    atomicAdd(params.count,add_val);
#endif

#else
// =====
#if REDUCE == 1
    uint cnt=optixGetPayload_0();
    optixSetPayload_0(cnt+1);
#elif REDUCE == 0
    atomicAdd(params.count,1);
#endif
    optixIgnoreIntersection();
#endif
    

#elif RTTC_METHOD == 1
    const uint primitive_id=optixGetPrimitiveIndex();
    const uint add_val=optixGetPayload_0();
    #if REDUCE == 1
        atomicAdd(params.count+primitive_id,add_val);
    #else
        atomicAdd(params.count,add_val);
    #endif
#endif

}

// extern "C" __global__ void __closesthit__ch(){

// }

extern "C" __global__ void __miss__ms(){
// #if RTTC_METHOD == 1
//     atomicAdd(params.miss_record, 1);
// #endif
}

