#include <optix_device.h>
#include "launch_params.h"

extern "C" {
    __constant__  LaunchParams params;
}

static __forceinline__ __device__ void trace_ray(
    OptixTraversableHandle handle,
    float3 ray_origin,
    float3 ray_direction,
    float tmin,
    float tmax,
    uint a_set_id
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
        a_set_id
    );

}

extern "C" __global__ void __raygen__tc(){    
    const uint ray_id=optixGetLaunchIndex().x;
    const uint a_set_id=params.ray_set_ids[ray_id];
    const float x=-0.5f-params.axis_offset.x;
    const float y=params.ray_origins[ray_id]/params.chunk_length-params.axis_offset.y;
    const float z=params.ray_origins[ray_id]%params.chunk_length-params.axis_offset.z;
    float ray_length=params.ray_length;
    float3 ray_direction=make_float3(1.0f,0.0f,0.0f);
    float3 ray_origin=make_float3(x,y,z);
    trace_ray(params.handle,ray_origin,ray_direction,0.0f,ray_length,a_set_id);
}

extern "C" __global__ void __anyhit__ah(){
    const uint triangle_id=optixGetPrimitiveIndex();
    const uint a_set_id=optixGetPayload_0();
    const uint b_set_id=params.triangle_set_ids[triangle_id];
    const uint count_offset=a_set_id*params.ray_length+b_set_id;
    atomicAdd(params.counts+count_offset,1U);
    // atomicAdd(params.counts,1U);
    optixIgnoreIntersection();
}

// extern "C" __global__ void __closesthit__ch(){

// }

extern "C" __global__ void __miss__ms(){
    
}

