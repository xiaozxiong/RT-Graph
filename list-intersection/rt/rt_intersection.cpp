#include "rt_intersection.h"
#include "timer.h"
#include "common.h"
#include <string.h>

#include <optix_stubs.h>

RTInter::RTInter(uint chunk_length,int device_id): RTBase(device_id), chunk_length_(chunk_length){
    triangle_eps_=0.2f;
}

RTInter::~RTInter(){
    std::cout<<"RT based Set Intersection:\n";
    std::cout<<"BVH memory size = "<<1.0*bvh_memory_size_/1024/1024/1024<<" GB\n"; 
    std::cout<<"BVH building time = "<<bvh_building_time_<<" ms, BVH compaction time = "<<bvh_compacted_time_<<" ms,";
    std::cout<<" RT counting time = "<<rt_counting_time_<<" ms\n";
    std::cout<<"RT total time = "<<bvh_building_time_+rt_counting_time_<<" ms\n";
}

void RTInter::BuildBVHAndComputeRay(Dataset dataset){
    tile_t A={0,dataset.num_of_sets_a,dataset.num_of_sets_a}; // first
    tile_t B={dataset.num_of_sets_a,dataset.num_of_sets_a+dataset.num_of_sets_b,dataset.num_of_sets_b}; // last
    combinations_=(size_t)A.size*B.size;
    std::cout<<"combinations = "<<combinations_<<std::endl;

    axis_offset_=make_float3((float)B.size/2,(float)dataset.max_element/chunk_length_/2,(float)chunk_length_/2);
    std::cout<<"axix offset = ("<<axis_offset_.x<<", "<<axis_offset_.y<<", "<<axis_offset_.z<<")\n";
    ray_length_=B.size;
    ComputeRays(A,dataset.elements,dataset.set_offsets);
    model_t triangle_model;
    ConvertSetToRT(B,dataset.elements,dataset.set_offsets,triangle_model);
    BuildAccel(triangle_model);
}

void RTInter::ComputeRays(tile_t tile,const std::vector<uint> &elements,const std::vector<uint> &set_offsets){
    for(int i=tile.start;i<tile.end;i+=1) rays_+=set_offsets[i+1]-set_offsets[i];
    std::cout<<"rays = "<<rays_<<std::endl;
    rt_ray_origins_.resize(rays_);
    ray_set_ids_.resize(rays_);
    uint offset=0;
    for(int i=tile.start;i<tile.end;i+=1){
        uint length=set_offsets[i+1]-set_offsets[i];
        memcpy(rt_ray_origins_.data()+offset,elements.data()+set_offsets[i],sizeof(uint)*length); // z of origin
        std::fill(ray_set_ids_.data()+offset,ray_set_ids_.data()+offset+length,i-tile.start); // set id of primitive
        offset+=length;
    }
}

void RTInter::ConvertSetToRT(tile_t tile,const std::vector<uint> &elements,const std::vector<uint> &set_offsets,model_t &triangle_model){
    for(int i=tile.start;i<tile.end;i+=1){
        triangle_model.num_of_triangles+=(set_offsets[i+1]-set_offsets[i]);
    }
    triangle_model.num_of_vertices=3*triangle_model.num_of_triangles;
    triangle_model.vertices.resize(triangle_model.num_of_vertices);
    triangle_set_ids_.resize(triangle_model.num_of_triangles);
    std::cout<<"number of triangles = "<<triangle_model.num_of_triangles<<std::endl;

    uint triangle_cnt=0;
    for(int i=tile.start;i<tile.end;i+=1){
        uint start=set_offsets[i];
        uint end=set_offsets[i+1];
        float x=1.0f*(i-tile.start)-axis_offset_.x;
        for(int j=start;j<end;j+=1){
            float y=1.0f*(elements[j]/chunk_length_)-axis_offset_.y;
            float z=1.0f*(elements[j]%chunk_length_)-axis_offset_.z;
            float3 center=make_float3(x,y,z);
            CenterToTriangle(center,triangle_model.vertices,triangle_cnt*3);
            triangle_set_ids_[triangle_cnt]=(i-tile.start);
            triangle_cnt+=1;
        }
    }
}

void RTInter::CenterToTriangle(float3 center,std::vector<float3> &vertices,uint pos){
    float below_y=center.y-triangle_eps_;
    float above_y=center.y+triangle_eps_;
    float below_z=center.z-triangle_eps_;
    float above_z=center.z+triangle_eps_;
    vertices[pos]=make_float3(center.x,center.y,above_z);
    vertices[pos+1]=make_float3(center.x,below_y,below_z);
    vertices[pos+2]=make_float3(center.x,above_y,below_z);
}

void RTInter::CountIntersection(std::vector<uint> &results){
    launch_params_.handle=gas_handle_;
    launch_params_.ray_length=ray_length_;
    launch_params_.axis_offset=axis_offset_;
    launch_params_.chunk_length=chunk_length_;
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.ray_origins,sizeof(uint)*rt_ray_origins_.size()));
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.ray_set_ids,sizeof(uint)*ray_set_ids_.size()));
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.triangle_set_ids,sizeof(uint)*triangle_set_ids_.size()));
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.counts,sizeof(uint)*combinations_));

    CUDA_CHECK(cudaMemcpy(launch_params_.ray_origins,rt_ray_origins_.data(),sizeof(uint)*rt_ray_origins_.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(launch_params_.ray_set_ids,ray_set_ids_.data(),sizeof(uint)*ray_set_ids_.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(launch_params_.triangle_set_ids,triangle_set_ids_.data(),sizeof(uint)*triangle_set_ids_.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(launch_params_.counts,0,sizeof(uint)*combinations_));

    LaunchParams *d_launch_params;
    CUDA_CHECK(cudaMalloc((void**)&d_launch_params,sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(d_launch_params,&launch_params_,sizeof(LaunchParams),cudaMemcpyHostToDevice));

    GPUTimer gpu_timer;
    gpu_timer.StartTiming();
    OPTIX_CHECK(optixLaunch(
        optix_pipeline_,
        cuda_stream_,
        reinterpret_cast<CUdeviceptr>(d_launch_params),
        sizeof(LaunchParams),
        &sbt_,
        rays_,
        1,1
    ));
    CUDA_SYNC_CHECK();
    gpu_timer.StopTiming();
    rt_counting_time_=gpu_timer.GetElapsedTime();

    results.resize(combinations_);
    CUDA_CHECK(cudaMemcpy(results.data(),launch_params_.counts,sizeof(uint)*combinations_,cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_launch_params));
    CUDA_CHECK(cudaFree(launch_params_.ray_origins));
    CUDA_CHECK(cudaFree(launch_params_.ray_set_ids));
    CUDA_CHECK(cudaFree(launch_params_.triangle_set_ids));
    CUDA_CHECK(cudaFree(launch_params_.counts));
}