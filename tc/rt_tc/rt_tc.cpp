#include "rt_tc.h"
#include "timer.h"
#include "cuda_helper.h"
#include "common.h"

#include <optix_stubs.h>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cassert>

RTTC::RTTC(int device_id): RTBase(device_id){
#if RTTC_METHOD == 0
    printf("RTTC: Using list intersecton based method (1A2)\n");
#elif RTTC_METHOD == 1
    printf("RTTC: Using hashmap based method (2A1)\n");
#else 
    printf("RTTC: Using wrong method\n");
#endif
}

RTTC::~RTTC(){

}

void RTTC::BuildBVHAndComputeRay(const int *adjs,const int *offsets,const graph_info_t &graph_info,bool if_compact){
    //* build BVH
    model_t triangle_model; // RT triangles
    ConvertGraphToRT(adjs,offsets,graph_info,triangle_model);
    BuildAccel(triangle_model,if_compact);
    //* compute ray
    ComputeRays(adjs,offsets,graph_info);
}

void RTTC::CountTriangles(){
    SetupLaunchParameters();
#if RTTC_METHOD == 0
    using param_t=ListIntersectionParams;
#elif RTTC_METHOD == 1
    using param_t=HashmapParams;
#else 
    using param_t=LaunchParams;
#endif
    param_t *d_launch_params;
    CUDA_CHECK(cudaMalloc((void**)&d_launch_params,sizeof(param_t)));
    CUDA_CHECK(cudaMemcpy(d_launch_params,&launch_params_,sizeof(param_t),cudaMemcpyHostToDevice));
    printf("RTTC: Launch size = %u\n",rt_launch_size_);
    GPUTimer gpu_timer;
    gpu_timer.StartTiming();
    OPTIX_CHECK(optixLaunch(
        optix_pipeline_,
        cuda_stream_,
        reinterpret_cast<CUdeviceptr>(d_launch_params),
        sizeof(param_t),
        &sbt_,
        rt_launch_size_,
        1,1
    ));
    CUDA_SYNC_CHECK();
    gpu_timer.StopTiming();
    counting_time_=gpu_timer.GetElapsedTime();
    std::cout<<"RTTC: Trace time = "<<counting_time_<<" ms";

#if REDUCE == 1
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    total_triangles_=ThrustReduce(launch_params_.count,count_size_);
    cpu_timer.StopTiming();
    double reduce_time=cpu_timer.GetElapsedTime();
    counting_time_+=reduce_time;
    std::cout<<", reduce time = "<<reduce_time<<" ms";

    //* to count miss
    // uint miss_record = ThrustCount(launch_params_.count,count_size_, 0);
    // printf("#---> miss = %u\n", miss_record);

#else
    CUDA_CHECK(cudaMemcpy(&total_triangles_,launch_params_.count,sizeof(uint),cudaMemcpyDeviceToHost));
#endif

#if RTTC_METHOD == 1
    //* to count miss
    uint miss_record;
    CUDA_CHECK(cudaMemcpy(&miss_record, launch_params_.miss_record, sizeof(uint), cudaMemcpyDeviceToHost));
    printf("\n#---> miss = %u\n", miss_record);
#endif

    std::cout<<"\n";
    CUDA_CHECK(cudaFree(d_launch_params));
    FreeLaunchParameters();
}

void RTTC::ConvertGraphToRT(const int *adjs,const int *offsets,const graph_info_t &graph_info,model_t &model){
    //* model.vertices stores vertice coordinate of triangles
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    
#if RTTC_METHOD == 0
#ifndef LIST_OPTIMIZATION
    //* compute the number of RT triangles
    int max_adj_len=0;
    for(int i = 0; i < graph_info.nodes; i += 1){
        // if(i % 1000 == 0) printf("i = %d\n", i);
        int start = offsets[i];
        int end = offsets[i+1];
        uint two_adj_size = 0;

// #pragma omp parallel for reduction(+ : two_adj_size)
        for(int j = start; j < end; j += 1){
            int adj = adjs[j];
            two_adj_size += offsets[adj+1] - offsets[adj];
        }

        max_adj_len=std::max(end-start,max_adj_len);
        rt_triangles_+=two_adj_size;
    }
    model.vertices.resize(rt_triangles_*3);
    axis_offset_=make_float3(max_adj_len/2,graph_info.nodes/2,graph_info.nodes/2); //
    // printf("-------------------\n");
    MethodBasedOnListIntersection(adjs,offsets,graph_info,model);

#else
    uint max_two_hop=0;
#pragma omp parallel for reduction(max: max_two_hop)
    for(int i=0;i<graph_info.nodes;i+=1){
        int start=offsets[i];
        int end=offsets[i+1];
        uint tmp_size=0;
        for(int j=start;j<end;j+=1){
            int adj=adjs[j];
            tmp_size+=(offsets[adj+1]-offsets[adj]);
        }
        max_two_hop=std::max(max_two_hop,tmp_size);
    }
    axis_offset_=make_float3(0.f,graph_info.nodes/2,graph_info.nodes/2); //

    std::vector<float3> temp_vertices;
    uint total_num = 0; // total two-hops neighbors
    int *two_hop_adjs=(int*)malloc(sizeof(int)*max_two_hop);
    for(int i=0;i<graph_info.nodes;i+=1){
        int start=offsets[i];
        int end=offsets[i+1];
        int offset=0;
        for(int j=start;j<end;j+=1){
            int adj=adjs[j];
            int len=offsets[adj+1]-offsets[adj];
            memcpy(two_hop_adjs+offset,adjs+offsets[adj],sizeof(int)*len);
            offset+=len;
        }
        uint two_adj_size=offset;
        total_num+=two_adj_size;
        if(two_adj_size==0) continue;
        // ThrustSort(two_hop_adjs,two_adj_size);
        std::sort(two_hop_adjs,two_hop_adjs+two_adj_size);
        int last_node=-1,cnt=0;
        for(int j=0;j<two_adj_size;j+=1){
            int cur_node=two_hop_adjs[j];
            if(last_node==cur_node) cnt+=1;
            else{
                if(last_node!=-1){
                    temp_vertices.push_back(make_float3(0.f, i, last_node));
                    triangle_vals_.push_back(cnt);
                }
                cnt=1;
            }
            last_node=cur_node;
        }
        if(last_node!=-1){
            temp_vertices.push_back(make_float3(0.f, i, last_node));
            triangle_vals_.push_back(cnt);
        }
    }
    free(two_hop_adjs);
    // uint sum = 0;
    // for(int i=0; i<triangle_vals_.size(); i++){
    //     sum += triangle_vals_[i];
    // }
    // assert(sum = total_num);
    // printf("sum = %u\n", sum);
    rt_triangles_ = triangle_vals_.size();
    model.vertices.resize(rt_triangles_*3);
    printf("Triangle vertices memory size = %.3f GB\n", 3.0 * rt_triangles_ * sizeof(float3) / 1024/1024/1024);

    // ExpandonGPU(temp_vertices.data(), model.vertices.data(),  rt_triangles_, rt_triangle_eps_, axis_offset_);
    ExpandonCPU(temp_vertices.data(), model.vertices.data(),  rt_triangles_, rt_triangle_eps_, axis_offset_);
    printf("RTTC: original triangles = %u, retained triangles = %u\n", total_num, rt_triangles_);
#endif

#elif RTTC_METHOD == 1
    rt_triangles_=graph_info.edges;
    model.vertices.resize(rt_triangles_*3);
    axis_offset_=make_float3(graph_info.nodes/2,0,graph_info.nodes/2); //
    MethodBasedOnHashMap(adjs,offsets,graph_info,model);
    std::cout<<"RTTC: number of RT triangles = "<<rt_triangles_<<"\n";
#else
    std::cerr<<"RT method error"<<std::endl;
    exit(EXIT_FAILURE);
#endif
    cpu_timer.StopTiming();
    convert_time_=cpu_timer.GetElapsedTime();

    std::cout<<"RTTC: time of converting graph to RT = "<<convert_time_<<" ms\n";
}

#if RTTC_METHOD == 0
void RTTC::MethodBasedOnListIntersection(const int *adjs,const int *offsets,const graph_info_t &graph_info,model_t &model){
    ConvertOnGPU(adjs,offsets,graph_info,axis_offset_,rt_triangle_eps_,model.vertices.data(),rt_triangles_*3);
}
#endif

#if RTTC_METHOD == 1
void RTTC::MethodBasedOnHashMap(const int *adjs,const int *offsets,const graph_info_t &graph_info,model_t &model){
    ConvertOnGPU(adjs,offsets,graph_info,axis_offset_,rt_triangle_eps_,model.vertices.data(),rt_triangles_*3);
}
#endif
//* Takes up most of the time
void RTTC::ComputeRays(const int *adjs,const int *offsets,const graph_info_t &graph_info){
    std::cout<<"Start computing rays\n";
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
#if RTTC_METHOD == 0
#ifdef LIST_OPTIMIZATION
    for(int i = 0; i < graph_info.nodes; i++){
        int start=offsets[i];
        int end=offsets[i+1];
        for(int j = start; j < end; j++){
            rt_ray_origin_.push_back(make_float2(i, adjs[j]));
        }
    }
#else
    // number of rays = number of edges
    rt_ray_offset_.resize(graph_info.nodes+1);
    memcpy(rt_ray_offset_.data(),offsets,sizeof(int)*(graph_info.nodes+1));
    rt_ray_origin_.resize(graph_info.edges);
    memcpy(rt_ray_origin_.data(),adjs,sizeof(int)*graph_info.edges);
    rt_ray_length_.resize(graph_info.nodes);
    for(int i=0;i<graph_info.nodes;i+=1){
        rt_ray_length_[i]=offsets[i+1]-offsets[i];
    }
#endif
    std:: cout << "number of rays = "<<rt_ray_origin_.size()<<"\n";
#elif RTTC_METHOD == 1
    // number of rays = the sum of the number of two hop adj of each node
    // compute max two hop size
    //TODO: GPU
    // TwoHopsFunction(adjs, offsets, graph_info, rt_ray_origin_); 
    //TODO: CPU
    
    uint max_two_hop=0;
#pragma omp parallel for reduction(max: max_two_hop)
    for(int i=0;i<graph_info.nodes;i+=1){
        int start=offsets[i];
        int end=offsets[i+1];
        uint tmp_size=0;
        for(int j=start;j<end;j+=1){
            int adj=adjs[j];
            tmp_size+=(offsets[adj+1]-offsets[adj]);
        }
        max_two_hop=std::max(max_two_hop,tmp_size);
    }

    // printf("Host two size = %u\n",max_two_hop);
    size_t total_num=0;
    int *two_hop_adjs=(int*)malloc(sizeof(int)*max_two_hop);
    for(int i=0;i<graph_info.nodes;i+=1){
        int start=offsets[i];
        int end=offsets[i+1];
        int offset=0;
        for(int j=start;j<end;j+=1){
            int adj=adjs[j];
            int len=offsets[adj+1]-offsets[adj];
            memcpy(two_hop_adjs+offset,adjs+offsets[adj],sizeof(int)*len);
            offset+=len;
        }
        uint two_adj_size=offset;
        total_num+=two_adj_size;
        if(two_adj_size==0) continue;
        // ThrustSort(two_hop_adjs,two_adj_size);
        std::sort(two_hop_adjs,two_hop_adjs+two_adj_size);
        int last_node=-1,cnt=0;
        for(int j=0;j<two_adj_size;j+=1){
            int cur_node=two_hop_adjs[j];
            if(last_node==cur_node) cnt+=1;
            else{
                if(last_node!=-1){
                    rt_ray_origin_.push_back(i);
                    rt_ray_origin_.push_back(last_node);
                    rt_ray_origin_.push_back(cnt);
                }
                cnt=1;
            }
            last_node=cur_node;
        }
        if(last_node!=-1){
            rt_ray_origin_.push_back(i);
            rt_ray_origin_.push_back(last_node);
            rt_ray_origin_.push_back(cnt);
        }
    }
    free(two_hop_adjs);
    printf("total two hops (duplicate) = %llu, ",total_num);
    printf("number of rays = %llu, rate = %.3f\n",rt_ray_origin_.size()/3, 1.0*rt_ray_origin_.size()/3/total_num);
    
#else 
    std::cerr<<"RT method error"<<std::endl;
    exit(EXIT_FAILURE);
#endif
    cpu_timer.StopTiming();
    std::cout<<"RTTC: Time of computing rays = "<<cpu_timer.GetElapsedTime()<<" ms\n";
}

void RTTC::SetupLaunchParameters(){
#if RTTC_METHOD == 0
    rt_launch_size_=rt_ray_origin_.size();
#ifdef LIST_OPTIMIZATION
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.ray_origin,sizeof(float2)*rt_ray_origin_.size()));
    CUDA_CHECK(cudaMemcpy(launch_params_.ray_origin,rt_ray_origin_.data(),sizeof(float2)*rt_ray_origin_.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.triangle_vals,sizeof(uint)*triangle_vals_.size()));
    CUDA_CHECK(cudaMemcpy(launch_params_.triangle_vals,triangle_vals_.data(),sizeof(uint)*triangle_vals_.size(),cudaMemcpyHostToDevice));
#else
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.ray_origin,sizeof(uint)*rt_ray_origin_.size()));
    CUDA_CHECK(cudaMemcpy(launch_params_.ray_origin,rt_ray_origin_.data(),sizeof(uint)*rt_ray_origin_.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.ray_offset,sizeof(uint)*rt_ray_offset_.size()));
    CUDA_CHECK(cudaMemcpy(launch_params_.ray_offset,rt_ray_offset_.data(),sizeof(uint)*rt_ray_offset_.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.ray_length,sizeof(uint)*rt_ray_length_.size()));
    CUDA_CHECK(cudaMemcpy(launch_params_.ray_length,rt_ray_length_.data(),sizeof(uint)*rt_ray_length_.size(),cudaMemcpyHostToDevice));
    launch_params_.nodes=rt_ray_length_.size();
#endif
#if REDUCE == 1
    // for each node:
    // count_size_=rt_ray_length_.size();
    // for each ray:
    count_size_=rt_launch_size_;
#else
    count_size_=1;
#endif
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.count,sizeof(uint)*count_size_)); // for each node
    CUDA_CHECK(cudaMemset(launch_params_.count,0,sizeof(uint)*count_size_));
    launch_params_.handle=gas_handle_;
    launch_params_.axis_offset=axis_offset_;
// =============================================================
#elif RTTC_METHOD == 1
// =================================================================
    rt_launch_size_=rt_ray_origin_.size()/3;
    assert(rt_launch_size_>0);
    launch_params_.total_load=rt_launch_size_;
    rt_launch_size_=(rt_launch_size_>max_rays_?max_rays_:rt_launch_size_); //* real launch size
    launch_params_.rays=rt_launch_size_;
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.ray_origin_and_val,sizeof(uint)*rt_ray_origin_.size()));
    CUDA_CHECK(cudaMemcpy(launch_params_.ray_origin_and_val,rt_ray_origin_.data(),sizeof(uint)*rt_ray_origin_.size(),cudaMemcpyHostToDevice));
#if REDUCE == 1
    count_size_=rt_triangles_;
#else
    count_size_=1;
#endif
    // printf("count size = %d\n",count_size_);
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.count,sizeof(uint)*count_size_)); // for each node
    CUDA_CHECK(cudaMemset(launch_params_.count,0,sizeof(uint)*count_size_));
    launch_params_.handle=gas_handle_;
    launch_params_.axis_offset=axis_offset_;

    //* to count miss
    CUDA_CHECK(cudaMalloc((void**)&launch_params_.miss_record,sizeof(uint)));
    CUDA_CHECK(cudaMemset(launch_params_.miss_record,0,sizeof(uint)));
    
#else 
    std::cerr<<"RT method error"<<std::endl;
    exit(EXIT_FAILURE);
#endif
}

void RTTC::FreeLaunchParameters(){
#if RTTC_METHOD == 0
    CUDA_CHECK(cudaFree(launch_params_.count));
#ifdef LIST_OPTIMIZATION
    CUDA_CHECK(cudaFree(launch_params_.triangle_vals));
    CUDA_CHECK(cudaFree(launch_params_.ray_origin));
#else 
    CUDA_CHECK(cudaFree(launch_params_.ray_offset));
    CUDA_CHECK(cudaFree(launch_params_.ray_origin));
    CUDA_CHECK(cudaFree(launch_params_.ray_length));
#endif
#elif RTTC_METHOD == 1
    CUDA_CHECK(cudaFree(launch_params_.count));
    // CUDA_CHECK(cudaFree(launch_params_.ray_offset));
    CUDA_CHECK(cudaFree(launch_params_.ray_origin_and_val));
#endif
}