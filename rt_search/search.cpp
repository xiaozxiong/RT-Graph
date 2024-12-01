#include <stdio.h>
#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>

#include "RTBase.h"
#include "helper.h"
#include "utils.h"
#include "simple_bs.h"
// #include "common.h"

#define METHOD 0
// 0: find the index of number that is equal to or greater than target in data (float)
// 1: for measure(int)

inline void SimpleGetTriangles(std::vector<float3> &vertices,float last_val,float cur_val,int idx,const float3& axis_offsets){
    float diff=std::fabs(cur_val-last_val);
    vertices[idx*3]=make_float3(cur_val-axis_offsets.x,0.0f-axis_offsets.y,diff-axis_offsets.z);
    vertices[idx*3+1]=make_float3(cur_val-axis_offsets.x,0.0f-axis_offsets.y,-diff-axis_offsets.z);
    vertices[idx*3+2]=make_float3(last_val-axis_offsets.x,0.0f-axis_offsets.y,0.0f-axis_offsets.z);
}

inline void GetTriangles(std::vector<float3> &vertices,const float3 &center,int idx,float add){
    // float upper_x=center.x+add;
    // float below_x=center.x-add;
    float upper_y=center.y+add;
    float below_y=center.y-add;
    float upper_z=center.z+add;
    float below_z=center.z-add;
    vertices[idx*3]=make_float3(center.x,center.y,upper_z);
    vertices[idx*3+1]=make_float3(center.x,upper_y,below_z);
    vertices[idx*3+2]=make_float3(center.x,below_y,below_z);
}
//* for query
float3 SimpleCoordinateTransfer(float val,float3 axis_offsets){
    float3 new_coordinate=make_float3(1.0*val-axis_offsets.x,-0.5f-axis_offsets.y,0.0f-axis_offsets.z);
    return new_coordinate;
}

template<typename T>
void GenerateVertices(std::vector<float3> &vertices,int triangles_num,const float3 &axis_offsets,T *data){

    vertices.resize(3*triangles_num);
    for(int i=0;i<triangles_num;i++){
#if METHOD == 0
        if(i==0) SimpleGetTriangles(vertices,data[i]-100.0f,data[i],i,axis_offsets);
        else SimpleGetTriangles(vertices,data[i-1],data[i],i,axis_offsets);
#else 
        float3 center=make_float3(1.0f*data[i]-axis_offsets.x,0.0f-axis_offsets.y,0.0f-axis_offsets.z);
        GetTriangles(vertices,center,i,0.2f); // triangle size
#endif
    }
}

template<typename T>
void GenerateRays(std::vector<float3> &ray_origins,std::vector<float> &ray_lengths,int rays_num,const float3 &axis_offsets,T *query,float ray_length){

    ray_origins.resize(rays_num);
    ray_lengths.resize(rays_num);
#pragma omp parallel for
    for(int i=0;i<rays_num;i++){
#if METHOD == 0
        float3 origin=SimpleCoordinateTransfer(query[i],axis_offsets);
#else 
        float3 origin=make_float3(1.0f*query[i]-0.5f-axis_offsets.x,0.0f-axis_offsets.y,0.0f-axis_offsets.z);
#endif
        ray_origins[i]=origin;
        ray_lengths[i]=ray_length;
    }
}

template<typename T>
void InitData(T *data,int data_size,T *query,int query_size,Distribution distribution, T* data_copy){
    
    if(distribution==Distribution::uniform){
        float minn=0.0f,maxn=1.0f*data_size; //*
        UniformDistribution(data,data_size,minn,maxn);
        UniformDistribution(query,query_size,minn,maxn);
    }
    else if(distribution==Distribution::normal){
        float minn=0.0f,maxn=1.0f*data_size;
        NormalDistribution(data,data_size,minn,maxn);
        NormalDistribution(query,query_size,minn,maxn);
    }
    else if(distribution==Distribution::exponential){
        float minn=0.0f,maxn=1.0f*data_size;
        ExponentialDistribution(data,data_size,minn,maxn);
        ExponentialDistribution(query,query_size,minn,maxn);
    }
    else{
        printf("Distribution error!\n");
        exit(1);
    }
    memcpy(data_copy, data, sizeof(T) * data_size);
    // Timer timer;
    // timer.StartTiming();
    std::sort(data,data+data_size); //* sort data
    // timer.EndTiming();
    // printf("Sorting time = %.3f\n", timer.GetTime());
}

void SetDevice(int device_id=0){
    int device_count=0;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,device_id);
    size_t available_memory,total_memory;
    cudaMemGetInfo(&available_memory,&total_memory);
    // std::cout<<"==========================================================\n";
    std::cout<<"Total GPUs visible: "<<device_count;
    std::cout<<", using ["<<device_id<<"]: "<<device_prop.name<<std::endl;
    std::cout<<"Available Memory: "<<int(available_memory/1024/1024)<<" MB, ";
    std::cout<<"Total Memory: "<<int(total_memory/1024/1024)<<" MB\n";
}

template<typename T>
void RTFunction(T *data,int data_size,T *query,int query_size,int *rt_results){
#if METHOD == 0
    float3 axis_offsets=make_float3(1.0f*data_size/2,0,0);
#else
    int length=(int)sqrt(1.0*data_size);
    float3 axis_offsets=make_float3(length/2,0,length/2);
#endif
    std::vector<float3> vertices(data_size);
    std::vector<float3> ray_origins(query_size);
    std::vector<float> ray_lengths(query_size);
    
    float ray_length=1.0f;
    GenerateVertices(vertices,data_size,axis_offsets,data);
    GenerateRays(ray_origins,ray_lengths,query_size,axis_offsets,query,ray_length);

    RTBase rt_search=RTBase();
    rt_search.Setup();
    rt_search.BuildAccel(vertices); // build BVH
    rt_search.Search(query_size,ray_origins,ray_lengths,rt_results); // launch rays
    rt_search.CleanUp();

}

template<typename T>
void HostBinarySearch(T *data,int data_size,T *query,int query_size,int *host_results){
    for(int i=0;i<query_size;i++){
        auto index=std::lower_bound(data,data+data_size,query[i])-data;
        host_results[i]=(index<data_size?index:-1);
    }
}


int main(int argc,char **argv){
    if(argc!=5){
        printf("Argument error: ./rt_search data_size query_size distribution device\n");
        exit(1);
    }
    int data_size=atoi(argv[1]);
    int query_size=atoi(argv[2]);
    int distribution_kind=atoi(argv[3]);
    int use_device_id=atoi(argv[4]);
    if(distribution_kind<0||distribution_kind>2){
        printf("Distribution error!\n");
        exit(1);
    }
    //* choose distribution
    Distribution distribution;
    if(distribution_kind==0) distribution=Distribution::uniform;
    else if(distribution_kind==1) distribution=Distribution::normal;
    else if(distribution_kind==2) distribution=Distribution::exponential;

    printf("data size = %d, query size = %d, distribution = ",data_size,query_size);
    if(distribution==Distribution::uniform) printf("uniform\n");
    else if(distribution==Distribution::normal) printf("normal\n");
    else if(distribution==Distribution::exponential) printf("exponential\n");
    // initialize data
    float *data=(float*)malloc(sizeof(float)*data_size);
    float *query=(float*)malloc(sizeof(float)*query_size);
    int* rt_results=(int*)malloc(sizeof(int)*query_size);
    int* bs_results=(int*)malloc(sizeof(int)*query_size);
    int* host_results=(int*)malloc(sizeof(int)*query_size);

    float *data_copy=(float*)malloc(sizeof(float)*data_size);

    InitData(data,data_size,query,query_size,distribution, data_copy);
    // for(int i=0;i<data_size;i++) printf("%f ",data[i]);printf("\n");
    // for(int i=0;i<query_size;i++) printf("%f ",query[i]);printf("\n");
    SetDevice(use_device_id);

    HostBinarySearch(data,data_size,query,query_size,host_results);
    BSFunction(data_copy,data_size,query,query_size,bs_results);
    printf(">> Binary Search <<\n");
    Check(host_results,bs_results,query_size);
    RTFunction(data_copy,data_size,query,query_size,rt_results);
    printf(">> RT Search <<\n");
    Check(host_results,rt_results,query_size);
    // for(int i=0;i<query_size;i++) printf("%d ",host_results[i]);printf("\n");
    // for(int i=0;i<query_size;i++) printf("%d ",bs_results[i]);printf("\n");
    // for(int i=0;i<query_size;i++) printf("%d ",rt_results[i]);printf("\n");

    free(host_results);
    free(bs_results);
    free(rt_results);
    free(query);
    free(data);

    return 0;
}


