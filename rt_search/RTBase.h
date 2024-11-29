#pragma once

#include <optix_types.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
// #include <optix_host.h>

#include <vector>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "helper.h"
#include "config.h"

class RTBase{
private:
    CUcontext cuda_context_=nullptr;
    OptixDeviceContext optix_context_=0;
    OptixModule optix_module_=nullptr;
#if PRIMITIVE == 0
    float triangle_size_=0.2f;
#elif PRIMITIVE == 1
    float sphere_radius_=0.2f;
    float3 *d_centers_=nullptr;
    // OptixModule sphere_module_=nullptr;
#elif PRIMITIVE == 2
    float aabb_side_=0.2f;
    // OptixModule aabb_module_=nullptr;
    float3 *d_centers_=nullptr;
#endif 
    OptixModuleCompileOptions module_compile_options_ = {};
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipelineLinkOptions pipeline_link_options_ = {};
    OptixProgramGroup raygen_prog_group_=nullptr;
    OptixProgramGroup miss_prog_group_=nullptr;
    OptixProgramGroup hitgroup_prog_group_=nullptr;
    // optix build input buffer
    std::vector<CUdeviceptr> build_input_buffer_;
    // scene offset
    SceneParameter scene_params_;
    
public:
    CUstream cuda_stream_;
    OptixPipeline optix_pipeline_;
    OptixShaderBindingTable sbt_={};
    OptixTraversableHandle gas_handle_;
    CUdeviceptr d_gas_output_buffer_{};
    LaunchParams h_params_;
    LaunchParams *d_params_ptr_=nullptr;
    int use_device_id_{0};
    float search_time_{0.0f};
    float build_time_{0.0f};

private:
    void CreateContext();
    void CreateModule();
    void CreateProgramGroups();
    void CreatePipeline();
    void CreateSBT();
public:
    RTBase(const SceneParameter &parameter);
    ~RTBase();
    void SetDevice(int device_id);
    void Setup();
    void BuildAccel(std::vector<int> &data);
    void Search(std::vector<int> &query,int *results);
    void CleanUp();
    // void PrintInfo();
};