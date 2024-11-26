#pragma once

#include <optix.h>
#include <optix_types.h>
#include <vector>

#include "model.h"

class RTBase {
protected:
    //* optix setting
    CUcontext cuda_context_ = nullptr;
    CUstream cuda_stream_;

    OptixDeviceContext optix_context_;

    OptixModule optix_module_ = nullptr;
    OptixModuleCompileOptions module_compile_options_ = {};

    OptixPipeline optix_pipeline_;
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipelineLinkOptions pipeline_link_options_ = {};

    std::vector<OptixProgramGroup> raygen_prog_groups_;
    std::vector<OptixProgramGroup> miss_prog_groups_;
    std::vector<OptixProgramGroup> hitgroup_prog_groups_;
    OptixShaderBindingTable sbt_ = {};

    OptixTraversableHandle gas_handle_{0};
    CUdeviceptr d_gas_output_memory_;

    int device_id_{0};

protected:
    void SetDevice(int device_id = 0);
    void CreateContext();
    void CreateModule();
    void CreateRaygenPrograms();
    void CreateMissPrograms();
    void CreateHitgroupPrograms();
    // void CreateProgramGroups();
    void CreatePipeline();
    void CreateSBT();
    void BuildAccel(const model_t &triangle_model, bool if_compact = false);

public:
    double bvh_building_time_{0.0};
    double bvh_compacted_time_{0.0};
    size_t bvh_memory_size_{0}; // bytes

    RTBase();
    RTBase(int device_id);
    ~RTBase();
    virtual void CountTriangles() = 0;
};