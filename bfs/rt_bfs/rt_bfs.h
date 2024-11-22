#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_types.h>
#include <vector>

#include "common.h"
#include "graph.h"
#include "launch_params.h"
#include "timer.h"

#define MB(x) (1ULL * (x)*1024 * 1024)
#define LAUNCH_LIMIT (2U << 29) //((2U<<30)-(2U<<29))
#define MAX_PRIMITIVES (2 << 29)

class RTBFS {
private:
    // TODO: optix setting
    CUcontext cuda_context_ = nullptr;
    OptixDeviceContext optix_context_ = 0;
    OptixModule optix_module_ = nullptr;
    OptixModuleCompileOptions module_compile_options_ = {};
    OptixPipelineCompileOptions pipeline_compile_options_ = {};
    OptixPipelineLinkOptions pipeline_link_options_ = {};
    OptixProgramGroup raygen_prog_group_ = nullptr;
    OptixProgramGroup miss_prog_group_ = nullptr;
    OptixProgramGroup hitgroup_prog_group_ = nullptr;

    CUstream cuda_stream_;
    OptixPipeline optix_pipeline_;
    OptixShaderBindingTable sbt_ = {};
    OptixTraversableHandle gas_handle_{0};
    CUdeviceptr d_gas_output_memory_;

    LaunchParams h_params_;
    LaunchParams *d_params_ptr_ = nullptr;
    int use_device_id_{0};
    // TODO: graph infromation and rt setting
    //  graph:
    graph_info_t graph_info_;
    int *offsets_;
    int *adjs_;

    int chunk_length_{0};
    int chunks_{0};            // total number of chunks
    int adjust_{0};            // half of chunks
    float triangle_eps_{0.1f}; // 0.1f
    float zoom_{1.f};          //! to determine the interval of triangles
    int number_of_triangle_{0};
    int number_of_origin_{0};
    // std::vector<int> origins_;
    std::vector<int> origin_offset_;   // node id -> ray id
    std::vector<int> ray_lengths_;     // length of ray
    float3 *device_vertices_{nullptr}; // triangle array on device
    int *device_triangle_id_{nullptr}; // primitive id -> node id
    std::vector<int> levels_;
    // measuring time
    CPUTimer cpu_timer_;
    float optix_setup_time_{0.0f};
    float bvh_building_time_{0.0f};
    float traversal_time_{0.0f};
    float total_time_{0.0f};

private:
    // optix
    void CreateContext();
    void CreateModule();
    void CreateProgramGroups();
    void CreatePipeline();
    void CreateSBT();
    // bfs
    // int ChooseLayerByNodes();
    void GenerateTrianglesOnCPU();
    std::pair<float, float> ConvertIdToCoordinate(int origin_id);
    void GetTriangleVertices(float3 center, int idx, std::vector<float3> &triangle_vertices);
    void TransferVerticesToDevice(std::vector<float3> &triangle_vertices,
                                  std::vector<int> &triangle_id);

    void FreeLaunchParamsMemory();
    void CleanUp();

    void TraversalOnCPU(std::vector<int> &levels, int source_node = 0);

public:
    RTBFS(Graph &graph, int chunck_length);
    ~RTBFS();
    void SetDevice(int device_id = 0);
    void OptiXSetup();
    // void PrepareRT();
    void BuildAccel(bool if_compact = false);
    void Traversal(int source_node = 0);
    void CheckResult();
    void PrintResult(int head = 40);
};