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

typedef struct Triangle {
    float3 center;
    float encode_data[8];
    int pos{0};         // current position in encode data
    float mul_cnt{0.f}; // cnt node encoded with multiple vertex, + 0.1f
    float3 vertices[3];
    int node_count{0}; // nodes per triangle

    Triangle(float x, float y, float z) {
        center = make_float3(x, y, z);
        pos = 0;
        mul_cnt = 0;
        node_count = 0;
        std::fill(encode_data, encode_data + 8, 1.0f);
    }

    Triangle(float3 _center) : center(_center) {
        pos = 0;
        mul_cnt = 0;
        node_count = 0;
        std::fill(encode_data, encode_data + 8, 1.0f);
    }
} tri_t;

class RTBFS_V2 {
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

    LaunchParamsV2 h_params_;
    LaunchParamsV2 *d_params_ptr_ = nullptr;
    int use_device_id_{0};

    // TODO: graph infromation and rt setting
    //  graph:
    graph_info_t graph_info_;
    int *offsets_;
    int *adjs_;

    const int nodes_per_triangle_ = 8;
    int chunk_length_{0};
    int chunks_{0}; // total number of chunks
    int adjust_{0}; // half of chunks
    // int scale_factor_;
    int max_digits_{0};
    int encode_digits_{0}; //* 1 + id_digits_
    int id_digits_{0};
    int encode_mod_{0};
    int id_mod_{0};
    int encoded_nodes_{0}; // for calculate the avg. nodes per triangle
    float scaler;
    float3 zoom_; //! to determine the interval of triangles: 2: 1: 2

    int number_of_triangle_{0};
    int number_of_origin_{0};
    // std::vector<int> origins_;
    std::vector<int> origin_offset_; // node id -> ray id
    // std::vector<int> ray_lengths_; // length of ray

    float3 *d_vertices_{nullptr}; // triangle array on device
    float3 *d_centers_{nullptr};  // primitive id -> node id
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

    void GenerateTrianglesOnCPU();

    std::pair<int, int> IdtoCoordinateinCurve(int id);

    int NumberOfDigits(int num);

    float EncodeNodeId(int id);

    void ZoomCoordinate(float3 &coo);

    bool CentertoTriangle(tri_t *triangle, int node_id, int src_node);

    void RecordTriangle(tri_t *triangle, std::vector<float3> &triangle_centers,
                        std::vector<float3> &triangle_vertices, int src_node);

    void TransferVerticesToDevice(std::vector<float3> &triangle_vertices,
                                  std::vector<float3> &triangle_centers);

    void FreeLaunchParamsMemory();
    void CleanUp();

    void TraversalOnCPU(std::vector<int> &levels, int source_node = 0);

public:
    RTBFS_V2(Graph &graph, int chunck_length, int digit = 2);

    ~RTBFS_V2();

    void SetDevice(int device_id = 0);

    void OptiXSetup();
    // void PrepareRT();
    void BuildAccel(bool if_compact = false);

    void Traversal(int source_node = 0, bool filter = true);

    void CheckResult();

    void PrintResult(int head = 40);
};