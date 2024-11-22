#include <assert.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <queue>

#include "cuda_helper.h"
#include "record.h"
#include "rt_bfs_v2.h"

extern "C" char embedded_ptx_code_v2[];

static void context_log_cb(unsigned int level, const char *tag, const char *message,
                           void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message
              << "\n";
}

void RTBFS_V2::SetDevice(int device_id) {

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_id < 0 || device_id >= device_count) {
        std::cerr << "Device id should in [0, " << device_count << ")" << std::endl;
        exit(1);
    }
    use_device_id_ = device_id;
    cudaSetDevice(device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    size_t available_memory, total_memory;
    cudaMemGetInfo(&available_memory, &total_memory);
    std::cout << "==========>>> Device Info <<<==========\n";
    std::cout << "Total GPUs visible = " << device_count;
    std::cout << ", using [" << device_id << "]: " << device_prop.name << std::endl;
    std::cout << "Available Memory = " << int(available_memory / 1024 / 1024) << " MB, ";
    std::cout << "Total Memory = " << int(total_memory / 1024 / 1024) << " MB\n";
}

void RTBFS_V2::CreateContext() {
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    cuCtxGetCurrent(&cuda_context_);
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr; //&context_log_cb;
    options.logCallbackLevel = 4;
    // cuda_context_=0;
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, &options, &optix_context_));
}

void RTBFS_V2::CreateModule() {
    module_compile_options_.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.numPayloadValues = 4;
    pipeline_compile_options_.numAttributeValues = 2;
    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
    // scene contains nothing but built-in triangles
    pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    const std::string ptx_code = embedded_ptx_code_v2;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    // <7.7: optixModuleCreateFromPTX
    // 7.7: optixModuleCreate
    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_, &module_compile_options_,
                                         &pipeline_compile_options_, ptx_code.c_str(),
                                         ptx_code.size(), log, &sizeof_log, &optix_module_));
}

void RTBFS_V2::CreateProgramGroups() {
    char log[2048];
    size_t sizeof_log = sizeof(log);
    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_prog_group_desc = {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = optix_module_;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__bfs";
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_, &raygen_prog_group_desc, 1, // numner of program groups
        &program_group_options, log, &sizeof_log, &raygen_prog_group_));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = optix_module_;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_, &miss_prog_group_desc, 1, // number of program groups
        &program_group_options, log, &sizeof_log, &miss_prog_group_));

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH = optix_module_;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    // hit_prog_group_desc.hitgroup.moduleCH=optix_module_;
    // hit_prog_group_desc.hitgroup.entryFunctionNameCH="__closesthit__ch";

    OPTIX_CHECK(
        optixProgramGroupCreate(optix_context_, &hit_prog_group_desc, 1, // num program groups
                                &program_group_options, log, &sizeof_log, &hitgroup_prog_group_));
}

void RTBFS_V2::CreatePipeline() {
    char log[2048];
    size_t sizeof_log = sizeof(log);

    const uint32_t max_trace_depth = 1;
    pipeline_link_options_.maxTraceDepth =
        max_trace_depth; // maximum recursion depth setting for recursive ray tracing
    // 7.7 remove debugLevel
    // <7.7 have
    pipeline_link_options_.debugLevel =
        OPTIX_COMPILE_DEBUG_LEVEL_FULL; // pipeline level settings for debugging
    OptixProgramGroup program_groups[3] = {raygen_prog_group_, miss_prog_group_,
                                           hitgroup_prog_group_};

    OPTIX_CHECK(optixPipelineCreate(
        optix_context_, &pipeline_compile_options_, &pipeline_link_options_, program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]), log, &sizeof_log, &optix_pipeline_));
    // ========================
    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(
            prog_group, &stack_sizes)); // 7.7: add parameter optix_pipeline_, < 7.7 don't
    }
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth, 0, 0, // maxCCDepth, maxDCDepth
        &direct_callable_stack_size_from_traversal, &direct_callable_stack_size_from_state,
        &continuation_stack_size));
    OPTIX_CHECK(
        optixPipelineSetStackSize(optix_pipeline_, direct_callable_stack_size_from_traversal,
                                  direct_callable_stack_size_from_state, continuation_stack_size,
                                  1 // maxTraversableDepth
                                  ));
}

void RTBFS_V2::CreateSBT() {
    // build raygen record
    CUdeviceptr d_raygen_record = 0;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_raygen_record), raygen_record_size));
    RayGenRecord rg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_, &rg_record));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_raygen_record), &rg_record, raygen_record_size,
                          cudaMemcpyHostToDevice));
    // build miss record
    CUdeviceptr d_miss_record = 0;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_miss_record), miss_record_size));
    MissRecord ms_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_, &ms_record));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_miss_record), &ms_record, miss_record_size,
                          cudaMemcpyHostToDevice));
    // build hitgroup record
    CUdeviceptr d_hitgroup_record = 0;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_hitgroup_record), hitgroup_record_size));
    HitGroupRecord hg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_, &hg_record));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_hitgroup_record), &hg_record,
                          hitgroup_record_size, cudaMemcpyHostToDevice));
    // build sbt
    sbt_.raygenRecord = d_raygen_record;
    sbt_.missRecordBase = d_miss_record;
    sbt_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = d_hitgroup_record;
    sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt_.hitgroupRecordCount = 1;
}

// TODO: build BVH
void RTBFS_V2::BuildAccel(bool if_compact) {
    printf("RT-BFSV2: Preparation before Building BVH\n");
    GenerateTrianglesOnCPU(); //*  convert graph to triangles
    printf("RT-BFSV2: building BVH ...\n");

    cpu_timer_.StartTiming();
    num_t vertices_num = 3U * number_of_triangle_; // number of triangle vertices
    printf("RT-BFSV2: Memory size of triangles: %.2f GB\n",
           1.0 * vertices_num * sizeof(float3) / 1024 / 1024 / 1024);
    CUdeviceptr d_vertices = (CUdeviceptr)d_vertices_;
    // build input is a simple list of non-indexed triangle vertices
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices_num);
    triangle_input.triangleArray.vertexBuffers = &d_vertices; // vertices on device
    /* set index (no index) */
    // triangle_input.triangleArray.indexFormat=OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    // triangle_input.triangleArray.indexStrideInBytes=sizeof(int3);
    // triangle_input.triangleArray.numIndexTriplets=;
    // triangle_input.triangleArray.indexBuffer=;
    /* one SBT entry and no-primitive materials */
    const uint32_t triangle_input_flag[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL}; //
    triangle_input.triangleArray.flags = triangle_input_flag; // Array of flags
    triangle_input.triangleArray.numSbtRecords = 1;
    triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    /*
    enum OptixBuildFlags:
        OPTIX_BUILD_FLAG_NONE,
        OPTIX_BUILD_FLAG_ALLOW_UPDATE,
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE,
        OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS
    */
    // Use default options for simplicity.
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags =
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_context_, &accel_options, &triangle_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));
    // printf("gas_buffer_sizes.outputSizeInBytes = %.2f
    // GB\n",1.0*gas_buffer_sizes.outputSizeInBytes/1024/1024/1024);
    printf("RT-BFSV2: GAS output size = %.2f GB\n",
           1.0 * gas_buffer_sizes.outputSizeInBytes / 1024 / 1024 / 1024);
    /* prepare compaction */
    size_t *d_compacted_size;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_compacted_size), sizeof(size_t)));
    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = (CUdeviceptr)d_compacted_size;
    // ==============================================

    CUdeviceptr d_temp_buffer_gas; // temporarily used during the building
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
                          gas_buffer_sizes.tempSizeInBytes));
    CUdeviceptr d_gas_output_buffer; // memory size of acceleration structure
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer),
                          gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(optix_context_,
                                cuda_stream_, // CUDA stream
                                &accel_options, &triangle_input,
                                1, // num build inputs
                                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                                d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes,
                                &gas_handle_,
                                &emit_property, // nullptr, // emitted property list
                                1               // num emitted properties
                                ));
    // cudaDeviceSynchronize();
    d_gas_output_memory_ = d_gas_output_buffer; //

    /* perform compaction */
    size_t compacted_size;
    CUDA_CHECK(
        cudaMemcpy(&compacted_size, d_compacted_size, sizeof(size_t), cudaMemcpyDeviceToHost));
    CUdeviceptr d_compacted_output_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_compacted_output_buffer), compacted_size));
    /*  avoid the compacting pass in cases where it is not beneficial */
    if (if_compact && compacted_size < gas_buffer_sizes.outputSizeInBytes) {
        printf("RT-BFS: compacted GAS size = %.2f GB\n", 1.0 * compacted_size / 1024 / 1024 / 1024);
        cpu_timer_.StartTiming();
        OPTIX_CHECK(optixAccelCompact(optix_context_, cuda_stream_, gas_handle_,
                                      d_compacted_output_buffer, compacted_size, &gas_handle_));
        cudaDeviceSynchronize();
        cpu_timer_.StopTiming();
        bvh_building_time_ += cpu_timer_.GetElapsedTime();
        d_gas_output_memory_ = d_compacted_output_buffer;
        cudaFree(reinterpret_cast<void *>(d_gas_output_buffer));
    }
    // free the scratch space buffer used during build and the vertex input, since they are not
    // needed
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(d_vertices_));

    cpu_timer_.StopTiming();
    bvh_building_time_ = cpu_timer_.GetElapsedTime();
}
// TODO: convert data to triangles on CPU
void RTBFS_V2::GenerateTrianglesOnCPU() {
    std::vector<float3> triangle_vertices; // vertex information
    std::vector<float3> triangle_centers;  // center coordinate of each triangle

    origin_offset_.resize(graph_info_.node_num + 1); // ray origins of each node

    max_digits_ = NumberOfDigits(graph_info_.node_num - 1);
    adjust_ = (graph_info_.edge_num / 2 / chunk_length_ / 1.5); //! should be half of chunks

    int org_offset = 0, total_trangles = 0;
    for (int i = 0; i < graph_info_.node_num; i++) {
        int start = offsets_[i];
        int end = offsets_[i + 1];

        // split neighbors into chunks
        // count chunks (or ray)
        origin_offset_[i] = org_offset;
        int adj_offset = start;
        int node_triangles = 0; // count triangles of current node
        tri_t *cur_triangle = nullptr;

        while (adj_offset < end) {

            bool encode_res = CentertoTriangle(cur_triangle, adjs_[adj_offset], i);
            // if(adjs_[adj_offset] == 21746228) printf("%d: insert: %d, triangle id = %d\n", i,
            // encode_res, triangle_centers.size());
            if (!encode_res) {
                //* current triangle is full, record it
                if (cur_triangle != nullptr) {
                    RecordTriangle(cur_triangle, triangle_centers, triangle_vertices, i);
                    delete cur_triangle;
                }
                //* a new triangle
                // center
                int origin_id = org_offset + (node_triangles / chunk_length_);
                std::pair<int, int> coo = IdtoCoordinateinCurve(origin_id); // (x, z)
                // float y = 0; // in a cube
                float y = (origin_id < adjust_ ? 0 : -2); //
                // int tr = node_triangles % chunk_length_;
                // float y = (org_offset < adjust_ ? tr: -1 - tr);

                float3 triangle_center = make_float3(1.f * coo.first, y, 1.f * coo.second);
                ZoomCoordinate(triangle_center);
                // if(triangle_centers.size() == 33848896) printf("(%f, %f, %f), origin_id = %d\n",
                // triangle_center.x, triangle_center.y, triangle_center.z, origin_id);

                cur_triangle = new Triangle(triangle_center);
                node_triangles += 1;
            } else {
                adj_offset += 1;
            }
        }
        // the last triangle
        if (cur_triangle != nullptr) {
            RecordTriangle(cur_triangle, triangle_centers, triangle_vertices, i);
            delete cur_triangle;
        }
        //*
        org_offset += (node_triangles + chunk_length_ - 1) / chunk_length_; // count ray origins
        total_trangles += node_triangles;                                   // count triangles
        chunks_ += (node_triangles + chunk_length_ - 1) / chunk_length_;
    }
    origin_offset_[graph_info_.node_num] = org_offset;
    number_of_origin_ = org_offset;
    number_of_triangle_ = total_trangles;

    assert(triangle_centers.size() == number_of_triangle_);

    // std::vector<int> print_nodes;
    // print_nodes.push_back(22540267);
    // // print_nodes.push_back(22541485);
    // print_nodes.push_back(22541486);
    // for(int i = 0; i < print_nodes.size(); i++){
    //     int print_node = print_nodes[i];
    //     printf("%d (degree = %d): ", print_node, offsets_[print_node+1] - offsets_[print_node]);
    //     for(int i = offsets_[print_node]; i < offsets_[print_node + 1]; i++)
    //         printf("%d, ", adjs_[i]);
    //     printf("\n");
    // }
    // int print_node = 0;
    // printf("%d (degree = %d): ", print_node, offsets_[print_node+1] - offsets_[print_node]);
    // for(int i = offsets_[print_node]; i < offsets_[print_node + 1]; i++)
    //     printf("%d, ", adjs_[i]);
    // printf("\n");

    //* transfer vertices to device
    TransferVerticesToDevice(triangle_vertices, triangle_centers);
}

void RTBFS_V2::RecordTriangle(tri_t *triangle, std::vector<float3> &triangle_centers,
                              std::vector<float3> &triangle_vertices, int src_node) {
    //* center
    triangle_centers.push_back(triangle->center);
    encoded_nodes_ += triangle->node_count;
    // if(src_node == 4000000) printf("triangle id = %d\n",triangle_centers.size() - 1);

    //* first vertex: 0, +, +
    float x = triangle->center.x;
    float y = triangle->center.y + triangle->encode_data[0];
    float z = triangle->center.z + triangle->encode_data[1];
    triangle_vertices.push_back(make_float3(x, y, z));

    // if(src_node == 1413125) printf("vertx A: (%.3f, %.3f, %.3f)\n", x, y, z);
    //* second vertex: -, +, -
    x = triangle->center.x - triangle->encode_data[2];
    y = triangle->center.y + triangle->encode_data[3];
    z = triangle->center.z - triangle->encode_data[4];
    triangle_vertices.push_back(make_float3(x, y, z));

    // if(src_node == 1413125) printf("vertx B: (%.3f, %.3f, %.3f)\n", x, y, z);
    //* third vertex: +, +, -
    x = triangle->center.x + triangle->encode_data[5];
    y = triangle->center.y + triangle->encode_data[6];
    z = triangle->center.z - triangle->encode_data[7];
    triangle_vertices.push_back(make_float3(x, y, z));

    // if(src_node == 1413125) printf("vertx C: (%.3f, %.3f, %.3f)\n", x, y, z);
}

std::pair<int, int> RTBFS_V2::IdtoCoordinateinCurve(int id) {
    //* put a half of chunks in another side
    if (id >= adjust_)
        id -= adjust_;
    // compute (x ,z) of origin coordinate
    int x = 0, z = 0;
    if (id == 0) {
        return std::make_pair(x, z);
    }
    int side = (int)sqrt(id);
    side += (side % 2 == 0 ? 1 : 2);
    int res = side * side - 1 - id;
    int temp = res / (side - 1);
    int axis = (int)(side / 2);
    if (temp == 0) {
        x = axis - res;
        z = -axis;
    } else if (temp == 1) {
        x = -axis;
        z = -axis + res - (side - 1);
    } else if (temp == 2) {
        x = -axis + res - 2 * (side - 1);
        z = axis;
    } else if (temp == 3) {
        x = axis;
        z = axis - (res - 3 * (side - 1));
    } else {
        printf("ERROR: IdtoCoordinateinCurve\n");
        exit(1);
    }
    return std::make_pair(x, z);
}

int RTBFS_V2::NumberOfDigits(int num) {
    if (num == 0)
        return 1; // Special case for 0
    return static_cast<int>(log10(abs(num)) + 1);
}

void RTBFS_V2::ZoomCoordinate(float3 &coo) {
    coo.x *= zoom_.x;
    coo.y *= zoom_.y;
    coo.z *= zoom_.z;
}

int Powerof10(int e) {
    int res = 1;
    for (int i = 0; i < e; i++)
        res *= 10;
    return res;
}

float RTBFS_V2::EncodeNodeId(int id) {
    float res = id;
    for (int i = 0; i < encode_digits_; i++)
        res /= 10;
    return res;
}

// TODO: encode nodes into triangle vertices
bool RTBFS_V2::CentertoTriangle(tri_t *triangle, int node_id, int src_node) {
    // TODO: encode 2 - store id into decimal part
    if (triangle == nullptr)
        return false;

    int digits = NumberOfDigits(node_id);             // digits
    int num = (digits + id_digits_ - 1) / id_digits_; // needed for id

    // if(node_id == 22541486) printf("needed num = %d, rest = %d\n",num, nodes_per_triangle_ -
    // triangle->pos);

    if (num <= nodes_per_triangle_ - triangle->pos) {
        if (num == 1) {
            triangle->encode_data[triangle->pos++] = triangle->mul_cnt + EncodeNodeId(node_id);
        } else {
            //* flag is placed at tenths unit
            triangle->mul_cnt += 0.1f;
            for (int i = num - 1; i >= 0; i--) {
                int par_id = node_id % id_mod_;
                triangle->encode_data[triangle->pos + i] =
                    triangle->mul_cnt + EncodeNodeId(par_id); // diff
                node_id /= id_mod_;
            }
            triangle->pos += num;
        }
        triangle->node_count += 1;
        return true;
    } else {
        return false;
    }
}

void RTBFS_V2::TransferVerticesToDevice(std::vector<float3> &triangle_vertices,
                                        std::vector<float3> &triangle_centers) {
    CUDA_CHECK(cudaMalloc(&d_vertices_, sizeof(float3) * number_of_triangle_ * 3));
    CUDA_CHECK(cudaMemcpy(d_vertices_, triangle_vertices.data(),
                          sizeof(float3) * number_of_triangle_ * 3, cudaMemcpyHostToDevice));

    // CUDA_CHECK(cudaMalloc(&d_centers_, sizeof(float3) * number_of_triangle_));
    // CUDA_CHECK(cudaMemcpy(d_centers_, triangle_centers.data(), sizeof(float3) *
    // number_of_triangle_, cudaMemcpyHostToDevice));
}

void RTBFS_V2::Traversal(int source_node, bool filter) {
    printf("==========>>> Method Info <<<==========\n");
    printf("Total triangles = %d, chunks = %d, adjust = %d, zoom = (%.2f, %.2f, %.2f)\n",
           number_of_triangle_, chunks_, adjust_, zoom_.x, zoom_.y, zoom_.z);
    printf("Encode: ");
    printf("maximal digits = %d, encode digits = %d, encode mod = %d, id_digits = %d\n",
           max_digits_, encode_digits_, encode_mod_, id_digits_);

    printf("RT-BFS V2: RT traversal starts from source node %d (degree = %d)\n", source_node,
           offsets_[source_node + 1] - offsets_[source_node]);
    // set launch params
    h_params_.handle = gas_handle_;
    //! ray length
    h_params_.max_ray_length = zoom_.y; // zoom_.y *chunk_length_;
    h_params_.adjust = adjust_;
    // h_params_.triangle_center = d_centers_;
    h_params_.zoom = zoom_;
    h_params_.nodes = graph_info_.node_num;
    h_params_.encode_digits = encode_digits_;
    h_params_.encode_mod = encode_mod_;

    if (filter)
        printf("----- Filter is used\n");
    else
        printf("----- Filter isn't used\n");

    int queue_size = 1;
    CUDA_CHECK(cudaMalloc((void **)&h_params_.origins, sizeof(int) * number_of_origin_));
    //* initial queue
    CUDA_CHECK(cudaMalloc((void **)&h_params_.queue, sizeof(int) * graph_info_.node_num));
    CUDA_CHECK(cudaMemcpy(h_params_.queue, &source_node, sizeof(int), cudaMemcpyHostToDevice));
    //* initial queue size
    CUDA_CHECK(cudaMalloc((void **)&h_params_.queue_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_params_.queue_size, &queue_size, sizeof(int), cudaMemcpyHostToDevice));
    //* result
    CUDA_CHECK(cudaMalloc((void **)&h_params_.levels, sizeof(int) * graph_info_.node_num));

    ThrustFill(h_params_.levels, graph_info_.node_num, -1); // initial levels
    CUDA_CHECK(cudaMemset(h_params_.levels + source_node, 0, sizeof(int)));
    // ThrustFill(h_params_.levels, 1, 0);
    // allocate device params memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params_ptr_), sizeof(LaunchParamsV2)));

    //* transfer origin offset
    int *d_origin_offset;
    CUDA_CHECK(cudaMalloc((void **)&d_origin_offset, sizeof(int) * origin_offset_.size()));
    CUDA_CHECK(cudaMemcpy(d_origin_offset, origin_offset_.data(),
                          sizeof(int) * origin_offset_.size(), cudaMemcpyHostToDevice));

    //* rays in each round
    int *d_origin_num;
    CUDA_CHECK(cudaMalloc((void **)&d_origin_num, sizeof(int)));

    //*
    cudaEvent_t launch_start, launch_end;
    CUDA_CHECK(cudaEventCreate(&launch_start));
    CUDA_CHECK(cudaEventCreate(&launch_end));
    double trace_time = 0.0;
    double filter_map_time = 0.0;
    h_params_.current_level = 1;
    int cnt = 0;
    int total_launch_size = 0;
    while (true) {
        // TODO:
        CUDA_CHECK(
            cudaMemcpy(&queue_size, h_params_.queue_size, sizeof(int), cudaMemcpyDeviceToHost));
    
        // printf("---------------------- queue size = %d\n",queue_size);
        if (queue_size == 0)
            break;

        CUDA_CHECK(cudaMemset(d_origin_num, 0, sizeof(int)));
        filter_map_time += GetOriginsByNodes(queue_size, h_params_.queue, d_origin_offset,
                                             d_origin_num, h_params_.origins, filter);

        CUDA_CHECK(cudaMemset(h_params_.queue_size, 0, sizeof(int)));
        CUDA_CHECK(
            cudaMemcpy(d_params_ptr_, &h_params_, sizeof(LaunchParamsV2), cudaMemcpyHostToDevice));

        // int temp_origin_num=0;
        // CUDA_CHECK(cudaMemcpy(&temp_origin_num,d_origin_num,sizeof(int),cudaMemcpyDeviceToHost));
        // printf("temp_origin_num = %d, degree = %d\n",temp_origin_num,offsets_[1]-offsets_[0]);

        // TODO: Launch ray trace
        int launch_size = 0;
        CUDA_CHECK(cudaMemcpy(&launch_size, d_origin_num, sizeof(int), cudaMemcpyDeviceToHost));
        if (launch_size == 0)
            break;
        // printf("Level %d: %d\n",h_params_.current_level,launch_size);
        // assert(launch_size<number_of_origin_);
        CUDA_CHECK(cudaEventRecord(launch_start, cuda_stream_));
        OPTIX_CHECK(optixLaunch(optix_pipeline_, cuda_stream_,
                                reinterpret_cast<CUdeviceptr>(d_params_ptr_),
                                sizeof(LaunchParamsV2), &sbt_, launch_size, 1, 1));
        CUDA_CHECK(cudaEventRecord(launch_end, cuda_stream_));
        CUDA_CHECK(cudaEventSynchronize(launch_end));
        float elapsed_time = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, launch_start, launch_end));
        trace_time += elapsed_time;
        // printf("Level %d: prepare time = %.2f ms, elapsed time = %.2f
        // ms\n",h_params_.current_level,prepare_time,elapsed_time);
        h_params_.current_level += 1;

        cnt += 1;
        total_launch_size += launch_size;
    }
    printf("Max level = %d, avg. rays = %.2f\n", h_params_.current_level - 1,
           1.0 * total_launch_size / cnt);
    CUDA_CHECK(cudaMemcpy(levels_.data(), h_params_.levels, sizeof(int) * graph_info_.node_num,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(launch_start));
    CUDA_CHECK(cudaEventDestroy(launch_end));
    CUDA_CHECK(cudaFree(d_origin_offset));

    // printf("Prepare time before launching = %.2f ms\n",prepare_time);
    traversal_time_ = filter_map_time + trace_time;
    printf("Traversal: filter and map time = %f ms, trace time = %f ms\n", filter_map_time,
           trace_time);
}

void RTBFS_V2::FreeLaunchParamsMemory() {
    CUDA_CHECK(cudaFree(h_params_.origins));
    // CUDA_CHECK(cudaFree(h_params_.triangle_center)); // <=> cudaFree(device_triangle_id_)
    CUDA_CHECK(cudaFree(h_params_.queue));
    CUDA_CHECK(cudaFree(h_params_.queue_size));
    CUDA_CHECK(cudaFree(h_params_.levels));
    // CUDA_CHECK(cudaFree(h_params_.ray_length));
}

void RTBFS_V2::CleanUp() {
    FreeLaunchParamsMemory();
    // CUDA_CHECK(cudaFree(device_triangle_id_));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt_.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt_.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt_.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(
        reinterpret_cast<void *>(d_gas_output_memory_))); // free gas memory, compacted or not

    OPTIX_CHECK(optixPipelineDestroy(optix_pipeline_));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group_));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group_));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group_));
    OPTIX_CHECK(optixModuleDestroy(optix_module_));
    CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context_));
}

void RTBFS_V2::OptiXSetup() {
    printf("RT-BFS: OptiX setup...\n");
    cpu_timer_.StartTiming();
    CreateContext();
    CreateModule();
    CreateProgramGroups();
    CreatePipeline();
    CreateSBT();
    cpu_timer_.StopTiming();
    optix_setup_time_ = cpu_timer_.GetElapsedTime();
}

RTBFS_V2::RTBFS_V2(Graph &graph, int chunk_length, int digit)
    : chunk_length_(chunk_length), id_digits_(digit) {
    graph_info_ = graph.GetGraphInfo();
    offsets_ = graph.GetOffsets();
    adjs_ = graph.GetAdjs();
    levels_.resize(graph_info_.node_num);

    zoom_ = make_float3(2.f, 1.f, 2.f);

    encode_digits_ = 1 + id_digits_;
    encode_mod_ = Powerof10(encode_digits_);
    id_mod_ = encode_mod_ / 10;
}

RTBFS_V2::~RTBFS_V2() { CleanUp(); }

void RTBFS_V2::PrintResult(int head) {
    printf("=====>>> RT BFS <<<===== \n");
    printf("RT Distance[:%d] = ", head);
    for (int i = 0; i < head; i += 1)
        printf(" %d", levels_[i]);
    printf("\n");
    total_time_ = optix_setup_time_ + bvh_building_time_ + traversal_time_;
    printf(" - avg. nodes per triangle = %.2f\n", 1.0 * encoded_nodes_ / number_of_triangle_);
    printf(" - Optix setup time = %f ms\n", optix_setup_time_);
    printf(" - BVH building time = %f ms\n", bvh_building_time_);
    printf(" - Traversal time = %f ms\n", traversal_time_);
    printf(" - Total counting time = %f ms\n", total_time_);
}

void RTBFS_V2::CheckResult() {
    std::vector<int> cpu_levels;
    TraversalOnCPU(cpu_levels);
    printf("CPU Distance[:40] = ");
    for (int i = 0; i < 40; i += 1)
        printf(" %d", cpu_levels[i]);
    printf("\n");
    int errors = 0;
    int right_min_level = graph_info_.node_num;
    int err_min_level = graph_info_.node_num;
    int min_level_node = -1;
    for (int i = 0; i < graph_info_.node_num; i += 1) {
        if (cpu_levels[i] != levels_[i]) {
            // printf("%d: %d - %d, ",i, levels_[i],cpu_levels[i]);
            errors += 1;
            if (right_min_level > cpu_levels[i]) {
                right_min_level = cpu_levels[i];
                err_min_level = levels_[i];
                min_level_node = i;
            }
        }
    }
    if (min_level_node != -1)
        printf("%d: %d -> %d\n", min_level_node, right_min_level, err_min_level);
    printf("Check Result: number of errors = %d\n", errors);
}

void RTBFS_V2::TraversalOnCPU(std::vector<int> &levels, int source_node) {
    std::queue<int> que;
    levels.resize(graph_info_.node_num);
    for (int i = 0; i < graph_info_.node_num; i += 1)
        levels[i] = -1;
    levels[source_node] = 0;
    que.push(source_node);
    int max_level = 0;
    while (!que.empty()) {
        int now = que.front();
        que.pop();
        int start = offsets_[now];
        int end = offsets_[now + 1];
        for (int i = start; i < end; i += 1) {
            int adj_id = adjs_[i];
            if (levels[adj_id] == -1) {
                que.push(adj_id);
                levels[adj_id] = levels[now] + 1;
                max_level = std::max(max_level, levels[adj_id]);
            }
        }
    }
    printf("CPU max level = %d\n", max_level);
}