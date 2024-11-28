#include "cuda_helper.h"
#include "record.h"
#include "rt_bfs.h"

#include <assert.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <optix_function_table_definition.h>
#include <queue>

extern "C" char embedded_ptx_code[];

static void context_log_cb(unsigned int level, const char *tag, const char *message,
                           void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message
              << "\n";
}

void RTBFS::SetDevice(int device_id) {

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

void RTBFS::CreateContext() {
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

void RTBFS::CreateModule() {
    module_compile_options_.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options_.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options_.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_.usesMotionBlur = false;
    pipeline_compile_options_.numPayloadValues = 1;
    pipeline_compile_options_.numAttributeValues = 0;
    pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options_.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // scene contains nothing but built-in triangles

    const std::string ptx_code = embedded_ptx_code;
    char log[2048];
    size_t sizeof_log = sizeof(log);
    // <7.7: optixModuleCreateFromPTX
    // 7.7: optixModuleCreate
    OPTIX_CHECK(optixModuleCreateFromPTX(optix_context_, &module_compile_options_,
                                         &pipeline_compile_options_, ptx_code.c_str(),
                                         ptx_code.size(), log, &sizeof_log, &optix_module_));
}

void RTBFS::CreateProgramGroups() {
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

void RTBFS::CreatePipeline() {
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

void RTBFS::CreateSBT() {
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
void RTBFS::BuildAccel(bool if_compact) {
    printf("RT-BFS: Preparation before Building\n");
    GenerateTrianglesOnCPU(); //! convert graph to triangles
    printf("RT-BFS: building BVH ...\n");

    cpu_timer_.StartTiming();
    num_t vertices_num = 3U * number_of_triangle_; // number of triangle vertices
    printf("RT-BFS: Memory size of triangles: %.2f GB\n",
           1.0 * vertices_num * sizeof(float3) / 1024 / 1024 / 1024);
    CUdeviceptr d_vertices = (CUdeviceptr)device_vertices_;
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
    printf("RT-BFS: GAS output size = %.2f GB\n",
           1.0 * gas_buffer_sizes.outputSizeInBytes / 1024 / 1024 / 1024);
    /* prepare compaction */
    size_t *d_compacted_size;
    cudaMalloc(reinterpret_cast<void **>(&d_compacted_size), sizeof(size_t));
    OptixAccelEmitDesc emit_property = {};
    emit_property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result = (CUdeviceptr)d_compacted_size;
    // ==============================================

    CUdeviceptr d_temp_buffer_gas; // temporarily used during the building
    cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes);
    CUdeviceptr d_gas_output_buffer; // memory size of acceleration structure
    cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes);

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
    cudaMemcpy(&compacted_size, d_compacted_size, sizeof(size_t), cudaMemcpyDeviceToHost);
    CUdeviceptr d_compacted_output_buffer;
    cudaMalloc(reinterpret_cast<void **>(&d_compacted_output_buffer), compacted_size);
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
    cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas));
    cudaFree(device_vertices_);

    cpu_timer_.StopTiming();
    bvh_building_time_ = cpu_timer_.GetElapsedTime();
}
// TODO: convert data to triangles on CPU
void RTBFS::GenerateTrianglesOnCPU() {
    number_of_triangle_ = graph_info_.edge_num;
    int number_of_vertex = 3 * number_of_triangle_;
    std::vector<float3> triangle_vertices(number_of_vertex);
    std::vector<int> triangle_id(graph_info_.edge_num);
    // origins of each node
    origin_offset_.resize(graph_info_.node_num + 1);
    //* get the number of origins
#pragma omp parallel for reduction(+ : chunks_)
    for (int i = 0; i < graph_info_.node_num; i += 1) {
        int adj_len = offsets_[i + 1] - offsets_[i];
        int chunk_num = (adj_len + chunk_length_ - 1) / chunk_length_;
        chunks_ += chunk_num;
    }
    adjust_ = chunks_ / 2; // *
    ray_lengths_.resize(chunks_);
    //* get the chunk offset of each node
    int offset = 0, triangle_count = 0;
    for (int i = 0; i < graph_info_.node_num; i += 1) {
        origin_offset_[i] = offset;
        int start = offsets_[i];
        int end = offsets_[i + 1];
        for (int j = start; j < end; j++) {
            int adj_id = adjs_[j];
            int origin_id = (j - start) / chunk_length_ + offset; // => (x, z)
            int id_in_chunck = (j - start) % chunk_length_;       // => y
            //* get the center coordinate of triangle
            std::pair<float, float> coo = ConvertIdToCoordinate(origin_id); //! id -> coordinate,
            float y = (origin_id < adjust_ ? 1.0f * id_in_chunck : -1.0f * id_in_chunck - 1.0f);
            float3 triangle_center = make_float3(1.0f * coo.first, y, 1.0f * coo.second);
            GetTriangleVertices(triangle_center, triangle_count, triangle_vertices);
            triangle_id[triangle_count] = adj_id;
            triangle_count += 1;
        }
        int chunk_num =
            (end - start + chunk_length_ - 1) / chunk_length_; // chunks number of this node
        // record real length of each chunk
        for (int j = 0; j < chunk_num; j += 1) {
            if (j == chunk_num - 1)
                ray_lengths_[offset + j] = (end - start - 1) % chunk_length_ + 1;
            else
                ray_lengths_[offset + j] = chunk_length_;
        }
        offset += chunk_num;
    }
    origin_offset_[graph_info_.node_num] = offset;
    number_of_origin_ = offset;
    // printf("origin offfset:");
    // for(int i=0;i<10;i+=1) printf(" (%d, %d)",origin_offset_[i],origin_offset_[i+1]);
    // printf("\n");
    assert(triangle_count == graph_info_.edge_num);
    // TODO: transfer vertices to device
    TransferVerticesToDevice(triangle_vertices, triangle_id);
}

std::pair<float, float> RTBFS::ConvertIdToCoordinate(int origin_id) {
    //* put a half of chunks in another side
    if (origin_id >= adjust_)
        origin_id -= adjust_;
    // compute (x ,z) of origin coordinate
    float x = 0, z = 0;
    if (origin_id == 0)
        return std::make_pair(x, z);
    int side = (int)sqrt(origin_id);
    side += (side % 2 == 0 ? 1 : 2);
    int res = side * side - 1 - origin_id;
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
    }
    //* zoom:
    x *= zoom_;
    z *= zoom_;
    return std::make_pair(x, z);
}

void RTBFS::GetTriangleVertices(float3 center, int idx, std::vector<float3> &triangle_vertices) {
    float just_below_x = center.x - triangle_eps_;
    float just_above_x = center.x + triangle_eps_;
    float just_below_z = center.z - triangle_eps_;
    float just_above_z = center.z + triangle_eps_;
    triangle_vertices[idx * 3] = make_float3(center.x, center.y, just_above_z);
    triangle_vertices[idx * 3 + 1] = make_float3(just_below_x, center.y, just_below_z);
    triangle_vertices[idx * 3 + 2] = make_float3(just_above_x, center.y, just_below_z);
}

void RTBFS::TransferVerticesToDevice(std::vector<float3> &triangle_vertices,
                                     std::vector<int> &triangle_id) {
    cudaMalloc(&device_vertices_, sizeof(float3) * number_of_triangle_ * 3);
    cudaMemcpy(device_vertices_, triangle_vertices.data(), sizeof(float3) * number_of_triangle_ * 3,
               cudaMemcpyHostToDevice);
    cudaMalloc(&device_triangle_id_, sizeof(int) * number_of_triangle_);
    cudaMemcpy(device_triangle_id_, triangle_id.data(), sizeof(int) * number_of_triangle_,
               cudaMemcpyHostToDevice);
}

void RTBFS::Traversal(int source_node, bool filter) {
    printf("RT-BFS: RT traversal starts from source node %d(degree = %d)\n", source_node,
           offsets_[source_node + 1] - offsets_[source_node]);
    // set launch params
    h_params_.handle = gas_handle_;
    h_params_.chunk_length = chunk_length_;
    h_params_.adjust = adjust_;
    h_params_.triangle_id = device_triangle_id_;
    h_params_.zoom = zoom_;

    int origins_mem_size = number_of_origin_;
    int queue_mem_size = graph_info_.node_num;
    
    if (filter) {
        printf("----- Filter is used\n");
    } else {
        origins_mem_size = graph_info_.edge_num * 3;
        queue_mem_size = graph_info_.edge_num * 3;
        printf("----- Filter isn't used\n");
    }

    int queue_size = 1;
    CUDA_CHECK(cudaMalloc((void **)&h_params_.origins, sizeof(int) * origins_mem_size));
    CUDA_CHECK(cudaMalloc((void **)&h_params_.queue, sizeof(int) * queue_mem_size));
    CUDA_CHECK(cudaMemcpy(h_params_.queue, &source_node, sizeof(int),
                          cudaMemcpyHostToDevice)); //* initial queue
    CUDA_CHECK(cudaMalloc((void **)&h_params_.queue_size, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(h_params_.queue_size, &queue_size, sizeof(int),
                          cudaMemcpyHostToDevice)); //* initial queue size
    CUDA_CHECK(cudaMalloc((void **)&h_params_.levels, sizeof(int) * graph_info_.node_num));
    CUDA_CHECK(cudaMalloc((void **)&h_params_.ray_length, sizeof(int) * number_of_origin_));
    CUDA_CHECK(cudaMemcpy(h_params_.ray_length, ray_lengths_.data(),
                          sizeof(int) * number_of_origin_, cudaMemcpyHostToDevice));
    ThrustFill(h_params_.levels, graph_info_.node_num, -1); // initial levels
    ThrustFill(h_params_.levels, 1, 0);
    // allocate device params memory
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params_ptr_), sizeof(LaunchParams)));
    //* transfer origin offset
    int *d_origin_offset;
    CUDA_CHECK(cudaMalloc((void **)&d_origin_offset, sizeof(int) * origin_offset_.size()));
    CUDA_CHECK(cudaMemcpy(d_origin_offset, origin_offset_.data(),
                          sizeof(int) * origin_offset_.size(), cudaMemcpyHostToDevice));
    int *d_origin_num;
    CUDA_CHECK(cudaMalloc((void **)&d_origin_num, sizeof(int)));
    //*
    cudaEvent_t launch_start, launch_end;
    CUDA_CHECK(cudaEventCreate(&launch_start));
    CUDA_CHECK(cudaEventCreate(&launch_end));
    double trace_time = 0.0;
    double filter_map_time = 0.0;
    h_params_.current_level = 1;
    while (true) {
        // TODO:
        CUDA_CHECK(
            cudaMemcpy(&queue_size, h_params_.queue_size, sizeof(int), cudaMemcpyDeviceToHost));
        // printf("level = %d, queue size = %d\n", h_params_.current_level - 1, queue_size);

        // int temp_level = 0;
        // cudaMemcpy(&temp_level, h_params_.levels, sizeof(int), cudaMemcpyDeviceToHost);
        // printf("l = %d, temp_level = %d\n", h_params_.current_level - 1, temp_level);

        // if (h_params_.current_level >= 5)
        //     break;

        if (queue_size == 0)
            break;
        CUDA_CHECK(cudaMemset(d_origin_num, 0, sizeof(int)));
        filter_map_time += GetOriginsByNodes(queue_size, h_params_.queue, d_origin_offset,
                                             d_origin_num, h_params_.origins, filter); //!
        CUDA_CHECK(cudaMemset(h_params_.queue_size, 0, sizeof(int)));
        CUDA_CHECK(
            cudaMemcpy(d_params_ptr_, &h_params_, sizeof(LaunchParams), cudaMemcpyHostToDevice));

        // int temp_origin_num=0;
        // CUDA_CHECK(cudaMemcpy(&temp_origin_num,d_origin_num,sizeof(int),cudaMemcpyDeviceToHost));
        // printf("temp_origin_num = %d, degree = %d\n",temp_origin_num,offsets_[1]-offsets_[0]);

        // TODO: Launch ray trace
        int launch_size = 0;
        CUDA_CHECK(cudaMemcpy(&launch_size, d_origin_num, sizeof(int), cudaMemcpyDeviceToHost));
        // printf("Level %d: %d\n",h_params_.current_level,launch_size);
        // assert(launch_size<number_of_origin_);
        CUDA_CHECK(cudaEventRecord(launch_start, cuda_stream_));
        OPTIX_CHECK(optixLaunch(optix_pipeline_, cuda_stream_,
                                reinterpret_cast<CUdeviceptr>(d_params_ptr_), sizeof(LaunchParams),
                                &sbt_, launch_size, 1, 1));
        CUDA_CHECK(cudaEventRecord(launch_end, cuda_stream_));
        CUDA_CHECK(cudaEventSynchronize(launch_end));
        float elapsed_time = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, launch_start, launch_end));
        trace_time += elapsed_time;
        // printf("Level %d: prepare time = %.2f ms, elapsed time = %.2f
        // ms\n",h_params_.current_level,prepare_time,elapsed_time);
        h_params_.current_level += 1;
    }
    // int temp_level = 0;
    // cudaMemcpy(&temp_level, h_params_.levels, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("temp_level = %d\n", temp_level);

    printf("Max level = %d\n", h_params_.current_level - 1);
    CUDA_CHECK(cudaMemcpy(levels_.data(), h_params_.levels, sizeof(int) * graph_info_.node_num,
                          cudaMemcpyDeviceToHost));
    // printf("0: %d\n", levels_[0]);

    CUDA_CHECK(cudaEventDestroy(launch_start));
    CUDA_CHECK(cudaEventDestroy(launch_end));
    CUDA_CHECK(cudaFree(d_origin_offset));
    // printf("Prepare time before launching = %.2f ms\n",prepare_time);
    traversal_time_ = filter_map_time + trace_time;
    printf("Traversal: filter and map time = %f ms, trace time = %f ms\n", filter_map_time,
           trace_time);
}

void RTBFS::FreeLaunchParamsMemory() {
    CUDA_CHECK(cudaFree(h_params_.origins));
    CUDA_CHECK(cudaFree(h_params_.triangle_id)); // <=> cudaFree(device_triangle_id_)
    CUDA_CHECK(cudaFree(h_params_.queue));
    CUDA_CHECK(cudaFree(h_params_.queue_size));
    CUDA_CHECK(cudaFree(h_params_.levels));
    CUDA_CHECK(cudaFree(h_params_.ray_length));
}

void RTBFS::CleanUp() {
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

void RTBFS::OptiXSetup() {
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

RTBFS::RTBFS(Graph &graph, int chunk_length) : chunk_length_(chunk_length) {
    graph_info_ = graph.GetGraphInfo();
    offsets_ = graph.GetOffsets();
    adjs_ = graph.GetAdjs();
    levels_.resize(graph_info_.node_num);
    // place_num_=graph_info_.node_num/2;
}

RTBFS::~RTBFS() { CleanUp(); }

void RTBFS::PrintResult(int head) {
    printf("=====>>> RT BFS <<<===== \n");
    printf("RT Distance[:%d] = ", head);
    for (int i = 0; i < head; i += 1)
        printf(" %d", levels_[i]);
    printf("\n");
    total_time_ = optix_setup_time_ + bvh_building_time_ + traversal_time_;
    printf(" - Optix setup time = %f ms\n", optix_setup_time_);
    printf(" - BVH building time = %f ms\n", bvh_building_time_);
    printf(" - Traversal time = %f ms\n", traversal_time_);
    printf(" - Total counting time = %f ms\n", total_time_);
}

void RTBFS::CheckResult() {
    std::vector<int> cpu_levels;
    TraversalOnCPU(cpu_levels);
    printf("CPU Distance[:40] = ");
    for (int i = 0; i < 40; i += 1)
        printf(" %d", cpu_levels[i]);
    printf("\n");
    int errors = 0;
    for (int i = 0; i < graph_info_.node_num; i += 1) {
        if (cpu_levels[i] != levels_[i]) {
            // printf("%d - %d, ",levels_[i],cpu_levels[i]);
            errors += 1;
        }
    }
    printf("Check Result: number of errors = %d\n", errors);
}

void RTBFS::TraversalOnCPU(std::vector<int> &levels, int source_node) {
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