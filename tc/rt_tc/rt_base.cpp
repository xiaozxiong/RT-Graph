#include "rt_base.h"
#include "common.h"
#include "records.h"
#include "timer.h"

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <string>
#include <iostream>
#include <iomanip>

extern "C" char embedded_ptx_code[];

#ifdef DEBUG
static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
    << message << "\n";
}
#endif

void RTBase::SetDevice(int device_id){
    int device_count=0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop,device_id));
    size_t available_memory,total_memory;
    CUDA_CHECK(cudaMemGetInfo(&available_memory,&total_memory));
    std::cout<<"==========>>> Device Info <<<==========\n";
    std::cout<<"Total GPUs visible = "<<device_count;
    std::cout<<", using ["<<device_id<<"]: "<<device_prop.name<<std::endl;
    std::cout<<"Available Memory = "<<int(available_memory/1024/1024)<<" MB, ";
    std::cout<<"Total Memory = "<<int(total_memory/1024/1024)<<" MB\n";
}

void RTBase::CreateContext(){
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    OPTIX_CHECK(optixInit());
    CUresult cu_res=cuCtxGetCurrent(&cuda_context_);
    if(cu_res!=CUDA_SUCCESS)  fprintf( stderr, "Error querying current context: error code %d\n",cu_res);

    OptixDeviceContextOptions options={};
#ifdef DEBUG
    options.logCallbackFunction=&context_log_cb;
#else
    options.logCallbackFunction=nullptr;
#endif
    options.logCallbackLevel=4;
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_,&options,&optix_context_));
}

void RTBase::CreateModule(){
    //* moduleCompileOptions
    module_compile_options_.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options_.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options_.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    //* pipelineCompileOptions
    pipeline_compile_options_.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_.usesMotionBlur=false;
    pipeline_compile_options_.numPayloadValues=2; //*
    pipeline_compile_options_.numAttributeValues=0; //*
    pipeline_compile_options_.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName="params";
    // scene contains nothing but built-in triangles
    pipeline_compile_options_.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    const std::string ptx_code=embedded_ptx_code;

    char log[2048];
    size_t sizeof_log=sizeof(log);
    //* create module
#if OPTIX_VERSION >= 70700
    OPTIX_CHECK(optixModuleCreate(
        optix_context_,
        &module_compile_options_,
        &pipeline_compile_options_,
        ptxCode.c_str(),
        ptxCode.size(),
        log,
        &sizeof_log,
        &optix_module_
    ));
#else
    OPTIX_CHECK(optixModuleCreateFromPTX(
        optix_context_,
        &module_compile_options_,
        &pipeline_compile_options_,
        ptx_code.c_str(),
        ptx_code.size(),
        log,
        &sizeof_log,
        &optix_module_
    ));
#endif
    // if(sizeof_log > 1) PRINT(log);
}

void RTBase::CreateRaygenPrograms(){
    // numProgramGroups=1
    raygen_prog_groups_.resize(1);

    OptixProgramGroupOptions program_group_options={};
    OptixProgramGroupDesc raygen_prog_group_desc={};
    raygen_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module=optix_module_;
    raygen_prog_group_desc.raygen.entryFunctionName="__raygen__tc";

    char log[2048];
    size_t sizeof_log=sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_,&raygen_prog_group_desc,1, // numProgramGroups
        &program_group_options,log,&sizeof_log,&raygen_prog_groups_[0]
    ));
    // if(sizeof_log > 1) PRINT(log);
}

void RTBase::CreateMissPrograms(){
    // numProgramGroups=1
    miss_prog_groups_.resize(1);

    OptixProgramGroupOptions program_group_options={};
    OptixProgramGroupDesc miss_prog_group_desc={};
    miss_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module=optix_module_;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";

    char log[2048];
    size_t sizeof_log=sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_,&miss_prog_group_desc,1, // numProgramGroups
        &program_group_options,log,&sizeof_log,&miss_prog_groups_[0]
    ));
    // if(sizeof_log > 1) PRINT(log);
}

void RTBase::CreateHitgroupPrograms(){
    // numProgramGroups=1, kind of hit
    hitgroup_prog_groups_.resize(1);

    OptixProgramGroupOptions program_group_options={};
    OptixProgramGroupDesc hit_prog_group_desc={};
    hit_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    //* any hit program
    hit_prog_group_desc.hitgroup.moduleAH=optix_module_;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH="__anyhit__ah";
    //* closest hit program
    // hit_prog_group_desc.hitgroup.moduleCH=optix_module_;
    // hit_prog_group_desc.hitgroup.entryFunctionNameCH="__closesthit__ch";

    char log[2048];
    size_t sizeof_log=sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_,&hit_prog_group_desc,1, // numProgramGroups
        &program_group_options,log,&sizeof_log,&hitgroup_prog_groups_[0]
    ));
    // if(sizeof_log > 1) PRINT(log);
}

//TODO: assembles the full pipeline of all programs
void RTBase::CreatePipeline(){
    std::vector<OptixProgramGroup> program_groups;
    for(auto &pg:raygen_prog_groups_)
        program_groups.push_back(pg);
    for(auto &pg:miss_prog_groups_)
        program_groups.push_back(pg);
    for(auto &pg:hitgroup_prog_groups_)
        program_groups.push_back(pg);

    //* OptixPipelineLinkOptions
    const uint32_t max_trace_depth=1;
    pipeline_link_options_.maxTraceDepth=max_trace_depth;
#if OPTIX_VERSION < 70700
    // pipeline level settings for debugging
    pipeline_link_options_.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    //* create optix pipeline
    char log[2048];
    size_t sizeof_log=sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
        optix_context_,
        &pipeline_compile_options_,
        &pipeline_link_options_,
        program_groups.data(),
        (int)program_groups.size(),
        log,
        &sizeof_log,
        &optix_pipeline_
    ));
    // if (sizeof_log > 1) PRINT(log);

    //* compute stack size
    OptixStackSizes stack_sizes={};
    for(auto& pg:program_groups){
    #if OPTIX_VERSION >= 70700
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg,&stack_sizes,optix_pipeline_));
    #else 
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg,&stack_sizes));
    #endif
    }
    //* variables for stack size
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        0,0, // maxCCDepth, maxDCDepth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));
    OPTIX_CHECK(optixPipelineSetStackSize(optix_pipeline_,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        1 // unsigned int maxTraversableGraphDepth
    ));
}

//* 
void RTBase::CreateSBT(){
    //* ===== build raygen records =====
    std::vector<RayGenRecord> raygen_records(raygen_prog_groups_.size());
    for(int i=0;i<raygen_prog_groups_.size();i+=1){
        RayGenRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_groups_[i],&rec));
        raygen_records[i]=rec;
    }
    CUdeviceptr d_raygen_records=0;
    const size_t raygen_records_size=sizeof(RayGenRecord)*raygen_records.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_records),raygen_records_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_records),raygen_records.data(),raygen_records_size,cudaMemcpyHostToDevice));
    sbt_.raygenRecord=d_raygen_records;
    //* ===== build miss records =====
    std::vector<MissRecord> miss_records(miss_prog_groups_.size());
    for(int i=0;i<miss_prog_groups_.size();i+=1){
        MissRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_groups_[i],&rec));
        miss_records[i]=rec;
    }
    CUdeviceptr d_miss_records=0;
    const size_t miss_records_size=sizeof(MissRecord)*miss_records.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records),miss_records_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_records),miss_records.data(),miss_records_size,cudaMemcpyHostToDevice));
    sbt_.missRecordBase=d_miss_records;
    sbt_.missRecordStrideInBytes=sizeof(MissRecord);
    sbt_.missRecordCount=(uint)miss_records.size();
    //* ===== uild hitgroup records =====
    std::vector<HitGroupRecord> hitgroup_records(hitgroup_prog_groups_.size());
    for(int i=0;i<hitgroup_prog_groups_.size();i+=1){
        HitGroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_groups_[i],&rec));
        hitgroup_records[i]=rec;
    }
    CUdeviceptr d_hitgroup_records=0;
    const size_t hitgroup_records_size=sizeof(HitGroupRecord)*hitgroup_records.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_records),hitgroup_records_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_records),hitgroup_records.data(),hitgroup_records_size,cudaMemcpyHostToDevice));
    sbt_.hitgroupRecordBase=d_hitgroup_records;
    sbt_.hitgroupRecordStrideInBytes=sizeof(HitGroupRecord);
    sbt_.hitgroupRecordCount=(uint)hitgroup_records.size();
}

void RTBase::BuildAccel(const model_t &triangle_model,bool if_compact){
    //* copy vertices to device
    CUdeviceptr d_vertices=0;
    const size_t vertices_size=sizeof(float3)*triangle_model.vertices.size();
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices),vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices),triangle_model.vertices.data(),vertices_size,cudaMemcpyHostToDevice));
    //* built-in triangle 
    OptixBuildInput triangle_input={};
    triangle_input.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    //* vertex setting
    triangle_input.triangleArray.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes=sizeof(float3);
    triangle_input.triangleArray.numVertices=static_cast<uint32_t>(triangle_model.vertices.size());
    triangle_input.triangleArray.vertexBuffers=&d_vertices; // vertices on device
    //* index setting
    /*
    triangle_input.triangleArray.indexFormat=OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes=sizeof(int3);
    triangle_input.triangleArray.numIndexTriplets=;
    triangle_input.triangleArray.indexBuffer=;
    */
    const uint32_t triangle_input_flag[1]={OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};//
    triangle_input.triangleArray.flags=triangle_input_flag; // Array of flags
    triangle_input.triangleArray.numSbtRecords=1;
    triangle_input.triangleArray.sbtIndexOffsetBuffer=0;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes=0;
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes=0;
    //*
    /*
    enum OptixBuildFlags:
        OPTIX_BUILD_FLAG_NONE, OPTIX_BUILD_FLAG_ALLOW_UPDATE, OPTIX_BUILD_FLAG_ALLOW_COMPACTION,
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE, OPTIX_BUILD_FLAG_PREFER_FAST_BUILD,
        OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS, OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS
    */
    OptixAccelBuildOptions accel_options={};
    accel_options.buildFlags=OPTIX_BUILD_FLAG_ALLOW_COMPACTION; // add by "|"
    accel_options.operation=OPTIX_BUILD_OPERATION_BUILD;
    //* compute GAS size
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context_,
        &accel_options,
        &triangle_input,
        1, // Number of build inputs
        &gas_buffer_sizes
    ));
    //* record gas memory size
    bvh_memory_size_=gas_buffer_sizes.outputSizeInBytes;
    //* operation before compaction
    size_t* d_compacted_size;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_size),sizeof(size_t)));
    OptixAccelEmitDesc emit_property={};
    emit_property.type=OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_property.result=(CUdeviceptr)d_compacted_size;
    CUdeviceptr d_temp_buffer_gas; // temporarily used during the building
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),gas_buffer_sizes.tempSizeInBytes));
    CUdeviceptr d_gas_output_buffer; // memory size of acceleration structure
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer),gas_buffer_sizes.outputSizeInBytes));
    //* build
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    OPTIX_CHECK(optixAccelBuild(
        optix_context_, 
        cuda_stream_,
        &accel_options,
        &triangle_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle_,
        &emit_property, // emitted property list
        1 // num emitted properties
    ));
    d_gas_output_memory_=d_gas_output_buffer;
    cpu_timer.StopTiming();
    bvh_building_time_=cpu_timer.GetElapsedTime();
    //* perform compaction
    size_t compacted_size;
    CUDA_CHECK(cudaMemcpy(&compacted_size,d_compacted_size,sizeof(size_t),cudaMemcpyDeviceToHost));
    CUdeviceptr d_compacted_output_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_output_buffer),compacted_size));
    if(if_compact&&compacted_size<gas_buffer_sizes.outputSizeInBytes){
        CPUTimer cpu_timer;
        cpu_timer.StartTiming();
        OPTIX_CHECK(optixAccelCompact(
            optix_context_,
            cuda_stream_,
            gas_handle_,
            d_compacted_output_buffer,
            compacted_size,
            &gas_handle_
        ));
        // cudaDeviceSynchronize();
        cpu_timer.StopTiming();
        bvh_compacted_time_=cpu_timer.GetElapsedTime();
        d_gas_output_memory_=d_compacted_output_buffer;
        cudaFree(reinterpret_cast<void*>(d_gas_output_buffer));
        //* compacted gas memory size
        d_gas_output_memory_=compacted_size;
    }
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
}

// void RTBase::CreateProgramGroups(){
//     CreateRaygenPrograms();
//     CreateMissPrograms();
//     CreateHitgroupPrograms();
// }

RTBase::RTBase(int device_id){
    SetDevice(device_id);
    CreateContext();
    CreateModule();
    CreateRaygenPrograms();
    CreateMissPrograms();
    CreateHitgroupPrograms();
    CreatePipeline();
    CreateSBT();
}

RTBase::RTBase(){}

RTBase::~RTBase(){
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_memory_)));

    OPTIX_CHECK(optixPipelineDestroy(optix_pipeline_));
    for(auto &pg:raygen_prog_groups_)
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
    for(auto &pg:miss_prog_groups_)
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
    for(auto &pg:hitgroup_prog_groups_)
        OPTIX_CHECK(optixProgramGroupDestroy(pg));
    OPTIX_CHECK(optixModuleDestroy(optix_module_));
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context_));
    CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
}


