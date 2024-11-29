#include <optix_function_table_definition.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <iostream>
#include <iomanip>

#include "RTBase.h"
#include "Record.h"
#include "Timing.h"
#include "casting_kernels.h"
#include "config.h"

extern "C" char embedded_ptx_code[];

#ifdef DEBUG
static void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}
#endif

void RTBase::SetDevice(int device_id=0){
    
    int device_count=0;
    cudaGetDeviceCount(&device_count);
    if(device_id<0 || device_id>=device_count){
        std::cerr<<"Device id should in [0, "<<device_count<<")"<<std::endl;
        exit(-1);
    }
    use_device_id_=device_id;
    cudaSetDevice(device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop,device_id);
    size_t available_memory,total_memory;
    cudaMemGetInfo(&available_memory,&total_memory);
    // std::cout<<"==========================================================\n";
    // std::cout<<"Total GPUs visible: "<<device_count;
    // std::cout<<", using ["<<device_id<<"]: "<<device_prop.name<<std::endl;
    // std::cout<<"Available Memory: "<<int(available_memory/1024/1024)<<" MB\n";
    // std::cout<<"Total Memory:     "<<int(total_memory/1024/1024)<<" MB\n";
    // std::cout<<"Shared Memory:    "<<device_prop.sharedMemPerBlock<<" B\n";
    // std::cout<<"Block Threads:    "<<device_prop.maxThreadsPerBlock<<"\n";
    // std::cout<<"SM Count:         "<<device_prop.multiProcessorCount<<"\n";
    // std::cout<<"==========================================================\n";
}

void RTBase::CreateContext(){
    CUDA_CHECK(cudaFree(0));
    OPTIX_CHECK(optixInit());
    cuCtxGetCurrent(&cuda_context_);
    CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    OptixDeviceContextOptions options={};
    options.logCallbackFunction=nullptr; // &context_log_cb;
    options.logCallbackLevel=4;
    // cuda_context_=0;
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_,&options,&optix_context_));
}

void RTBase::CreateModule(){
    module_compile_options_.maxRegisterCount=OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options_.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options_.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    pipeline_compile_options_.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options_.usesMotionBlur=false;
    pipeline_compile_options_.numPayloadValues=2; // ray payload
    pipeline_compile_options_.numAttributeValues=0; // attribute in optixReportIntersection()
    pipeline_compile_options_.exceptionFlags=OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options_.pipelineLaunchParamsVariableName="params";
#if PRIMITIVE == 0
    pipeline_compile_options_.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE; // scene contains nothing but built-in triangles
#elif PRIMITIVE == 1
    pipeline_compile_options_.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
#elif PRIMITIVE == 2

#endif

    const std::string ptx_code=embedded_ptx_code;
    char log[2048];
    size_t sizeof_log=sizeof(log);

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

// #if PRIMITIVE == 1
//     OptixBuiltinISOptions builtin_is_options={};
//     builtin_is_options.usesMotionBlur=false;
//     builtin_is_options.builtinISModuleType=OPTIX_PRIMITIVE_TYPE_SPHERE;
//     OPTIX_CHECK(optixBuiltinISModuleGet(
//         optix_context_,
//         &module_compile_options_,
//         &pipeline_compile_options_,
//         &builtin_is_options,
//         &sphere_module_
//     ));
// #endif
}

void RTBase::CreateProgramGroups(){
    char log[2048];
    size_t sizeof_log=sizeof(log);
    OptixProgramGroupOptions program_group_options={};

    OptixProgramGroupDesc raygen_prog_group_desc={};
    raygen_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module=optix_module_;
    raygen_prog_group_desc.raygen.entryFunctionName="__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_,&raygen_prog_group_desc,1, // num program groups
        &program_group_options,log,&sizeof_log,&raygen_prog_group_
    ));

    OptixProgramGroupDesc miss_prog_group_desc={};
    miss_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module=optix_module_;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_,&miss_prog_group_desc,1, // num program groups
        &program_group_options,log,&sizeof_log,&miss_prog_group_
    ));

    OptixProgramGroupDesc hit_prog_group_desc={};
    hit_prog_group_desc.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleAH=nullptr;//optix_module_;
    hit_prog_group_desc.hitgroup.entryFunctionNameAH=nullptr;//"__anyhit__ah";
    hit_prog_group_desc.hitgroup.moduleCH=optix_module_;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH="__closesthit__ch";

#if PRIMITIVE == 1
    hit_prog_group_desc.hitgroup.moduleIS=optix_module_;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS="__intersection__sphere";
#elif PRIMITIVE == 2
    hit_prog_group_desc.hitgroup.moduleIS=optix_module_;
    hit_prog_group_desc.hitgroup.entryFunctionNameIS="__intersection__aabb";
#endif

    OPTIX_CHECK(optixProgramGroupCreate(
        optix_context_,&hit_prog_group_desc,1,  // num program groups
        &program_group_options,log,&sizeof_log,&hitgroup_prog_group_
    ));
}

void RTBase::CreatePipeline(){
    char log[2048];
    size_t sizeof_log=sizeof(log);

    const uint32_t max_trace_depth=1;
    pipeline_link_options_.maxTraceDepth=1; // maximum recursion depth setting for recursive ray tracing
    pipeline_link_options_.debugLevel=OPTIX_COMPILE_DEBUG_LEVEL_FULL;// pipeline level settings for debugging
    OptixProgramGroup program_groups[3]={
        raygen_prog_group_,
        miss_prog_group_,
        hitgroup_prog_group_
    };

    OPTIX_CHECK(optixPipelineCreate(
        optix_context_,
        &pipeline_compile_options_,
        &pipeline_link_options_,
        program_groups,
        sizeof(program_groups)/sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &optix_pipeline_
    ));
    // ========================
    OptixStackSizes stack_sizes={};
    for(auto& prog_group : program_groups){
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group,&stack_sizes));
    }
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
        1  // maxTraversableDepth
    ));
}

void RTBase::CreateSBT(){
    // build raygen record
    CUdeviceptr d_raygen_record=0;
    const size_t raygen_record_size=sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record),raygen_record_size));
    RayGenRecord rg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group_,&rg_record));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record),&rg_record,raygen_record_size,cudaMemcpyHostToDevice));
    // build miss record
    CUdeviceptr d_miss_record=0;
    const size_t miss_record_size=sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record),miss_record_size));
    MissRecord ms_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group_,&ms_record));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record),&ms_record,miss_record_size,cudaMemcpyHostToDevice));
    // build hitgroup record
    CUdeviceptr d_hitgroup_record=0;
    const size_t hitgroup_record_size=sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record),hitgroup_record_size));
    HitGroupRecord hg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group_,&hg_record));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record),&hg_record,hitgroup_record_size,cudaMemcpyHostToDevice));
    // build sbt
    sbt_.raygenRecord=d_raygen_record;
    sbt_.missRecordBase=d_miss_record;
    sbt_.missRecordStrideInBytes=sizeof(MissRecord);
    sbt_.missRecordCount=1;
    sbt_.hitgroupRecordBase=d_hitgroup_record;
    sbt_.hitgroupRecordStrideInBytes=sizeof(HitGroupRecord);
    sbt_.hitgroupRecordCount=1;

}

void RTBase::BuildAccel(std::vector<int> &data){
    //* switch optix input 
    OptixBuildInput build_input={};
#if PRIMITIVE == 0
    //TODO: choose triangle
    std::cout<<"primitive type = triangle"<<std::endl;
    uint vertices_num=data.size()*3;
    // float triangle_size=0.2f;
    float3 *d_vertices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices),sizeof(float3)*vertices_num));
    //* translate data into triangles on device
    TranslateTrianglesOnDevice(d_vertices,data.data(),data.size(),scene_params_,triangle_size_);

    // std::vector<float3> host_triangles;
    // host_triangles.resize(vertices_num);
    // CUDA_CHECK(cudaMemcpy(host_triangles.data(),d_vertices,sizeof(float3)*vertices_num,cudaMemcpyDeviceToHost));
    // for(int i=0;i<data.size();i++){
    //     printf("(%f, %f, %f) ",host_triangles[i*3].x,host_triangles[i*3].y,host_triangles[i*3].z);
    //     printf("(%f, %f, %f) ",host_triangles[i*3+1].x,host_triangles[i*3+1].y,host_triangles[i*3+1].z);
    //     printf("(%f, %f, %f)\n",host_triangles[i*3+2].x,host_triangles[i*3+2].y,host_triangles[i*3+2].z);
    // }

    CUdeviceptr d_vertex_buffer=reinterpret_cast<CUdeviceptr>(d_vertices);
    build_input_buffer_.emplace_back(d_vertex_buffer); // free memory

    build_input.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes=sizeof(float3);
    build_input.triangleArray.numVertices=vertices_num; //* number of vertcies
    build_input.triangleArray.vertexBuffers=&d_vertex_buffer;
    // //* one SBT entry and no-primitive materials
    const uint32_t triangle_input_flag[1]={OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};//{OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};//
    build_input.triangleArray.flags=triangle_input_flag;
    build_input.triangleArray.numSbtRecords=1;

#elif PRIMITIVE == 1
    //TODO: choose sphere
    std::cout<<"primitive type = sphere"<<std::endl;
    //* translate
    uint vertices_num=data.size();
    // float3 *d_vertices;
    float *d_radius;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centers_),sizeof(float3)*vertices_num)); // center of sphere
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_radius),sizeof(float))); // single radius for all spheres
    TranslateSpheresOnDevice(d_centers_,d_radius,data.data(),data.size(),scene_params_,sphere_radius_);
    //* 
    CUdeviceptr d_vertex_buffer=reinterpret_cast<CUdeviceptr>(d_centers_);
    CUdeviceptr d_radius_buffer=reinterpret_cast<CUdeviceptr>(d_radius);
    // build_input_buffer_.emplace_back(d_vertex_buffer);
    build_input_buffer_.emplace_back(d_radius_buffer);

    build_input.type=OPTIX_BUILD_INPUT_TYPE_SPHERES;
    build_input.sphereArray.vertexBuffers=&d_vertex_buffer;
    build_input.sphereArray.numVertices=vertices_num;
    build_input.sphereArray.radiusBuffers=&d_radius_buffer;
    build_input.sphereArray.singleRadius=true; // Boolean value indicating whether a single radius per radius buffer is used,
    const uint32_t sphere_input_flags[1]={OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};
    build_input.sphereArray.flags=sphere_input_flags;
    build_input.sphereArray.numSbtRecords=1;
#else
    //TODO: choose aabb
    std::cout<<"primitive type = aabbs"<<std::endl;
    //* translate
    uint aabbs_num=data.size();
    OptixAabb *d_aabbs;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabbs),sizeof(OptixAabb)*aabbs_num));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_centers_),sizeof(float3)*aabbs_num));
    TranslateAABBsOnDevice(d_aabbs,d_centers_,data.data(),data.size(),scene_params_,aabb_side_);
    CUdeviceptr d_aabb_buffer=reinterpret_cast<CUdeviceptr>(d_aabbs);
    build_input_buffer_.emplace_back(d_aabb_buffer);

    // std::vector<OptixAabb> host_aabbs;
    // host_aabbs.resize(aabbs_num);
    // CUDA_CHECK(cudaMemcpy(host_aabbs.data(),d_aabbs,sizeof(OptixAabb)*aabbs_num,cudaMemcpyDeviceToHost));
    // for(int i=0;i<data.size();i++){
    //     printf("(%f, %f, %f) ",host_aabbs[i].minX,host_aabbs[i].minY,host_aabbs[i].minZ);
    //     printf("(%f, %f, %f)\n",host_aabbs[i].maxX,host_aabbs[i].maxY,host_aabbs[i].maxZ);
    // }

    build_input.type=OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    build_input.customPrimitiveArray.aabbBuffers=&d_aabb_buffer;
    build_input.customPrimitiveArray.numPrimitives=aabbs_num; //* number of aabb
    const uint32_t aabb_input_flags[1]={OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT}; //{OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};//
    build_input.customPrimitiveArray.flags=aabb_input_flags;
    build_input.customPrimitiveArray.numSbtRecords=1;
#endif

    // Use default options for simplicity.
    OptixAccelBuildOptions accel_options={};
    accel_options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;// OPTIX_BUILD_FLAG_ALLOW_COMPACTION, OPTIX_BUILD_FLAG_NONE
    accel_options.operation=OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context_,
        &accel_options,
        &build_input,
        1, // Number of build inputs
        &gas_buffer_sizes
    ));

    /* prepare compaction */
    // size_t* d_compacted_size;
    // cudaMalloc(reinterpret_cast<void**>(&d_compacted_size),sizeof(size_t));
    // OptixAccelEmitDesc emit_property={};
    // emit_property.type=OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    // emit_property.result=(CUdeviceptr)d_compacted_size;

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas),gas_buffer_sizes.tempSizeInBytes));
    CUdeviceptr d_gas_output_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer),gas_buffer_sizes.outputSizeInBytes));
    CUDA_SYNC_CHECK();
    cudaEvent_t build_start,build_end;
    CUDA_CHECK(cudaEventCreate(&build_start));
    CUDA_CHECK(cudaEventCreate(&build_end));
    CUDA_CHECK(cudaEventRecord(build_start,cuda_stream_));
    OPTIX_CHECK(optixAccelBuild(
        optix_context_, 
        cuda_stream_, // CUDA stream
        &accel_options,
        &build_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle_,
        nullptr,//&emit_property, // emitted property list
        0 // num of emitted properties
    ));
    CUDA_CHECK(cudaEventRecord(build_end,cuda_stream_));
    CUDA_CHECK(cudaEventSynchronize(build_end));
    CUDA_CHECK(cudaEventElapsedTime(&build_time_,build_start,build_end));
    CUDA_CHECK(cudaEventDestroy(build_start));
    CUDA_CHECK(cudaEventDestroy(build_end));
    // cudaDeviceSynchronize();
    CUDA_SYNC_CHECK();
    // timer.EndTiming();
    // build_time_=timer.GetTime();

    /* perform compaction */
    // size_t compacted_size;
    // cudaMemcpy(&compacted_size,d_compacted_size,sizeof(size_t),cudaMemcpyDeviceToHost);
    // void *d_compacted_output_buffer;
    // cudaMalloc(&d_compacted_output_buffer,compacted_size);
    /*  avoid the compacting pass in cases where it is not beneficial */
    // if(compacted_size<gas_buffer_sizes.outputSizeInBytes){
    //     OPTIX_CHECK(optixAccelCompact(
    //         optix_context_,
    //         cuda_stream_,
    //         gas_handle_,
    //         (CUdeviceptr)d_compacted_output_buffer,
    //         compacted_size,
    //         &gas_handle_
    //     ));
    // }
    /* clear up */
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    //* free the memory of buffer
    for(auto e:build_input_buffer_)
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(e)));

    d_gas_output_buffer_=d_gas_output_buffer;
    printf("BVH Building Time = %f ms\n",build_time_);
}

void RTBase::Search(std::vector<int> &query,int *results){
    //* translate queries into rays
    float ray_offset=0.0f;
#if PRIMITIVE == 0
    ray_offset=0.1f;
#endif
    Ray *d_rays;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rays),sizeof(Ray)*query.size()));
    TranslateRaysOnDevice(d_rays,query.data(),query.size(),scene_params_,ray_offset);
    //* launch rays
    h_params_.handle=gas_handle_;
    h_params_.rays=d_rays;

#if PRIMITIVE == 1
    h_params_.centers=d_centers_;
    h_params_.side=sphere_radius_;
#elif PRIMITIVE == 2
    h_params_.centers=d_centers_;
    h_params_.side=aabb_side_;
#endif

    // std::vector<Ray> host_rays;
    // host_rays.resize(query.size());
    // CUDA_CHECK(cudaMemcpy(host_rays.data(),d_rays,sizeof(Ray)*query.size(),cudaMemcpyDeviceToHost));
    // for(int i=0;i<query.size();i++){
    //     printf("(%f, %f, %f) ",host_rays[i].origin.x,host_rays[i].origin.y,host_rays[i].origin.z);
    // }
    // printf("\n");
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&h_params_.ray_origins),sizeof(float3)*ray_origins.size()));
    // CUDA_CHECK(cudaMemcpy(h_params_.ray_origins,ray_origins.data(),sizeof(float3)*ray_origins.size(),cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&h_params_.ray_lengths),sizeof(float)*ray_lengths.size()));
    // CUDA_CHECK(cudaMemcpy(h_params_.ray_lengths,ray_lengths.data(),sizeof(float)*ray_lengths.size(),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&h_params_.results),sizeof(size_t)*query.size()));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params_ptr_),sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params_ptr_),&h_params_,sizeof(LaunchParams),cudaMemcpyHostToDevice));
    
    cudaEvent_t launch_start,launch_end;
    CUDA_CHECK(cudaEventCreate(&launch_start));
    CUDA_CHECK(cudaEventCreate(&launch_end));
    CUDA_CHECK(cudaEventRecord(launch_start,cuda_stream_));
    OPTIX_CHECK(optixLaunch(
        optix_pipeline_,
        cuda_stream_,
        reinterpret_cast<CUdeviceptr>(d_params_ptr_),
        sizeof(LaunchParams),
        &sbt_,
        query.size(),
        1,1
    ));
    CUDA_CHECK(cudaEventRecord(launch_end,cuda_stream_));
    CUDA_CHECK(cudaEventSynchronize(launch_end));
    CUDA_CHECK(cudaEventElapsedTime(&search_time_,launch_start,launch_end));
    CUDA_CHECK(cudaEventDestroy(launch_start));
    CUDA_CHECK(cudaEventDestroy(launch_end));
    
    printf("RT Search Time = %f ms\n",search_time_);
    cudaMemcpy(results,h_params_.results,sizeof(int)*query.size(),cudaMemcpyDeviceToHost);
    
    // cudaFree(h_params_.ray_lengths);
    CUDA_CHECK(cudaFree(h_params_.rays));
    CUDA_CHECK(cudaFree(h_params_.results));
    CUDA_CHECK(cudaFree(d_params_ptr_));
}

void RTBase::CleanUp(){
    // printf("Clean up RT ...\n");
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt_.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer_)));

    OPTIX_CHECK(optixPipelineDestroy(optix_pipeline_));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group_));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group_));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group_));
    OPTIX_CHECK(optixModuleDestroy(optix_module_));
    CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    OPTIX_CHECK(optixDeviceContextDestroy(optix_context_));
    
#if PRIMITIVE != 0
    CUDA_CHECK(cudaFree(d_centers_)); // center of sphere or aabb
#endif
}

void RTBase::Setup(){
    // printf("Create optix context ...\n");
    CreateContext();
    // printf("Create optix module ...\n");
    CreateModule();
    // printf("Create optix program groups ...\n");
    CreateProgramGroups();
    // printf("Create optix pipeline ...\n");
    CreatePipeline();
    // printf("Create optix SBT ...\n");
    CreateSBT();
}

RTBase::RTBase(const SceneParameter &parameter):scene_params_(parameter){

}

RTBase::~RTBase(){
    // printf("Clean up optix ...\n");
    // CleanUp();
}

// void RTBase::PrintInfo(){
//     printf("RT Search Time = %f ms",search_time_);
//     printf(", BVH Building Time = %f ms\n",build_time_);
    
// }