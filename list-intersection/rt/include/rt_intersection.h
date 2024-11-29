#pragma once

#include "rt_base.h"
#include "launch_params.h"
#include "util.h"

class RTInter: public RTBase{
private:
    LaunchParams launch_params_;
    std::vector<uint> rt_ray_origins_;
    std::vector<uint> ray_set_ids_;
    std::vector<uint> triangle_set_ids_;
    uint rays_{0};
    uint ray_length_{0};
    float3 axis_offset_;
    float triangle_eps_{0.0f};
    uint chunk_length_{0}; // chunk length in z axis
    size_t combinations_{0};
    double rt_counting_time_{0.0};

    void ConvertSetToRT(tile_t tile,const std::vector<uint> &elements,const std::vector<uint> &set_offsets,model_t &triangle_model);
    void ComputeRays(tile_t tile,const std::vector<uint> &elements,const std::vector<uint> &set_offsets);
    void CenterToTriangle(float3 center,std::vector<float3> &vertices,uint pos);
public:
    RTInter(uint chunk_length,int device_id=0);
    ~RTInter();
    void BuildBVHAndComputeRay(Dataset dataset);
    void CountIntersection(std::vector<uint> &results);
};