#pragma once

#include "config.h"
#include "launch_params.h"
#include "rt_base.h"

class RTTC : public RTBase {
private:
#if RTTC_METHOD == 0
    ListIntersectionParams launch_params_;
    std::vector<uint> rt_ray_length_;
    std::vector<uint> rt_ray_offset_; //
#ifdef LIST_OPTIMIZATION
    std::vector<uint> triangle_vals_;
    std::vector<float2> rt_ray_origin_;
#else
    std::vector<uint> rt_ray_origin_;
#endif
#elif RTTC_METHOD == 1
    HashmapParams launch_params_;
    std::vector<uint> rt_ray_origin_;
#else
    LaunchParams launch_params_;
#endif

    float3 axis_offset_;
    float rt_triangle_eps_{0.2f};
    uint rt_triangles_{0}; // number of triangles
    uint rt_launch_size_{0};

    void ConvertGraphToRT(const int *adjs, const int *offsets, const graph_info_t &graph_info,
                          model_t &model);
#if RTTC_METHOD == 0
    void MethodBasedOnListIntersection(const int *adjs, const int *offsets,
                                       const graph_info_t &graph_info, model_t &model);
#elif RTTC_METHOD == 1
    void MethodBasedOnHashMap(const int *adjs, const int *offsets, const graph_info_t &graph_info,
                              model_t &model);
#endif

    void SetupLaunchParameters();
    void FreeLaunchParameters();
    void ComputeRays(const int *adjs, const int *offsets, const graph_info_t &graph_info);

public:
    uint count_size_{0};
    double convert_time_{0.0};
    double counting_time_{0.0};
    uint total_triangles_{0};
#if RTTC_METHOD == 1
    uint max_rays_{100000000U};
#endif

public:
    RTTC(int device_id = 0);
    ~RTTC();
    void BuildBVHAndComputeRay(const int *adjs, const int *offsets, const graph_info_t &graph_info,
                               bool if_compact = false);
    void CountTriangles();
};