#pragma once

#include <optix.h>

struct LaunchParams {
    OptixTraversableHandle handle;
    int chunk_length;
    int adjust;
    float zoom;
    int current_level; // current level
    int *origins;      // ray origins in current step
    int *triangle_id;  // primitive id -> node id
    int *queue;        // nodes expanded in current step
    int *queue_size;   //
    int *levels;       // level of each node
    int *ray_length;
};

//* ===== BFS with encoding =====
struct LaunchParamsV2 {
    OptixTraversableHandle handle;
    int nodes;            // number of nodes in graph
    float max_ray_length; // maximal ray length
    int adjust;
    float3 zoom;

    int encode_digits;
    int encode_mod;

    int *origins; // ray origins in current step
    // float3* triangle_center; // primitive id -> node id

    int current_level; // current level
    int *queue;        // nodes expanded in current step
    int *queue_size;   //
    int *levels;       // level of each node
};