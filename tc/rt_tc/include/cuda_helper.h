#pragma once

#include "model.h"
#include <stddef.h>

void ThrustSort(int *array, int n);

uint ThrustCount(uint *d_array, uint n, uint e);

uint ThrustReduce(uint *d_array, uint n);

void ConvertOnGPU(const int *adjs, const int *offsets, const graph_info_t &graph_info,
                  float3 axis_offset, float triangle_eps, float3 *vertices, uint num_of_vertices);

// TODO: for u-v-w
void TwoHopsFunction(const int *adjs, const int *offsets, const graph_info_t &graph_info,
                     std::vector<uint> &rt_ray_origin);

void ExpandonGPU(float3 *centers, float3 *triangle_vertices, uint num, float triangle_eps,
                 float3 axis_offset);
                 
void ExpandonCPU(float3 *centers, float3 *triangle_vertices, uint num, float triangle_eps,
                 float3 axis_offset);