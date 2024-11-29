#pragma once

#include <optix.h>
#include <thrust/device_vector.h>

#include "helper.h"

void TranslateRaysOnDevice(Ray *d_rays, int *query, int query_size, SceneParameter scene_params,
                           float ray_offset = 0.0f);

void TranslateTrianglesOnDevice(float3 *d_vertices, int *data, int data_size,
                                SceneParameter scene_params, float size);

void TranslateSpheresOnDevice(float3 *d_vertices, float *d_radius, int *data, int data_size,
                              SceneParameter scene_params, float radius);

void TranslateAABBsOnDevice(OptixAabb *d_aabbs, float3 *d_centers, int *data, int data_size,
                            SceneParameter scene_params, float side);