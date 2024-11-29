#include "casting_kernels.h"
#include "helper.h"

#define BLOCK_SIZE 256
#define EPS 0.2f

__global__ void TranslateRaysKernel(int *query, int num, Ray *rays, SceneParameter scene_params,
                                    float ray_offset) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float3 offsets = scene_params.axis_offsets;
    for (int i = tid; i < num; i += blockDim.x * gridDim.x) {
        int x = query[i] % scene_params.mod;
        int y = query[i] / (scene_params.mod * scene_params.mod);
        int z = query[i] / scene_params.mod;
        rays[i].origin = make_float3(x - offsets.x, y - ray_offset - offsets.y, z - offsets.z);
        rays[i].length = 0.2f; //* length of rays
    }
}

void TranslateRaysOnDevice(Ray *d_rays, int *query, int query_size, SceneParameter scene_params,
                           float ray_offset) {
    thrust::device_vector<int> d_query(query, query + query_size);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, TranslateRaysKernel, BLOCK_SIZE,
                                                  0);
    int blocks = blocks_per_sm * 100;
    TranslateRaysKernel<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_query.data()),
                                                query_size, d_rays, scene_params, ray_offset);
    CUDA_SYNC_CHECK();
}

//* Triangles:
__forceinline__ __device__ void AddTriangle(float3 *vertices, const float3 &center, int idx,
                                            float size) {
    float upper_x = center.x + size;
    float below_x = center.x - size;
    // float upper_y=center.y+add;
    // float below_y=center.y-add;
    float upper_z = center.z + size;
    float below_z = center.z - size;
    vertices[idx * 3] = make_float3(center.x, center.y, upper_z);
    vertices[idx * 3 + 1] = make_float3(upper_x, center.y, below_z);
    vertices[idx * 3 + 2] = make_float3(below_x, center.y, below_z);
}

__global__ void TranslateTrianglesKernel(int *data, int num, float3 *vertices,
                                         SceneParameter scene_params, float size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float3 offsets = scene_params.axis_offsets;
    for (int i = tid; i < num; i += blockDim.x * gridDim.x) {
        int x = data[i] % scene_params.mod;
        int y = data[i] / (scene_params.mod * scene_params.mod);
        int z = data[i] / scene_params.mod;
        float3 center = make_float3(x - offsets.x, y - offsets.y, z - offsets.z);
        AddTriangle(vertices, center, i, size);
    }
}

void TranslateTrianglesOnDevice(float3 *d_vertices, int *data, int data_size,
                                SceneParameter scene_params, float size) {
    thrust::device_vector<int> d_data(data, data + data_size);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, TranslateTrianglesKernel,
                                                  BLOCK_SIZE, 0);
    int blocks = blocks_per_sm * 100;
    TranslateTrianglesKernel<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_data.data()),
                                                     data_size, d_vertices, scene_params, size);
    CUDA_SYNC_CHECK();
}

//* Spheres:

__global__ void TranslateSpheresKernel(int *data, int num, float3 *vertices,
                                       SceneParameter scene_params) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float3 offsets = scene_params.axis_offsets;
    for (int i = tid; i < num; i += blockDim.x * gridDim.x) {
        int x = data[i] % scene_params.mod;
        int y = data[i] / (scene_params.mod * scene_params.mod);
        int z = data[i] / scene_params.mod;
        float3 center = make_float3(x - offsets.x, y - offsets.y, z - offsets.z);
        vertices[i] = center;
        // radius[i]=r;
    }
}

void TranslateSpheresOnDevice(float3 *d_vertices, float *d_radius, int *data, int data_size,
                              SceneParameter scene_params, float radius) {
    thrust::device_vector<int> d_data(data, data + data_size);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, TranslateSpheresKernel,
                                                  BLOCK_SIZE, 0);
    int blocks = blocks_per_sm * 100;
    TranslateSpheresKernel<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_data.data()),
                                                   data_size, d_vertices, scene_params);
    CUDA_CHECK(cudaMemcpy(d_radius, &radius, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SYNC_CHECK();
}

//* AABBs:
__forceinline__ __device__ void AddAABB(OptixAabb *aabbs, const float3 &center, int idx,
                                        float side) {
    aabbs[idx] = {center.x - side, center.y - side, center.z - side,
                  center.x + side, center.y + side, center.z + side};
}

__global__ void TranslateAABBsKernel(int *data, int num, OptixAabb *aabbs, float3 *centers,
                                     SceneParameter scene_params, float side) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float3 offsets = scene_params.axis_offsets;
    for (int i = tid; i < num; i += blockDim.x * gridDim.x) {
        int x = data[i] % scene_params.mod;
        int y = data[i] / (scene_params.mod * scene_params.mod);
        int z = data[i] / scene_params.mod;
        float3 center = make_float3(x - offsets.x, y - offsets.y, z - offsets.z);
        centers[i] = center;
        AddAABB(aabbs, center, i, side);
    }
}

void TranslateAABBsOnDevice(OptixAabb *d_aabbs, float3 *d_centers, int *data, int data_size,
                            SceneParameter scene_params, float side) {
    thrust::device_vector<int> d_data(data, data + data_size);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, TranslateAABBsKernel, BLOCK_SIZE,
                                                  0);
    int blocks = blocks_per_sm * 100;
    TranslateAABBsKernel<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_data.data()), data_size,
                                                 d_aabbs, d_centers, scene_params, side);
    CUDA_SYNC_CHECK();
}