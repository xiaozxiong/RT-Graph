#include "cuda_helper.h"
#include "timer.h"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// #define MaxBlocks 65535
#define ThreadsPerBlock 128
#define WarpSize 32
#define WarpsPerBlock (ThreadsPerBlock / WarpSize)
#define FULL_MASK 0xFFFFFFFF

// __global__ void TriangleKernel(int nodes,int* adjs,int* offsets,float3*
// vertices,int *node_id){
//     int tid=threadIdx.x+blockIdx.x*blockDim.x;
//     int warp_id=(tid>>5);
//     int lane_id=(tid&((1<<5)-1));
//     for(int i=warp_id;i<nodes;i+=gridDim.x*WarpsPerBlock){
//         int start=offsets[i];
//         int end=offsets[i+1];
//         for(int j=start+lane_id;j<end;j+=WarpSize){
//             int adj_id=adjs[j];
//             //...
//         }

//     }
// }

// //TODO:
// void GenerateTriangles(int nodes,int edges,int *adjs,int* offsets,float3*
// &d_vertices,int* &d_node_id){
//     int *d_adjs,* d_offsets;
//     cudaMalloc(&d_adjs,sizeof(int)*edges);
//     cudaMemcpy(d_adjs,adjs,sizeof(int)*edges,cudaMemcpyHostToDevice);
//     cudaMalloc(&d_offsets,sizeof(int)*(nodes+1));
//     cudaMemcpy(d_offsets,offsets,sizeof(int)*(nodes+1),cudaMemcpyHostToDevice);

//     int num_of_vertex=3*edges;
//     cudaMalloc(&d_vertices,sizeof(float3)*num_of_vertex);
//     cudaMalloc(&d_node_id,sizeof(int)*edges);

//     cudaFree(d_adjs);
//     cudaFree(d_offsets);

// }

// TODO: node -> origins
__global__ void OriginKernelThread(int *node_list, int nodes, int *origin_offset, int *origin_num,
                                   int *origin_list) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid / WarpSize;
    int lane_id = tid % WarpSize;
    for (int i = warp_id; i < nodes; i += gridDim.x * WarpsPerBlock) {
        int node_id = node_list[i];

        int start = origin_offset[node_id];
        int end = origin_offset[node_id + 1];
        int pos = 0;
        if (lane_id == 0)
            pos = atomicAdd(origin_num, end - start);
        __syncwarp();
        pos = __shfl_sync(FULL_MASK, pos, 0);
        for (int j = start + lane_id; j < end; j += WarpSize) {
            origin_list[pos + j - start] = j;
        }
    }
}

double GetOriginsByNodes(int nodes, int *d_node_list, int *d_origin_offset, int *d_origin_num,
                         int *d_origin_list, bool filter) {
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();

    int new_nodes;
    //* ========= filter =========
    if (filter) {
        thrust::sort(thrust::device, d_node_list, d_node_list + nodes);
        new_nodes = thrust::unique(thrust::device, d_node_list, d_node_list + nodes) - d_node_list;
    }
    //* ========= no filter ========
    else {
        new_nodes = nodes;
    }
    //* ====================
    int block_size = ThreadsPerBlock;
    int blocks = (new_nodes + WarpsPerBlock - 1) / WarpsPerBlock;
    OriginKernelThread<<<blocks, block_size>>>(d_node_list, new_nodes, d_origin_offset,
                                               d_origin_num, d_origin_list);
    cudaDeviceSynchronize();

    cpu_timer.StopTiming();
    return cpu_timer.GetElapsedTime();
}

void ThrustFill(int *array, int size, int val) {
    thrust::fill(thrust::device, array, array + size, val);
}