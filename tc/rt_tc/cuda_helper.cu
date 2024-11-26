#include "cuda_helper.h"
#include "config.h"
#include "common.h"
#include "timer.h"

#include <omp.h>

#include <assert.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <cub/cub.cuh>

#define MAX_BLOCKS 65535
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE/WARP_SIZE)
#define FULL_MASK 0xFFFFFFFF
#define ELEMENTS_PER_THREAD 16


void ThrustSort(int *array,int n){
    thrust::device_vector<int> d_vector(array,array+n);
    thrust::sort(d_vector.begin(),d_vector.end(),thrust::less<int>()); // 
    thrust::copy(d_vector.begin(),d_vector.end(),array);
}

uint ThrustCount(uint *d_array, uint n, uint e){
    return thrust::count(thrust::device,d_array,d_array+n, 0);
}

uint ThrustReduce(uint *d_array,uint n){
    return thrust::reduce(thrust::device,d_array,d_array+n);
}

__device__ void CenterToTriangle(float3 center,float eps,float3 *vertices,uint pos){
#if RTTC_METHOD == 0
    float below_y=center.y-eps;
    float above_y=center.y+eps;
    float below_z=center.z-eps;
    float above_z=center.z+eps;
    vertices[pos]=make_float3(center.x,center.y,above_z);
    vertices[pos+1]=make_float3(center.x,below_y,below_z);
    vertices[pos+2]=make_float3(center.x,above_y,below_z);
#elif RTTC_METHOD == 1
    float below_x=center.x-eps;
    float above_x=center.x+eps;
    float below_z=center.z-eps;
    float above_z=center.z+eps;
    vertices[pos]=make_float3(center.x,center.y,above_z);
    vertices[pos+1]=make_float3(below_x,center.y,below_z);
    vertices[pos+2]=make_float3(above_x,center.y,below_z);
#else

#endif
}

#if RTTC_METHOD == 0
__global__
void ListIntersectionVerticesKernel(int *adjs,int *offsets,uint nodes,float eps,float3 *vertices,uint *vertex_count,float3 axis_offset){
    int node_id=blockIdx.x;
    __shared__ uint start_pos[WARPS_PER_BLOCK];
    while(node_id<nodes){
        const int start=offsets[node_id];
        const int end=offsets[node_id+1];
        const uint adj_size=end-start;
        if(adj_size>0){
            int warp_id=(threadIdx.x>>5);
            for(int i=warp_id;i<adj_size;i+=WARPS_PER_BLOCK){ // 1-hop adjs
                const int adj=adjs[start+i];
                const int adj_start=offsets[adj];
                const int adj_end=offsets[adj+1];
                const uint two_adj_size=adj_end-adj_start;
                int lane_id=(threadIdx.x&31);
                if(lane_id==0) start_pos[warp_id]=atomicAdd(vertex_count,3U*two_adj_size);
                __syncwarp(FULL_MASK);
                for(int j=lane_id;j<two_adj_size;j+=WARP_SIZE){ // 2-hop adjs
                    const int two_adj=adjs[adj_start+j];
                    float3 center=make_float3(i-axis_offset.x,node_id-axis_offset.y,two_adj-axis_offset.z);
                    CenterToTriangle(center,eps,vertices,start_pos[warp_id]+3*j);
                }
            }
        }
        node_id+=gridDim.x;
    }
}
#endif

#if RTTC_METHOD == 1
__global__
void HashmapVerticesKernel(int *adjs,int *offsets,uint nodes,float eps,float3 *vertices,uint *vertex_count,float3 axis_offset){
    int node_id=blockIdx.x;
    while(node_id<nodes){
        // printf("node id = %d\n",node_id);
        const int start=offsets[node_id];
        const int end=offsets[node_id+1];
        const uint adj_size=end-start;
        if(adj_size>0){
            __shared__ uint start_pos;
            if(threadIdx.x==0) start_pos=atomicAdd(vertex_count,3U*adj_size);
            __syncthreads();
            for(int i=threadIdx.x;i<adj_size;i+=blockDim.x){
                int adj_id=adjs[start+i];
                float3 center=make_float3(node_id-axis_offset.x,0-axis_offset.y,adj_id-axis_offset.z);
                CenterToTriangle(center,eps,vertices,start_pos+3*i);
            }
        }
        node_id+=gridDim.x;
        __syncthreads();
    }
}

//TODO: get u-v-w on GPU

typedef struct EdgePair{
    int w;
    uint count;

    EdgePair() = default;

    __host__ __device__
    EdgePair(int x,uint y):w(x), count(y){};

    __host__ __device__
    EdgePair& operator = (const EdgePair& p) {
        w = p.w;
        count = p.count;
        return *this;
    };

    __host__ __device__
    bool operator == (const EdgePair& p) const {
        return (w == p.w) && (count == p.count);
    };

    __host__ __device__
    bool operator < (const EdgePair& p) const {
        if(w == p.w)
            return count < p.count;
        return count <p.count;
    };
    
} pair_t;


__global__ 
void CountKernel(int *adjs,int *offsets,int nodes, uint *wedge_counts){
    //* traversal
    // for(int node_id=blockIdx.x;node_id<nodes;node_id+=gridDim.x){
    //     int start=offsets[node_id];
    //     int end=offsets[node_id+1];
    //     // thread per 1-hop neighbor
    //     for(int i=start+threadIdx.x;i<end;i+=blockDim.x){
    //         int neighbor=adjs[i];
    //         int t_start=offsets[neighbor];
    //         int t_end=offsets[neighbor+1];
    //         atomicAdd(wedge_counts+node_id,(uint)(t_end-t_start));
    //     }
    // }

    //* v <- u -> w
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for(int i = tid; i < nodes; i+= gridDim.x * blockDim.x){
        unsigned int res = offsets[i+1] - offsets[i];
        res = res * (res - 1) / 2;
        // atomicAdd(wedge_counts, res);
        wedge_counts[i] = res;
    }
}

__global__ 
void TwoHopsKernel(int *adjs,int *offsets,int nodes, int *wedges, uint *wedge_offsets, uint *real_counts){
    //* traversal
    for(int node_id=blockIdx.x;node_id<nodes;node_id+=gridDim.x){
        //* ===== get 2-hop neighbors =====
        int warp_id=threadIdx.x/WARP_SIZE;
        int lane_id=threadIdx.x%WARP_SIZE;
        int warps=(blockDim.x/WARP_SIZE);
        int start=offsets[node_id];
        int end=offsets[node_id+1];
        int *node_wedges=wedges+wedge_offsets[node_id]; // offset of wedges
        
        // warp per 1-hop neighbor
        for(int i=start+warp_id;i<end;i+=warps){
            int neighbor=adjs[i];
            int t_start=offsets[neighbor];
            int t_end=offsets[neighbor+1];
            // record two-hop neighbor
            for(int j=t_start+lane_id;j<t_end;j+=WARP_SIZE){
                int two_hop_id=adjs[j];
                node_wedges[atomicAdd(real_counts+node_id, 1U)] = two_hop_id;
            }
        }
    }
} 

__global__
void AssignKernel(int nodes, pair_t *wedges, uint *wedge_offsets, uint *ray_info, uint *rays){
    
    for(int node_id=blockIdx.x;node_id<nodes;node_id+=gridDim.x){
        
        uint start=wedge_offsets[node_id];
        uint end=wedge_offsets[node_id+1];
        for(int i=start+threadIdx.x;i<end;i+=blockDim.x){
            if(wedges[i].w>=nodes) break;
            int pos=atomicAdd(rays,3);
            ray_info[pos]=node_id;
            ray_info[pos+1]=wedges[i].w;
            ray_info[pos+2]=wedges[i].count;
        }
    }
}

//* get the max number of two hops neighbors
// __global__
// void MaxTwoHopSizeKernel(int *adjs, int *offsets, int nodes, uint *max_two_hop_size){
//     int tid=threadIdx.x+blockIdx.x*blockDim.x;
//     int warp_id=tid/WARP_SIZE;
//     int lane_id=tid%WARP_SIZE;
//     int warps=(gridDim.x*blockDim.x)/WARP_SIZE;

//     for(int node_id=warp_id;node_id<nodes;node_id+=warps){
//         int start=offsets[node_id];
//         int end=offsets[node_id+1];
//         uint two_hop_size=0;
//         for(int i=start+lane_id;i<end;i+=WARP_SIZE){
//             int neighbor=adjs[i];
//             two_hop_size+=(offsets[neighbor+1]-offsets[neighbor]);
//         }
//         //* warp reduce
//         if(lane_id<32){
//             two_hop_size+=__shfl_down_sync(FULL_MASK,two_hop_size,16);
//             two_hop_size+=__shfl_down_sync(FULL_MASK,two_hop_size,8);
//             two_hop_size+=__shfl_down_sync(FULL_MASK,two_hop_size,4);
//             two_hop_size+=__shfl_down_sync(FULL_MASK,two_hop_size,2);
//             two_hop_size+=__shfl_down_sync(FULL_MASK,two_hop_size,1);
//         }
//         if(lane_id==0){
//             atomicMax(max_two_hop_size,two_hop_size);
//         }
//     }
// }

//* underutilize GPU resources
// __global__ void SortBlockData(int *two_hops, int *hop_counts, int nodes,uint stride) {
    
//     typedef cub::BlockRadixSort<int, BLOCK_SIZE, ELEMENTS_PER_THREAD> BlockRadixSort;
//     // Allocate shared memory for BlockRadixSort
//     __shared__ typename BlockRadixSort::TempStorage temp_storage;
//     // Per-thread tile data
//     int data[ELEMENTS_PER_THREAD];
//     // sort segment
//     for(int i=0;i<nodes;i++){
//         int* node_offset=two_hops+i*stride;
//         int blockOffset=blockIdx.x*blockDim.x*ELEMENTS_PER_THREAD;
        
//         for(int j=0;j<ELEMENTS_PER_THREAD;j++){
//             int idx=blockOffset+threadIdx.x*ELEMENTS_PER_THREAD+j;
//             if(idx<hop_counts[i]){
//                 data[j]=node_offset[idx];
//             }
//         }
//         __syncthreads();

//         BlockRadixSort(temp_storage).Sort(data);

//         for (int j=0;j<ELEMENTS_PER_THREAD;j++) {
//             int idx=blockOffset+threadIdx.x*ELEMENTS_PER_THREAD+j;
//             if(idx<hop_counts[i]){
//                 node_offset[idx]=data[j];
//             }
//         }
//     }
// }

void TwoHopsFunction(const int *adjs, const int *offsets, const graph_info_t &graph_info, std::vector<uint> &rt_ray_origin){
    double two_hops_time=0.0;

    int *d_adjs, *d_offsets;
    CUDA_CHECK(cudaMalloc((void**)&d_adjs,sizeof(int)*graph_info.edges));
    CUDA_CHECK(cudaMemcpy(d_adjs,adjs,sizeof(int)*graph_info.edges,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_offsets,sizeof(int)*(graph_info.nodes+1)));
    CUDA_CHECK(cudaMemcpy(d_offsets,offsets,sizeof(int)*(graph_info.nodes+1),cudaMemcpyHostToDevice));

    //* get number of 2-hop neighbors for each vertex

    uint *d_wedge_counts;
    CUDA_CHECK(cudaMalloc((void**)&d_wedge_counts,sizeof(uint)*(graph_info.nodes+1))); //+1
    CUDA_CHECK(cudaMemset(d_wedge_counts,0U,sizeof(uint)*(graph_info.nodes+1)));
    //TODO: count
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    // MaxTwoHopSizeKernel<<<blocks,block_size>>>(d_adjs, d_offsets, graph_info.nodes, d_max_two_hop_size);
    int block_size = BLOCK_SIZE;
    int blocks = (graph_info.nodes + block_size - 1) / block_size;
    CountKernel<<<blocks,block_size>>>(d_adjs, d_offsets, graph_info.nodes, d_wedge_counts);
    // total number of two hops neighbors
    size_t total_size=thrust::reduce(thrust::device,d_wedge_counts,d_wedge_counts+graph_info.nodes,(size_t)0); // = 5413528
    // prefiex sum
    thrust::exclusive_scan(thrust::device, d_wedge_counts, d_wedge_counts+(graph_info.nodes+1), d_wedge_counts);
    // CUDA_SYNC_CHECK();
    
    cpu_timer.StopTiming();
    two_hops_time+=cpu_timer.GetElapsedTime();

    uint *h_wedge_offsets = new uint[(graph_info.nodes+1)];
    CUDA_CHECK(cudaMemcpy(h_wedge_offsets, d_wedge_counts, sizeof(uint) * (graph_info.nodes+1), cudaMemcpyDeviceToHost));

    printf("total size = %llu, needed memory = %.2f MB\n",total_size,1.0*total_size*sizeof(int)/1024/1024);
    
    int *d_wedges;
    CUDA_CHECK(cudaMalloc((void**)&d_wedges,sizeof(int)*total_size));
    // thrust::fill(thrust::device,d_wedges,d_wedges+total_size,pair_t(graph_info.nodes,0U));
    uint *d_real_counts; // the number of unique (u, w) =  3413500
    CUDA_CHECK(cudaMalloc((void**)&d_real_counts,sizeof(uint)*graph_info.nodes);)
    CUDA_CHECK(cudaMemset(d_real_counts,0,sizeof(uint)*graph_info.nodes));

    //* traversal
    cpu_timer.StartTiming();
    TwoHopsKernel<<<blocks,block_size>>>(d_adjs, d_offsets, graph_info.nodes, d_wedges, d_wedge_counts,d_real_counts);
    CUDA_SYNC_CHECK();
    thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(d_wedges);
    uint rays_num = 0;
    for(int i=0;i<graph_info.nodes; i++){
        thrust::sort(d_ptr + h_wedge_offsets[i], d_ptr + h_wedge_offsets[i+1]);

        thrust::device_vector<int> d_keys(h_wedge_offsets[i + 1] - h_wedge_offsets[i]);
        thrust::device_vector<int> d_values(h_wedge_offsets[i + 1] - h_wedge_offsets[i]);
        auto new_end = thrust::reduce_by_key(d_ptr + h_wedge_offsets[i], d_ptr + h_wedge_offsets[i+1], 
            thrust::constant_iterator<int>(1), d_keys.begin(), d_values.begin()
        );
        int num_unique_elements = new_end.first - d_keys.begin();
        rays_num += num_unique_elements;


    }
    // = 3413500
    cpu_timer.StopTiming();
    two_hops_time+=cpu_timer.GetElapsedTime();
    CUDA_CHECK(cudaFree(d_real_counts));
    printf("rays_num = %u\n",rays_num);
    
    rt_ray_origin.resize(rays_num*3);
    uint *d_rt_ray_origin;
    CUDA_CHECK(cudaMalloc((void**)&d_rt_ray_origin,sizeof(rays_num*3)));
    uint *d_rays_num;
    CUDA_CHECK(cudaMalloc((void**)&d_rays_num,sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_rays_num,0U,sizeof(uint)));
    //* transfer
    cpu_timer.StartTiming();
    // AssignKernel<<<blocks,block_size>>>(graph_info.nodes, d_wedges, d_wedge_counts, d_rt_ray_origin, d_rays_num);
    CUDA_SYNC_CHECK();
    cpu_timer.StopTiming();
    two_hops_time+=cpu_timer.GetElapsedTime();

    uint rays;
    CUDA_CHECK(cudaMemcpy(&rays,d_rays_num,sizeof(uint),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rays_num));
    printf("rays = %u\n",rays);
    assert(rays==3*rays_num);
    
    CUDA_CHECK(cudaMemcpy(rt_ray_origin.data(),d_rt_ray_origin,sizeof(uint)*rays,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rt_ray_origin));

    CUDA_CHECK(cudaFree(d_wedges));
    CUDA_CHECK(cudaFree(d_wedge_counts));
    CUDA_CHECK(cudaFree(d_adjs));
    CUDA_CHECK(cudaFree(d_offsets));

    printf("Device: tow hops time = %.2f ms\n",two_hops_time);
}

#endif

void ConvertOnGPU(const int *adjs,const int *offsets,const graph_info_t &graph_info,float3 axis_offset,
    float triangle_eps,float3 *vertices,uint num_of_vertices){
    
    int *d_adjs, *d_offsets;
    float3 *d_vertices;
    uint *d_vertex_count;
    CUDA_CHECK(cudaMalloc((void**)&d_adjs,sizeof(int)*graph_info.edges));
    CUDA_CHECK(cudaMemcpy(d_adjs,adjs,sizeof(int)*graph_info.edges,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_offsets,sizeof(int)*(graph_info.nodes+1)));
    CUDA_CHECK(cudaMemcpy(d_offsets,offsets,sizeof(int)*(graph_info.nodes+1),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_vertices,sizeof(float3)*num_of_vertices));
    CUDA_CHECK(cudaMalloc((void**)&d_vertex_count,sizeof(uint)));
    CUDA_CHECK(cudaMemset(d_vertex_count,0,sizeof(uint)));

#if RTTC_METHOD == 0
    int block_size=BLOCK_SIZE;
    int blocks=(graph_info.nodes<=MAX_BLOCKS?graph_info.nodes:MAX_BLOCKS);
    ListIntersectionVerticesKernel<<<blocks,block_size>>>(
        d_adjs,d_offsets,graph_info.nodes,triangle_eps,d_vertices,d_vertex_count,axis_offset
    );
    CUDA_SYNC_CHECK();
#elif RTTC_METHOD == 1
    int block_size=BLOCK_SIZE;
    int blocks=(graph_info.nodes<=MAX_BLOCKS?graph_info.nodes:MAX_BLOCKS);
    HashmapVerticesKernel<<<blocks,block_size>>>(
        d_adjs,d_offsets,graph_info.nodes,triangle_eps,d_vertices,d_vertex_count,axis_offset
    );
    CUDA_SYNC_CHECK();

#else
    printf("ConvertOnGPU: method error\n");
#endif

    CUDA_CHECK(cudaMemcpy(vertices,d_vertices,sizeof(float3)*num_of_vertices,cudaMemcpyDeviceToHost));
    uint vertex_count=0;
    CUDA_CHECK(cudaMemcpy(&vertex_count,d_vertex_count,sizeof(uint),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_vertex_count));
    CUDA_CHECK(cudaFree(d_vertices));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_adjs));
    assert(vertex_count==num_of_vertices);
}

#ifdef LIST_OPTIMIZATION
__global__ void ExpandKernel(float3 *centers, float3 *triangle_vertices, uint num, float triangle_eps, float3 axis_offset){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < num){
        float3 center = make_float3(centers[tid].x - axis_offset.x, centers[tid].y - axis_offset.y, centers[tid].z - axis_offset.z);
        CenterToTriangle(center, triangle_eps, triangle_vertices, tid*3);
    }
}

void ExpandonGPU(float3 *centers, float3 *triangle_vertices, uint num, float triangle_eps, float3 axis_offset){
    float3* d_centers;
    CUDA_CHECK(cudaMalloc((void**)&d_centers,sizeof(float3)*num));
    CUDA_CHECK(cudaMemcpy(d_centers,centers,sizeof(float3)*num,cudaMemcpyHostToDevice));
    float3* d_triangle_vertices;
    CUDA_CHECK(cudaMalloc((void**)&d_triangle_vertices,sizeof(float3)*num*3));

    int block_size = 256;
    int blocks = (num + block_size - 1) / block_size;
    ExpandKernel<<<blocks, block_size>>>(d_centers, d_triangle_vertices, num, triangle_eps, axis_offset);

    CUDA_CHECK(cudaMemcpy(triangle_vertices,d_triangle_vertices,sizeof(float3)*num*3, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_triangle_vertices));
    CUDA_CHECK(cudaFree(d_centers));
}

void ExpandonCPU(float3 *centers, float3 *triangle_vertices, uint num, float triangle_eps, float3 axis_offset){
#pragma omp parallel for
    for(int i=0; i < num; i ++){
        float3 center = make_float3(centers[i].x - axis_offset.x, centers[i].y - axis_offset.y, centers[i].z - axis_offset.z);
        float below_y=center.y-triangle_eps;
        float above_y=center.y+triangle_eps;
        float below_z=center.z-triangle_eps;
        float above_z=center.z+triangle_eps;
        uint pos = i*3;
        triangle_vertices[pos]=make_float3(center.x,center.y,above_z);
        triangle_vertices[pos+1]=make_float3(center.x,below_y,below_z);
        triangle_vertices[pos+2]=make_float3(center.x,above_y,below_z);
    }
}
#endif