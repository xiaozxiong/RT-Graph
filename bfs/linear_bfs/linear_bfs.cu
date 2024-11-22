#include "common.h"
#include "linear_bfs.cuh"
#include "timer.h"

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <queue>

#define FULL_MASK 0xFFFFFFFF

void cpu_bfs(int nodes, const int *adjs, const int *offsets, int *levels, int source_node = 0) {
    for (int i = 0; i < nodes; i += 1)
        levels[i] = -1;
    std::queue<int> que;
    que.push(source_node);
    levels[source_node] = 0;
    while (!que.empty()) {
        int cur_node = que.front();
        que.pop();
        int start = offsets[cur_node];
        int end = offsets[cur_node + 1];

        for (int i = start; i < end; i += 1) {
            int adj_node = adjs[i];
            if (levels[adj_node] == -1) {
                levels[adj_node] = levels[cur_node] + 1;
                que.push(adj_node);
                // if (adj_node == 968)
                //     printf("%d -> %d\n", cur_node, adj_node);
            }
        }
    }
}

void Check(const int *adjs, const int *offsets, int nodes, int *gpu_levels) {
    int *cpu_levels = (int *)malloc(sizeof(int) * nodes);
    cpu_bfs(nodes, adjs, offsets, cpu_levels);
    int errors = 0;
    for (int i = 0; i < nodes; i += 1) {
        errors += (cpu_levels[i] != gpu_levels[i]);
    }

    // int min_level = nodes;
    // int node = -1;
    // for (int i = 0; i < nodes; i++) {
    //     if (cpu_levels[i] != gpu_levels[i]) {
    //         if (min_level > cpu_levels[i] && cpu_levels[i] != -1) {
    //             min_level = cpu_levels[i];
    //             node = i;
    //         }
    //         // printf("node %d: cpu = %d, gpu = %d\n", i, cpu_levels[i], gpu_levels[i]);
    //     }
    // }
    // printf("node 969: cpu level = %d, gpu level = %d\n", cpu_levels[969], gpu_levels[969]);
    // printf("node %d: cpu = %d, gpu = %d\n", node, cpu_levels[node], gpu_levels[node]);

    free(cpu_levels);
    printf("#Number of errors: %d\n", errors);
}

// (d_new_queue,queue_size,d_old_queue,d_old_queue_size,d_adjs,d_new_offsets,d_levels,level)
__global__ void ExpandOneLevel(int *queue, int queue_size, int *next_queue, int *next_queue_size,
                               const int *adjs, const int *offsets, int *levels,
                               int current_level) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= queue_size)
        return;
    // if(current_level == 5) return;
    // if(tid == 0 && current_level <= 5){
    // 	for(int i = 94904; i < 94977; i++){
    // 		printf("#969: %d- %d\n", i, adjs[i]);
    // 	}
    // }

    int current_id = queue[tid]; // new node id

    int start = offsets[current_id];
    int end = offsets[current_id + 1];

    // if(current_id == 31973){
    // 	printf("%d: [%d, %d), %d\n", current_id, start, end, (int)(adjs[94909] == 968));
    // 	for(int k = start; k < end; k ++){
    // 		printf("%d: ---> %d\n", k, adjs[k]);
    // 	}
    // }

    for (int i = start; i < end; i += 1) {
        int v = adjs[i];
        if (levels[v] == -1) {
            levels[v] = current_level;
            int pos = atomicAdd(next_queue_size, 1); // next_queue_size has limit
            next_queue[pos] = v;
        }
    }
}
// TODO: allocate a thread to each node in queue
__global__ void NodeMappingThread(int *queue, int queue_size, int *node_offsets, int *new_queue,
                                  int *new_queue_size, int level) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = tid; i < queue_size; i += gridDim.x * blockDim.x) {
        int node = queue[i]; // old node
                             //* new node
        int start = node_offsets[node];
        int end = node_offsets[node + 1];

        int pos = atomicAdd(new_queue_size, end - start);
        for (int j = start; j < end; j += 1) {
            new_queue[pos + j - start] = j;
        }
    }
}
// TODO: allocate a warp to each node in queue
__global__ void NodeMappingWarp(int *queue, int queue_size, int *node_offsets, int *new_queue,
                                int *new_queue_size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int warp_id = (tid >> 5);    // warp id
    int lane_id = (tid & 31);    // id in a warp
    int warps = 4 * gridDim.x;   // number of warps
    __shared__ int positions[4]; // 4 warps in a block
    for (int i = warp_id; i < queue_size; i += warps) {
        int node = queue[i];
        int start = node_offsets[node];
        int end = node_offsets[node + 1];
        int len = end - start;
        if (lane_id == 0)
            positions[warp_id & 3] = atomicAdd(new_queue_size, len);
        __syncwarp(FULL_MASK);
        for (int j = lane_id; j < len; j += 32) {
            new_queue[positions[warp_id & 3] + j] = start + j;
        }
    }
}

// TODO: remove duplicatvie nodes and map node to new nodes
double FilterandMap(int *d_queue, int old_size, int *d_node_offsets, int *d_new_queue,
                    int *d_new_size, int level, bool f = true) {

    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    int filtered_nodes;
    //* filter
    if (f) {
        thrust::sort(thrust::device, d_queue, d_queue + old_size);
        filtered_nodes = thrust::unique(thrust::device, d_queue, d_queue + old_size) - d_queue;
    } else {
        //* no filter for road_usa
        filtered_nodes = old_size;
    }

    int block_size = 128;
    int blocks = (filtered_nodes + block_size - 1) / block_size;
    NodeMappingThread<<<blocks, block_size>>>(d_queue, filtered_nodes, d_node_offsets, d_new_queue,
                                              d_new_size, level);
    // NodeMappingWarp<<<blocks,block_size>>>(d_queue,filtered_nodes,d_node_offsets,d_new_queue,d_new_size);
    CUDA_SYNC_CHECK();
    // cudaDeviceSynchronize();

    // int new_size = 0;
    // CUDA_CHECK(cudaMemcpy(&new_size, d_new_size, sizeof(int), cudaMemcpyDeviceToHost));
    // printf("new queue size after mapping: %d\n", new_size);

    cpu_timer.StopTiming();
    return cpu_timer.GetElapsedTime();
}
// TODO: devide adjacency list to multiple chunks
double Preprocessing(const int *offsets, int nodes, int edges, int chunk_length, int &new_nodes,
                     int *&node_offsets, int *&new_offsets) {
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    //* original node -> new node
    node_offsets = (int *)malloc(sizeof(int) * (nodes + 1));
    new_nodes = 0;
    for (int i = 0; i < nodes; i += 1) {
        node_offsets[i] = new_nodes;
        int adjs_len = offsets[i + 1] - offsets[i];
        int chunk_num = (adjs_len + chunk_length - 1) / chunk_length;
        new_nodes += chunk_num;
    }
    node_offsets[nodes] = new_nodes;

    // int debug_node = 969;
    // printf("%d: offsets = [%d, %d)\n", debug_node, offsets[debug_node], offsets[debug_node + 1]);
    // printf("%d: new node_offsets = [%d, %d)\n", debug_node, node_offsets[debug_node],
    // node_offsets[debug_node + 1]);

    //* get the new offese for new nodes
    new_offsets = (int *)malloc(sizeof(int) * (new_nodes + 1)); // offset of each new node
    for (int i = 0; i < nodes; i += 1) {
        int start = offsets[i];
        int end = offsets[i + 1];
        int adjs_len = end - start;
        int chunk_num = (adjs_len + chunk_length - 1) / chunk_length;

        int offset = start; // = old offset
        for (int j = 0; j < chunk_num; j += 1) {
            new_offsets[node_offsets[i] + j] = offset;
            offset += chunk_length; // last chunk
        }
    }
    new_offsets[new_nodes] = offsets[nodes];
    cpu_timer.StopTiming();
    return cpu_timer.GetElapsedTime();
}

__global__ void Testkernel(const int *adjs, int start, int end) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= end - start)
        return;
    printf("#test > %d: %d\n", tid + start, adjs[start + tid]);
}

double LinearBFS(const int *adjs, const int *offsets, int nodes, int edges, int chunk_length,
                 int source_node, bool filter) {
    //* preprocessing
    int *node_offsets, *new_offsets;
    int new_nodes = 0;
    double preprocessing_time =
        Preprocessing(offsets, nodes, edges, chunk_length, new_nodes, node_offsets, new_offsets);
    // printf("old nodes = %d, old edges = %d, new nodes = %d, new edges = %d\n", nodes, edges,
    //        new_nodes, new_offsets[new_nodes]);

    // int debug_node = 969;
    // printf("%d -> adj list:", debug_node);
    // for(int i = offsets[debug_node]; i < offsets[debug_node + 1]; i++){
    // 	printf(" %d", adjs[i]);
    // }
    // printf("\n");
    // for(int i = 94907; i < 94910; i++){
    // 	printf("i = %d: adj = %d\n", i, adjs[i]);
    // }
    if (filter) {
        printf("-- Filter is used --\n");
    } else {
        printf("-- Filter isn't used --\n");
    }

    //* new nodes
    int *d_node_offsets, *d_new_offsets;
    CUDA_CHECK(
        cudaMalloc((void **)&d_node_offsets, sizeof(int) * (nodes + 1))); // old node -> new node
    CUDA_CHECK(cudaMemcpy(d_node_offsets, node_offsets, sizeof(int) * (nodes + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc((void **)&d_new_offsets, sizeof(int) * (new_nodes + 1))); // new node -> adjs
    CUDA_CHECK(cudaMemcpy(d_new_offsets, new_offsets, sizeof(int) * (new_nodes + 1),
                          cudaMemcpyHostToDevice));
    //* graph
    int *d_adjs; //,*d_offsets;
    CUDA_CHECK(cudaMalloc((void **)&d_adjs, sizeof(int) * edges));
    CUDA_CHECK(cudaMemcpy(d_adjs, adjs, sizeof(int) * edges, cudaMemcpyHostToDevice));

    //* levels
    int *levels = (int *)malloc(sizeof(int) * nodes);
    for (int i = 0; i < nodes; i += 1)
        levels[i] = -1;
    levels[source_node] = 0;
    //* level:
    int *d_levels;
    CUDA_CHECK(cudaMalloc((void **)&d_levels, sizeof(int) * nodes));
    CUDA_CHECK(cudaMemcpy(d_levels, levels, sizeof(int) * nodes, cudaMemcpyHostToDevice));
    //* queue
    //! if filter is not used, the size of both queue would be not enough
    int *d_old_queue, *d_new_queue, *d_old_queue_size, *d_new_queue_size;

    int new_queue_mem_size = new_nodes + 1;
    int old_queue_mem_size = nodes + 1;
    if (!filter) { // prevent out-of-bounds access
        new_queue_mem_size = edges * 2;
        old_queue_mem_size = edges * 2;
    }
    // store new nodes after mapping:
    CUDA_CHECK(cudaMalloc((void **)&d_new_queue,
                          sizeof(int) * new_queue_mem_size)); //* new nodes, might be not enough
    CUDA_CHECK(cudaMalloc((void **)&d_new_queue_size, sizeof(int)));
    // get the result after expanding
    CUDA_CHECK(cudaMalloc(
        (void **)&d_old_queue,
        sizeof(int) * old_queue_mem_size)); //* old nodes, if no filter, this might be not enough
    CUDA_CHECK(cudaMalloc((void **)&d_old_queue_size, sizeof(int)));

    //* initial queue
    thrust::fill(thrust::device, d_old_queue, d_old_queue + 1, source_node);
    thrust::fill(thrust::device, d_old_queue_size, d_old_queue_size + 1, 1);

    int queue_size = 0;
    int level = 0;
    double filer_map_time = 0.0, expand_time = 0.0;
    while (true) { // new node queue -> old node queue
        CUDA_CHECK(cudaMemcpy(&queue_size, d_old_queue_size, sizeof(int), cudaMemcpyDeviceToHost));
        if (queue_size == 0) {
            break;
        }
        // printf("level %d -> %d: original queue size = %d\n", level, level + 1, queue_size);
        // printf("test kernel before filter\n");
        // Testkernel<<<1, 32>>>(d_adjs, 94907, 94910);
        // cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemset(d_new_queue_size, 0, sizeof(int)));
        filer_map_time += FilterandMap(d_old_queue, queue_size, d_node_offsets, d_new_queue,
                                       d_new_queue_size, level, filter);

        CUDA_CHECK(cudaMemcpy(&queue_size, d_new_queue_size, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemset(d_old_queue_size, 0, sizeof(int)));

        // printf("test kernel before expand\n");
        // Testkernel<<<1, 32>>>(d_adjs, 94907, 94910);
        // cudaDeviceSynchronize();

        //* expand
        level += 1;
        int block_size = 256;
        int blocks = (queue_size + block_size - 1) / block_size;

        GPUTimer gpu_timer;
        gpu_timer.StartTiming();
        ExpandOneLevel<<<blocks, block_size>>>(d_new_queue, queue_size, d_old_queue,
                                               d_old_queue_size, d_adjs, d_new_offsets, d_levels,
                                               level); // d_old_queue is result
        CUDA_SYNC_CHECK();
        gpu_timer.StopTiming();

        expand_time += gpu_timer.GetElapsedTime();

        // if(level == 5) break;
    }
    //* copy back the result
    printf("Max level = %d\n", level);
    CUDA_CHECK(cudaMemcpy(levels, d_levels, sizeof(int) * nodes, cudaMemcpyDeviceToHost));

    // Testkernel<<<1, 32>>>(d_adjs, 94907, 94910);
    // cudaDeviceSynchronize();

    // int * temp_adj = new int[edges];
    // CUDA_CHECK(cudaMemcpy(temp_adj, d_adjs, sizeof(int) * edges, cudaMemcpyDeviceToHost));
    // printf("adj check: temp_adj[%d] = %d\n",94909, temp_adj[94909]);
    // for(int i = 0; i < edges; i++){
    // 	if(temp_adj[i] != adjs[i]){
    // 		printf("adj check: %d != %d\n", temp_adj[i], adjs[i]);
    // 		break;
    // 	}
    // }

    CUDA_CHECK(cudaFree(d_old_queue));
    CUDA_CHECK(cudaFree(d_new_queue));
    CUDA_CHECK(cudaFree(d_old_queue_size));
    CUDA_CHECK(cudaFree(d_new_queue_size));
    CUDA_CHECK(cudaFree(d_levels));
    CUDA_CHECK(cudaFree(d_adjs));
    CUDA_CHECK(cudaFree(d_new_offsets));
    CUDA_CHECK(cudaFree(d_node_offsets));
    free(node_offsets);
    free(new_offsets);
    // TODO: compare with cpu bfs
    Check(adjs, offsets, nodes, levels);
    free(levels);

    printf("Linear BFS: preprocessing_time = %f ms, filer_map_time = %f ms, expand_time = %f ms\n",
           preprocessing_time, filer_map_time, expand_time);

    return filer_map_time + expand_time;
}
