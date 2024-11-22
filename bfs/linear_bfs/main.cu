#include <stdio.h>
#include <string>

#include "common.h"
#include "graph.h"
#include "linear_bfs.cuh"

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: ./linear_bfs graph_path chunk_length device_id\n");
        exit(1);
    }
    std::string file_path = argv[1];
    int chunk_length = atoi(argv[2]);
    int device_id = atoi(argv[3]);
    int source_node = 0;
    bool filter = false;

    Graph graph;
    graph.ReadMtx(file_path);

    graph_info_t graph_info = graph.GetGraphInfo();
    int *adjs = graph.GetAdjs();
    int *offsets = graph.GetOffsets();
    cudaSetDevice(device_id);
    double traversal_time = LinearBFS(adjs, offsets, graph_info.node_num, graph_info.edge_num,
                                      chunk_length, source_node, filter);
    printf("Linear BFS: traversal time = %f ms\n", traversal_time);
    return 0;
}