#include <stdio.h>

#include "bs_tc.h"
#include "graph.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./bs_rt graph device_id\n");
        exit(1);
    }
    std::string graph_file = argv[1];
    int device_id = atoi(argv[2]);
    Graph graph;
    graph.ReadBinaryEdges(graph_file);
    graph.TCPreprocessing();
    graph.ConvertToCSR();

    CountOnGPU(graph.adjs_.data(), graph.offsets_.data(), graph.graph_info_.nodes,
               graph.graph_info_.edges, device_id);
    return 0;
}