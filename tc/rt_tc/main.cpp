#include "graph.h"
#include "model.h"
#include "rt_tc.h"

#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./rt_tc graph_file device_id\n");
        exit(1);
    }
    std::string graph_file = argv[1];
    int device_id = atoi(argv[2]);

    Graph graph(device_id);
    graph.ReadBinaryEdges(graph_file);
    graph.TCPreprocessing();
    graph.ConvertToCSR();
    printf("After preprocessing: nodes = %u, edges = %u\n", graph.graph_info_.nodes,
           graph.graph_info_.edges);

    RTTC tc = RTTC(device_id);
    tc.BuildBVHAndComputeRay(graph.adjs_.data(), graph.offsets_.data(), graph.graph_info_);
    tc.CountTriangles();

    std::cout << "==========>>> Result <<<==========\n";
    std::cout << "GAS Memory Size = " << 1.0 * tc.bvh_memory_size_ / 1024 / 1024 / 1024 << " GB\n";
    std::cout << "BVH Building Time = " << tc.bvh_building_time_ << " ms\n";
    std::cout << "Counting Time = " << tc.counting_time_ << " ms\n";
    std::cout << "Triangle Counting = " << tc.total_triangles_ << std::endl;
    return 0;
}
