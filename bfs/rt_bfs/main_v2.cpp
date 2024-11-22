#include "graph.h"
#include "rt_bfs_v2.h"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    //============================================
    if (argc != 5) {
        std::cerr << "Usage: "
                  << "./rt_bfs_v2 graph_path chunk_length digit device_id" << std::endl;
        exit(1);
    }
    std::string graph_path = argv[1];       //* file path
    std::string str_chunk_length = argv[2]; //* chunk length
    std::string str_digit = argv[3];        //* chunk length
    std::string str_device_id = argv[4];
    int chunk_length = std::stoi(str_chunk_length);
    int digit = std::stof(str_digit);
    int device_id = std::stoi(str_device_id);

    Graph graph;
    graph.ReadMtx(graph_path);

    RTBFS_V2 rt_bfs_v2(graph, chunk_length, digit);

    rt_bfs_v2.SetDevice(device_id);
    rt_bfs_v2.OptiXSetup();
    rt_bfs_v2.BuildAccel();
    rt_bfs_v2.Traversal(); // 2759251
    rt_bfs_v2.PrintResult();
    rt_bfs_v2.CheckResult();
    return 0;
}