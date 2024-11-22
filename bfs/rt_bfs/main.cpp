#include "graph.h"
#include "rt_bfs.h"

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {
    //============================================
    if (argc != 4) {
        std::cerr << "Usage: "
                  << "./rt_bfs graph_path chunk_length device_id" << std::endl;
        exit(1);
    }
    std::string graph_path = argv[1];       //* file path
    std::string str_chunk_length = argv[2]; //* chunk length
    std::string str_device_id = argv[3];
    int chunk_length = std::stoi(str_chunk_length);
    int device_id = std::stoi(str_device_id);

    Graph graph;
    graph.ReadMtx(graph_path);
    RTBFS rt_bfs(graph, chunk_length);
    rt_bfs.SetDevice(device_id);
    rt_bfs.OptiXSetup();
    rt_bfs.BuildAccel();
    rt_bfs.Traversal();
    rt_bfs.PrintResult();
    rt_bfs.CheckResult();
    return 0;
}