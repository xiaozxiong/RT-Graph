#include "graph.h"
#include "thrust_helper.h"
#include "timer.h"
#include "common.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>
#include <thread>

#define USE_THRUST

Graph::Graph(int device_id) { CUDA_CHECK(cudaSetDevice(device_id)); }

Graph::Graph() {}

Graph::~Graph() {}

void Graph::ReadBinaryEdges(std::string &input_file_path) {
    std::ifstream file(input_file_path, std::ios::in | std::ios::binary);
    file.seekg(0, std::ios::end);
    std::streampos end = file.tellg();
    size_t file_size = end; // Bytes
    file.close();
    std::cout << "The size of graph file: " << file_size << " bytes\n";
    graph_info_.edges = file_size / (sizeof(int) * 2); // fromId -> toId
    edge_list_.resize(graph_info_.edges);
    std::ifstream read_file(input_file_path, std::ios::binary);
    if (read_file.bad()) {
        std::cerr << "Error: file not found" << std::endl;
        exit(-1);
    }
    read_file.read((char *)edge_list_.data(), sizeof(edge_t) * graph_info_.edges);
    read_file.close();

    int max_id = 0, start_index = 1;
    for (auto &e : edge_list_) {
        if ((e.src == 0 || e.dst == 0) && start_index == 1)
            start_index = 0;
        max_id = std::max(max_id, std::max(e.src, e.dst));
    }
    graph_info_.nodes = (start_index == 0 ? max_id + 1 : max_id);
    int *befores = (int *)malloc(sizeof(int) * (graph_info_.nodes + start_index));
    memset(befores, sizeof(int) * graph_info_.nodes, 0);
#pragma omp parallel for
    for (auto &e : edge_list_) {
        befores[e.src] = 1;
        befores[e.dst] = 1;
    }
    for (int i = start_index + 1; i < start_index + graph_info_.nodes; i += 1) {
        befores[i] += befores[i - 1];
    }
#pragma omp parallel for
    for (auto &e : edge_list_) {
        e.src = befores[e.src] - 1;
        e.dst = befores[e.dst] - 1;
    }
    graph_info_.nodes = befores[graph_info_.nodes + start_index - 1];
    delete[] befores;
#ifdef DEBUG
    std::cout << "Index starts from " << start_index << ", max id = " << max_id << "\n";
#endif
    std::cout << "Graph Info: nodes = " << graph_info_.nodes << ", edges = " << graph_info_.edges
              << std::endl;
}

void Graph::ConvertToCSR() {
    //* The edge_list has been sorted and needs to remove illegal edges.
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();

    adjs_.resize(graph_info_.edges - useless_edges_); // zeros
    offsets_.resize(graph_info_.nodes + 1);           // zeros
    int null_edges = 0;
    for (int i = 0; i < graph_info_.edges; i += 1) {
        edge_t e = edge_list_[i];
        if (e.src == -1 || e.dst == -1) {
            null_edges += 1;
            continue;
        }
        offsets_[e.src] += 1;
        adjs_[i - null_edges] = e.dst;
    }
    assert(null_edges == useless_edges_);
    graph_info_.edges -= useless_edges_;

    int last = offsets_[0];
    offsets_[0] = 0;
    for (int i = 1; i <= graph_info_.nodes; i += 1) {
        int temp = offsets_[i];
        offsets_[i] = offsets_[i - 1] + last;
        last = temp;
    }
    assert(offsets_[graph_info_.nodes] == graph_info_.edges);

    cpu_timer.StopTiming();
    std::cout << "CSR converting time = " << cpu_timer.GetElapsedTime() << " ms\n";
}

void Graph::TCPreprocessing() {
    CPUTimer cpu_timer;
    cpu_timer.StartTiming();
    
    //* comopute degree
    int *degree = (int *)malloc(sizeof(int) * graph_info_.nodes);
    memset(degree, 0, sizeof(int) * graph_info_.nodes);
    for (auto &e : edge_list_) {
        degree[e.src] += 1;
        degree[e.dst] += 1;
    }

#pragma omp parallel for
    for (auto &e : edge_list_) {
        int src = e.src;
        int dst = e.dst;
        if (degree[src] > degree[dst] || (degree[src] == degree[dst] && src > dst)) {
            e.src = dst;
            e.dst = src;
        }
    }

    int *removes = (int *)malloc(sizeof(int) * graph_info_.nodes);
    int removed_edges = 0;
    removes[0] = static_cast<int>(degree[0] <= 1); // index starts from 0
    for (int i = 1; i < graph_info_.nodes; i += 1) {
        removes[i] = removes[i - 1] + static_cast<int>(degree[i] <= 1);
    }
#pragma omp parallel for reduction(+ : removed_edges)
    for (auto &e : edge_list_) {
        int src = e.src;
        int dst = e.dst;
        if (degree[e.src] <= 1)
            e.src = -1;
        if (degree[e.dst] <= 1)
            e.dst = -1;
        if (e.src == -1 || e.dst == -1) {
            removed_edges += 1;
            continue;
        }
        e.src = (src - removes[src]); // removes[src]<=src
        e.dst = (dst - removes[dst]);
    }
    // std::cout<<"removed edges = "<<removed_edges<<std::endl;
    graph_info_.nodes -= removes[graph_info_.nodes - 1];
    graph_info_.edges -= removed_edges;
    
    free(removes);
    free(degree);

    //* sort edges
    //* There are useless edges in edge_list, the actual size should be graph_info_.edges.
    //* And there are some duplicated edges in edge_list.
    ThrustSortEdgesInBatch(edge_list_, graph_info_.nodes);
    
    //* mark duplicated edges and self-loop
    int last_src = -1, last_dst = -1;
    useless_edges_ = 0;
    for (int i = 0; i < graph_info_.edges; i += 1) {
        int src = edge_list_[i].src;
        int dst = edge_list_[i].dst;
        assert(src != -1 && dst != -1);
        if (src == dst || (src == last_src && dst == last_dst)) {
            edge_list_[i].src = -1;
            edge_list_[i].dst = -1;
            useless_edges_ += 1;
        }
        last_src = src;
        last_dst = dst;
    }

    cpu_timer.StopTiming();
    std::cout << "TC Preprocessing Time = " << cpu_timer.GetElapsedTime() << " ms\n";
}

void Graph::Txt2Bin(const std::string &txt_file, const std::string &bin_file){
    std::ifstream infile(txt_file);
    if (!infile) {
        std::cerr << "Error opening file: " << txt_file << std::endl;
        exit(1);
    }

    std::ofstream outfile(bin_file, std::ios::binary | std::ios::out);
    if (!outfile) {
        std::cerr << "Error opening file: " << bin_file << std::endl;
        exit(1);
    }

    int node1, node2;
    std::string line;

    // Read the text file line by line
    while (std::getline(infile, line)) {
        std::stringstream ss(line);
        
        // Read the two integers (nodes) from the current line
        if (ss >> node1 >> node2) {
            // Write the two integers to the binary file (as binary)
            outfile.write(reinterpret_cast<const char*>(&node1), sizeof(int));
            outfile.write(reinterpret_cast<const char*>(&node2), sizeof(int));
        } else {
            std::cerr << "Error parsing line: " << line << std::endl;
        }
    }

    // Close the files
    infile.close();
    outfile.close();

    std::cout << "Edge list has been successfully converted to binary." << std::endl;
}