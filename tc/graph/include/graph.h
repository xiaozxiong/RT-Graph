#pragma once

#include "model.h"

#include <string>
#include <vector>

class Graph {
public:
    Graph(int device_id);
    Graph();
    ~Graph();
    void ReadBinaryEdges(std::string &input_file_path);
    
    void ConvertToCSR();

    void TCPreprocessing();
    
    //* edge list: text file to binary file
    static void Txt2Bin(const std::string &txt_file, const std::string &bin_file);
    
    //
    std::vector<int> adjs_;
    std::vector<int> offsets_;
    graph_info_t graph_info_;

private:
    std::vector<edge_t> edge_list_;
    int useless_edges_{0};
};