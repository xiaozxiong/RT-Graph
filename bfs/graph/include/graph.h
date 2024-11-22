#pragma once

#include "common.h"
#include "../mmio/mmio.h"

#include <string>
#include <vector>

// template<typename T,typename W>
struct CoordinateMtx{
public:
    int num_rows,num_columns,num_nonzeros;
    std::vector<int> row_indices;
    std::vector<int> column_indices;
    std::vector<double> nonzero_values;

    CoordinateMtx(int _num_rows,int _num_columns,int _num_nonzeros):
        num_rows(_num_rows),num_columns(_num_columns),num_nonzeros(_num_nonzeros){
        
        row_indices.resize(num_nonzeros);
        column_indices.resize(num_nonzeros);
        nonzero_values.resize(num_nonzeros);
    }
};

class Graph{

public:
    Graph();
    ~Graph();
    void ReadMtx(const std::string& file_path);

    graph_info_t GetGraphInfo();
    int *GetOffsets();
    int *GetAdjs();
    double *GetEdgeWeights();

private:
    //* mtx format
    MM_typecode code_;
    //* graph info
    int node_num_{0},edge_num_{0};
    int min_degree_{0},max_degree_{0};
    double avg_degree_{0.0};

    std::vector<int> offsets_;
    std::vector<int> adjs_;
    std::vector<double> weights_;
    
private:
    void ConvertCoordinateToCSR(const CoordinateMtx& coo);
    void ComputeGraphInfo();
};
