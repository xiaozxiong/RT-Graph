#include "include/graph.h"
#include "mmio/mmio.h"
#include <iostream>


Graph::Graph(){

}

Graph::~Graph(){

}

void Graph::ReadMtx(const std::string& file_path){
    FILE* file;
    if((file=fopen(file_path.c_str(),"r"))==NULL){
        std::cerr << "File could not be opened: " << file_path << std::endl;
        exit(1);
    }
    if(mm_read_banner(file, &code_)!=0){
        std::cerr << "Could not process Matrix Market banner" << std::endl;
        exit(1);
    }
    // Make sure we're actually reading a matrix, and not an array.
    if(mm_is_array(code_)){
        std::cerr << "File is not a sparse matrix" << std::endl;
        exit(1);
    }
    int num_rows,num_columns,num_nonzeros;
    if((mm_read_mtx_crd_size(file,&num_rows,&num_columns,&num_nonzeros))!=0){
        std::cerr << "Could not read file info (M, N, NNZ)" << std::endl;
        exit(1);
    }
    std::cout<<"Mtx Reading Check: ("<<num_rows<<", "<<num_columns<<", "<<num_nonzeros<<")"<<std::endl;

    CoordinateMtx coo(num_rows,num_columns,num_nonzeros);
    if(mm_is_pattern(code_)){
        for(int i=0;i<num_nonzeros;i+=1){
            int row_index{0},col_index{0};
            auto num_assigned=fscanf(file, " %d %d \n", &row_index, &col_index);
            if(num_assigned != 2){
                std::cerr<<"Could not read edge from market file"<<std::endl;
                exit(1);
            }
            if(row_index==0||col_index==0){
                std::cerr<<"Market file is zero-indexed"<<std::endl;
                exit(1);
            }
            // 
            coo.row_indices[i]=row_index-1;
            coo.column_indices[i]=col_index-1;
            coo.nonzero_values[i]=1.0;//* weight
        }
    }
    else if(mm_is_real(code_)||mm_is_integer(code_)){
        for (int i=0;i<num_nonzeros;i+=1){
            int row_index{0},col_index{0};
            double weight{0.0};
            auto num_assigned=fscanf(file, " %d %d %lf \n", &row_index, &col_index, &weight);
            if(num_assigned!=3){
                std::cerr<<"Could not read weighted edge from market file"<<std::endl;
                exit(1);
            }
            if(row_index==0){
                std::cerr<<"Market file is zero-indexed"<<std::endl;
                exit(1);
            }
            if(col_index==0){
                std::cerr<<"Market file is zero-indexed"<<std::endl;
                exit(1);
            }
            coo.row_indices[i]=row_index - 1;
            coo.column_indices[i]=col_index - 1;
            coo.nonzero_values[i]=weight;
        }
    }
    else{
        std::cerr<<"Unrecognized matrix market format type"<<std::endl;
        exit(1);
    }
    //TODO: only the entries on or below the main diagonal are stored in the mtx
    //* undirected graph
    if(mm_is_symmetric(code_)){
        num_t off_diagonals=0;
        for(int i=0;i<num_nonzeros;i+=1){
            if(coo.row_indices[i]!=coo.column_indices[i]) off_diagonals+=1;
        }
        num_t nonzeros=2U*off_diagonals+(num_nonzeros-off_diagonals);

        std::vector<int> temp_row_indices(nonzeros);
        std::vector<int> temp_column_indices(nonzeros);
        std::vector<double> temp_nonzero_values(nonzeros);

        num_t temp_count=0;
        for(int i=0;i<coo.num_nonzeros;i+=1){
            if(coo.row_indices[i]!=coo.column_indices[i]){
                temp_row_indices[temp_count]=coo.row_indices[i];
                temp_column_indices[temp_count]=coo.column_indices[i];
                temp_nonzero_values[temp_count++]=coo.nonzero_values[i];

                temp_row_indices[temp_count]=coo.column_indices[i];
                temp_column_indices[temp_count]=coo.row_indices[i];
                temp_nonzero_values[temp_count++]=coo.nonzero_values[i];
            }
            else{
                temp_row_indices[temp_count]=coo.row_indices[i];
                temp_column_indices[temp_count]=coo.column_indices[i];
                temp_nonzero_values[temp_count++]=coo.nonzero_values[i];
            }
        }
        coo.row_indices=temp_row_indices;
        coo.column_indices=temp_column_indices;
        coo.nonzero_values=temp_nonzero_values;
        coo.num_nonzeros=nonzeros;
    }
    fclose(file);
    ConvertCoordinateToCSR(coo);
    ComputeGraphInfo();
}

void Graph::ConvertCoordinateToCSR(const CoordinateMtx& coo){
    int number_of_rows=coo.num_rows;
    // int number_of_columns=coo.num_columns;
    int number_of_nonzeros=coo.num_nonzeros;

    offsets_.resize(number_of_rows+1);
    adjs_.resize(number_of_nonzeros);
    weights_.resize(number_of_nonzeros);

    for(int i=0;i<number_of_nonzeros;i+=1){
        offsets_[coo.row_indices[i]]+=1;
    }
    //TODO: offsets
    int prefix_sum=0;
    for(int i=0;i<number_of_rows;i+=1){
        int temp=offsets_[i];
        offsets_[i]=prefix_sum;
        prefix_sum+=temp;
    }
    offsets_[number_of_rows]=prefix_sum;
    for(int i=0;i<number_of_nonzeros;i+=1){
        int src=coo.row_indices[i];
        int dst=coo.column_indices[i];
        adjs_[offsets_[src]]=dst;
        weights_[offsets_[src]]=dst;
        offsets_[src]+=1;
    }
    //* update offsets
    for(int i=0,last=0;i<=number_of_rows;i+=1){
        int temp=offsets_[i];
        offsets_[i]=last;
        last=temp;
    }
}

void Graph::ComputeGraphInfo(){
    node_num_=offsets_.size()-1;
    edge_num_=adjs_.size();
    min_degree_=node_num_;
    max_degree_=0;
    for(int i=0;i<node_num_;i++){
        min_degree_=std::min(min_degree_,offsets_[i+1]-offsets_[i]);
        max_degree_=std::max(max_degree_,offsets_[i+1]-offsets_[i]);
    }
    avg_degree_=1.0*edge_num_/node_num_;
    
    std::cout<<"==========>>> Graph Info <<<==========\n";
    printf("Nodes = %d, Edges = %d\n",node_num_,edge_num_);
    printf("Min Degree = %d, Max Degree = %d, Avg Degree = %.2f\n",min_degree_,max_degree_,avg_degree_);
    // std::cout<<"======================================\n";
}

graph_info_t Graph::GetGraphInfo(){
    return graph_info_t{node_num_,edge_num_,min_degree_,max_degree_,avg_degree_};
}

int* Graph::GetOffsets(){
    return offsets_.data();
}

int* Graph::GetAdjs(){
    return adjs_.data();
}

double* Graph::GetEdgeWeights(){
    return weights_.data();
}
