#include "util.h"

void ReadData(const std::string &path,Dataset &dataset){
    std::ifstream infile;
    infile.open(path,std::ios::binary|std::ios::in);
    infile.read(reinterpret_cast<char*>(&dataset.num_of_sets_a),sizeof(uint));
    infile.read(reinterpret_cast<char*>(&dataset.num_of_sets_b),sizeof(uint));
    infile.read(reinterpret_cast<char*>(&dataset.max_element),sizeof(uint));
    infile.read(reinterpret_cast<char*>(&dataset.intersection),sizeof(uint));
    
    dataset.num_of_sets=dataset.num_of_sets_a+dataset.num_of_sets_b; // total number of sets
    // size_t combinations=(size_t)dataset.num_of_sets_a*dataset.num_of_sets_b;
    dataset.set_offsets.resize(dataset.num_of_sets+1);
    // dataset.intersection_sizes.resize(combinations);
    infile.read(reinterpret_cast<char*>(dataset.set_offsets.data()),sizeof(uint)*(dataset.num_of_sets+1));
    // infile.read(reinterpret_cast<char*>(dataset.intersection_sizes.data()),sizeof(uint)*combinations);
    // read all elements
    uint element_size=dataset.set_offsets[dataset.num_of_sets];
    dataset.elements.resize(element_size);
    infile.read(reinterpret_cast<char*>(dataset.elements.data()),sizeof(uint)*element_size);

    infile.close();
}

void PrintDataInfo(const Dataset& dataset){
    std::cout<<"Dataset: number of sets = ("<<dataset.num_of_sets_a<<", "<<dataset.num_of_sets_b<<")"<<std::endl;
    std::cout<<"Dataset: length of set a = "<<dataset.set_offsets[1]-dataset.set_offsets[0];
    std::cout<<", length of set b = "<<dataset.set_offsets[dataset.num_of_sets]-dataset.set_offsets[dataset.num_of_sets-1]<<std::endl;
    // std::cout<<"Dataset: A set size =";
    // for(int i=0;i<dataset.num_of_sets_a;i+=1) std::cout<<" "<<dataset.set_offsets[i+1]-dataset.set_offsets[i];
    // std::cout<<"Dataset: B set size =";
    // for(int i=dataset.num_of_sets_a;i<dataset.num_of_sets;i+=1) std::cout<<" "<<dataset.set_offsets[i+1]-dataset.set_offsets[i];
    // std::cout<<"\n";
    std::cout<<"Dataset: intersection size = "<<dataset.intersection<<std::endl;
    // for(auto s: dataset.intersection_sizes) std::cout<<" "<<s;
    // std::cout<<"\n";
}

void Check(const Dataset &dataset,const std::vector<uint> &results){
    // if(results.size()!=dataset.intersection_sizes.size()){
    //     std::cout<<"Check: results size is incorrect"<<std::endl;
    //     return;
    // }
    for(int i=0;i<results.size();i+=1){
        if(results[i]!=dataset.intersection){
            std::cout<<"Check: "<<i<<"-th result ("<<results[i]<<" != "<<dataset.intersection<<") is incorrect"<<std::endl;
            return;
        }
    }
    std::cout<<"Check: accepted"<<std::endl;
}