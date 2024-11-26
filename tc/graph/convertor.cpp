#include "include/graph.h"

#include <string>

int main(int argc, char *argv[]){

    if(argc != 3){
        printf("Usage: ./bin/convertor txt_file bin_file");
        exit(1);
    }

    std::string txt_file = argv[1];
    std::string bin_file = argv[2];
    
    //  graph_convertor;
    Graph::Txt2Bin(txt_file, bin_file);

    return 0;
}