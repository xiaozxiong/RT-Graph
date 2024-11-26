#include "include/graph.h"

#include <cstdio>
#include <string>

int main(int argc, char *argv[]){
    if(argc != 3){
        printf("Usage: ./convertor input output\n");
        exit(1);
    }

    std::string input_file = argv[1];
    std::string out_file = argv[2];

    Graph graph;
    graph.Mtx2Parlay(input_file, out_file);


    return 0;
}