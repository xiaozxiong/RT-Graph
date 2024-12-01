#include "util.h"
#include "cxxopts.hpp"
#include "rt_intersection.h"

int main(int argc,char *argv[]){
    try{
        cxxopts::Options options(argv[0],"RT based Set Itersection");
        options.add_options()
            ("device","chose GPU",cxxopts::value<uint>())
            ("dataset","path of dataset",cxxopts::value<std::string>())
            ("chunk_length","length of chunk in z axis",cxxopts::value<uint>())
            ("h,help","Print Usage")
        ;
        auto result=options.parse(argc,argv);
        if(result.count("help")){
            std::cout<<options.help()<<std::endl;
            exit(0);
        }
        //* arguments
        uint device_id=result["device"].as<uint>();
        std::string path=result["dataset"].as<std::string>();
        uint chunk_length=result["chunk_length"].as<uint>();

        //* data
        Dataset dataset;
        ReadData(path,dataset);
        PrintDataInfo(dataset);
        std::vector<uint> count_results;

        RTInter rt_inter(chunk_length,device_id);
        rt_inter.BuildBVHAndComputeRay(dataset);
        rt_inter.CountIntersection(count_results);
        Check(dataset,count_results);

    }
    catch(const cxxopts::OptionException& e){
        std::cerr<<e.what()<<std::endl;
        exit(1);
    }
    return 0;
}