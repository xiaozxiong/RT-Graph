#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_set>

#include "RTBase.h"
// #include "casting_kernels.h"
#include "cxxopts.hpp"
#include "helper.h"
#include "simple_bs.h"
#include "utils.h"

int GetPow(int x) {
    int n = 0;
    while (x) {
        x >>= 1;
        n += 1;
    }
    return n - 1;
}

void InitData(std::vector<int> &data, int data_size, std::vector<int> &query, int query_size,
              int &max_element) {
    int n = GetPow(data_size);
    std::cout << "=====>>> 2^" << n << ":\n";
    std::cout << "data size = " << data_size << ", query size = " << query_size
              << ", distribution = uniform\n";
    std::default_random_engine generator;

    int minn = 0, maxn = std::min(INT32_MAX, data_size * 10); //*
    std::uniform_int_distribution<int> distribution(minn, maxn);
    // Avoid duplicate elements in data
    // std::unordered_set<int> set;
    // for(int i=0;i<data_size;i++){
    //     int x=distribution(generator);
    //     while(set.count(x)>0) x=distribution(generator);
    //     data[i]=x;
    //     set.insert(x);
    // }

    //* ===== generate data and query =====
    data.resize(data_size);
    for (int i = 0; i < data_size; i++)
        data[i] = distribution(generator);
    //* sort data
    // std::sort(data.begin(),data.end());
    query.resize(query_size);
    for (int i = 0; i < query_size; i++)
        query[i] = distribution(generator); // query is not sorted

    //* sort query
    // std::sort(query,query+query_size);

    max_element = maxn;
    printf("data and query are both unsorted\n");
}

void SetDevice(int device_id = 0) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cudaSetDevice(device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    size_t available_memory, total_memory;
    cudaMemGetInfo(&available_memory, &total_memory);
    // std::cout<<"Total GPUs visible: "<<device_count;
    // std::cout<<", using ["<<device_id<<"]: "<<device_prop.name<<std::endl;
    // std::cout<<"Available Memory: "<<int(available_memory/1024/1024)<<" MB, ";
    // std::cout<<"Total Memory: "<<int(total_memory/1024/1024)<<" MB\n";
}

void RTFunction(std::vector<int> &data, std::vector<int> &query, int *rt_results, int max_element) {
    //* compute scene parameters
    SceneParameter scene_params;
    scene_params.mod = (int)pow(1.0 * max_element, 1.0 / 3); // 1000
    scene_params.axis_offsets = make_float3(scene_params.mod / 2, scene_params.mod / 2,
                                            scene_params.mod / 2); // make_float3(0,0,0);

    RTBase rt_search = RTBase(scene_params);
    rt_search.Setup();
    rt_search.BuildAccel(data);          // build BVH
    rt_search.Search(query, rt_results); // launch rays
    rt_search.CleanUp();
}

void HostBinarySearch(std::vector<int> &data, std::vector<int> &query, int *host_results) {
    //* host sorting
    std::sort(data.begin(), data.end());
    //* host search
    for (int i = 0; i < query.size(); i++) {
        auto index = std::lower_bound(data.begin(), data.end(), query[i]) - data.begin();
        host_results[i] = (data[index] == query[i] ? index : -1);
    }
}

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], "Compare the performance of searching");
        // clang-format off
        options.add_options()
            ("d,data", "data size", cxxopts::value<int>() -> default_value("1024"))
            ("q,query", "query size", cxxopts::value<int>() -> default_value("16777216")) // 1<<24
            ("g,device", "device id", cxxopts::value<int>() -> default_value("2"))
            // ("p,primitive","optix build input type",cxxopts::value<int>() -> default_value("0"))
            ("h,help", "print help")
        ;
        // clang-format on
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        // arguments
        int data_size = result["data"].as<int>();
        int query_size = result["query"].as<int>();
        int device_id = result["device"].as<int>();
        // int type=result["primitive"].as<int>();

        SetDevice(device_id);

        std::vector<int> data, query;
        int max_element = 0;
        InitData(data, data_size, query, query_size, max_element);

        //* result
        // binary search cost = log(n)
        auto host_results = new int[query_size];
        auto bs_results = new int[query_size];
        auto rt_results = new int[query_size];

        BSAccessCountFunction(data.data(), data.size(), query.data(), query.size(), bs_results);
        HostBinarySearch(data, query, host_results);
        RTFunction(data, query, rt_results, max_element);
        printf("BS: ");
        Check(host_results, bs_results, query_size);
        printf("RT: ");
        Check(host_results, rt_results, query_size);
        // std::cout<<"Data: ";
        // for(int i=0;i<data.size();i++) std::cout<<data[i]<<" ";
        // std::cout<<std::endl;
        // std::cout<<"Query: ";
        // for(int i=0;i<query.size();i++) std::cout<<query[i]<<" ";
        // std::cout<<std::endl;
        // std::cout<<"host: ";
        // for(int i=0;i<query.size();i++) std::cout<<host_results[i]<<" ";
        // std::cout<<std::endl;
        // std::cout<<"RT: ";
        // for(int i=0;i<query.size();i++) std::cout<<rt_results[i]<<" ";
        // std::cout<<std::endl;

        delete[] rt_results;
        delete[] bs_results;
        delete[] host_results;

        std::cout << "\n";
    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}