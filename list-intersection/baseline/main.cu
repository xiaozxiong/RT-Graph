#include "common.h"
#include "set_intersection.h"

// #include <fmt/core.h>
#include <cxxopts.hpp>
#include <iostream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    try {
        cxxopts::Options options(argv[0], "Binary Search based Set Intesection");
        options.add_options()("device", "chose GPU", cxxopts::value<uint>())(
            "dataset", "path of dataset", cxxopts::value<std::string>())(
            "baseline", "chose the type of baseline", cxxopts::value<int>())("help", "print help");
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        // arguments
        uint device_id = result["device"].as<uint>();
        std::string path = result["dataset"].as<std::string>();
        Baseline baseline = static_cast<Baseline>(result["baseline"].as<int>());

        // query device attributes
        uint device_count;
        cuDeviceGetCount((int *)&device_count);
        if (device_id >= device_count) {
            std::cerr << "Device id (" << device_id << ") is larger than device count ("
                      << device_count << ")" << std::endl;
            exit(1);
        }
        uint max_shared_memory_per_block;
        cuDeviceGetAttribute((int *)(&max_shared_memory_per_block),
                             CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device_id);
        uint max_thread_per_block;
        cuDeviceGetAttribute((int *)&max_thread_per_block,
                             CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device_id);
        std::cout << "Max shared memory per block = " << max_shared_memory_per_block
                  << ", max thread per block = " << max_thread_per_block << std::endl;

        SetIntersection(path, baseline, device_id);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }

    return 0;
}