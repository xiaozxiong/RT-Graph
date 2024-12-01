#include "data.h"
#include <cxxopts.hpp>

int main(int argc, char *argv[]) {

    try {
        cxxopts::Options options(argv[0], "Data Generator");
        options.add_options()("l,len", "Size of set A", cxxopts::value<uint32_t>())(
            "s,selectivity", "Selectivity", cxxopts::value<double>())(
            "n,num_b", "number of sets B per set A", cxxopts::value<uint>()->default_value("1"))(
            "d,distribution", "distribution of set", cxxopts::value<uint>()->default_value("0"))(
            "b,lambda", "lambda of exponential distribution",
            cxxopts::value<double>()->default_value("1"))(
            "r,skew_ratio", "skew ratio", cxxopts::value<double>()->default_value("1"))(
            "f,output_dir", "Directory of output binary file",
            cxxopts::value<std::string>()->default_value("../../dataset/"))("h,help",
                                                                            "Print Usage");
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        uint32_t len = result["len"].as<uint32_t>();
        double selectivity = result["selectivity"].as<double>();
        uint num_of_sets_b = result["num_b"].as<uint>();
        Distribution distribution = static_cast<Distribution>(result["distribution"].as<uint>());
        double lambda = result["lambda"].as<double>();
        double skew = result["skew_ratio"].as<double>();
        std::string output_dir = result["output_dir"].as<std::string>();

        uint num_of_sets_a = 1000U;
        Data data(num_of_sets_a, num_of_sets_b);
        data.Generator(len, skew, selectivity, distribution, lambda);
        data.Writer(output_dir);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}

/*
len =
*/