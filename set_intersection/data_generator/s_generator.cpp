#include "data.h"
#include <cxxopts.hpp>

int main(int argc, char *argv[]) {

    try {
        cxxopts::Options options(argv[0], "Data Generator");
        options.add_options()("a,num_a", "number of set A",
                              cxxopts::value<uint32_t>()->default_value("1000"))(
            "l,len", "length of set A", cxxopts::value<uint32_t>())(
            "r,skew_ratio", "Skew ratio", cxxopts::value<double>())("s,selectivity", "Selectivity",
                                                                    cxxopts::value<double>())(
            "d,density", "Density", cxxopts::value<double>()->default_value("0.1"))(
            "b,lambda", "lambda in exponential", cxxopts::value<double>()->default_value("0"))(
            "n,num_b", "number of sets B per set A", cxxopts::value<uint>()->default_value("1"))(
            "f,output_dir", "Directory of output binary file",
            cxxopts::value<std::string>()->default_value("../dataset/"))("h,help", "Print Usage");
        auto result = options.parse(argc, argv);
        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }
        uint32_t len = result["len"].as<uint32_t>();
        double skew_ratio = result["skew_ratio"].as<double>();
        double selectivity = result["selectivity"].as<double>();
        double density = result["density"].as<double>();
        double lambda = result["lambda"].as<double>();
        bool vary_distribution = (lambda > 0);
        uint num_of_sets_b = result["num_b"].as<uint>();
        std::string output_dir = result["output_dir"].as<std::string>();

        uint num_of_sets_a = result["num_a"].as<uint>();
        Data data(num_of_sets_a, num_of_sets_b);
        if (vary_distribution) {
            data.Generator(len, skew_ratio, selectivity, density, lambda);
        } else {
            data.Generator(len, skew_ratio, selectivity, density);
        }
        data.Writer(output_dir);

    } catch (const cxxopts::OptionException &e) {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}