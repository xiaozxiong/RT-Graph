#pragma once

#include <random>
#include <stdint.h>
#include <string>
#include <vector>

struct DataInfo {
    uint32_t num_of_sets_a; // first
    uint32_t num_of_sets_b; // last
    uint32_t a_length;
    uint32_t b_length;
    uint32_t c_length;
    uint32_t max_element;
};

enum Distribution { Uniform = 0, Normal = 1, Exponential = 2 };

class Data {
    void UniformGenerator(std::default_random_engine &generator, uint32_t len_a, uint32_t len_b,
                          uint32_t len_c, uint32_t min_ele, uint32_t max_ele);
    void NormalGenerator(std::default_random_engine &generator, uint32_t len_a, uint32_t len_b,
                         uint32_t len_c, uint32_t min_ele, uint32_t max_ele);
    void ExponentialGenerator(std::default_random_engine &generator, uint32_t len_a, uint32_t len_b,
                              uint32_t len_c, uint32_t min_ele, uint32_t max_ele,
                              double lambda = 1.0);

public:
    Data(uint num_of_sets_a, uint num_of_sets_b = 1);
    ~Data();
    void Generator(uint32_t len, double skew_ratio, double selectivity,
                   double density); // single set
    void Generator(uint32_t len, double skew_ratio, double selectivity, double density,
                   double lambda); // single set
    void Generator(uint32_t len, double skew_ratio, double selectivity,
                   Distribution distribution_type, double lambda = 1.0); // multiple sets
    void Writer(std::string dir = "../../dataset/");

private:
    uint num_of_sets_a_{0};
    uint num_of_sets_b_{0};
    std::string data_name_;
    std::vector<std::vector<uint32_t>> a_sets_;
    std::vector<std::vector<uint32_t>> b_sets_;
    std::vector<uint> set_offsets_;
    DataInfo data_info_;
};