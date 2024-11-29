#include "data.h"

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unordered_set>

Data::Data(uint num_of_sets_a, uint num_of_sets_b)
    : num_of_sets_a_(num_of_sets_a), num_of_sets_b_(num_of_sets_b) {
    a_sets_.resize(num_of_sets_a);
    b_sets_.resize(num_of_sets_b);
}

Data::~Data() {}

void Data::Generator(uint32_t len, double skew_ratio, double selectivity, double density) {
    assert(b_sets_.size() == 1U);
    assert(selectivity >= 0 && selectivity <= 1.0);
    uint32_t len_a = len;
    uint32_t len_b = static_cast<uint32_t>(len / skew_ratio);
    uint32_t len_c = static_cast<uint32_t>(len * selectivity);
    assert(len_c <= len_a && len_c <= len_b);

    uint32_t range = std::min(static_cast<uint32_t>((len_a + len_b - len_c) / density),
                              static_cast<uint32_t>(RAND_MAX));
    std::cout << "RAND_MAX: " << RAND_MAX << "\n";
    // generate a set of length len_a
    std::vector<uint32_t> tmp(len_a);
    std::unordered_set<uint32_t> element_set;
    for (int i = 0; i < len_a; i += 1) {
        int x = 0;
        do {
            x = rand() % range;
        } while (element_set.find(x) != element_set.end());
        element_set.insert(x);
        tmp[i] = x;
    }
    std::sort(tmp.begin(), tmp.end());
    // generate sets A and B
    for (int i = 0; i < num_of_sets_a_; i += 1)
        a_sets_[i] = tmp; // sets A are all the same
    for (int i = 0; i < num_of_sets_b_; i += 1) {
        b_sets_[i].resize(len_b);
        memcpy(b_sets_[i].data(), tmp.data(), sizeof(uint32_t) * len_c);
        int x = 0;
        std::unordered_set<uint32_t> tmp_set = element_set;
        for (uint32_t j = len_c; j < len_b; j += 1) {
            do {
                x = rand() % range;
            } while (tmp_set.find(x) != tmp_set.end());
            tmp_set.insert(x);
            b_sets_[i][j] = x;
        }
        std::sort(b_sets_[i].begin(), b_sets_[i].end());
    }
    // offset of sets
    uint num_of_sets = num_of_sets_a_ + num_of_sets_b_;
    set_offsets_.resize(num_of_sets + 1);
    uint offset = 0U;
    for (int i = 0; i < num_of_sets; i += 1) {
        set_offsets_[i] = offset;
        if (i < num_of_sets_a_)
            offset += len_a;
        else
            offset += len_b;
    }
    set_offsets_[num_of_sets] = offset;
    // store data information
    data_info_ = DataInfo{num_of_sets_a_, num_of_sets_b_, len_a, len_b, len_c, range};

    std::string suffix = ".s_data";
    data_name_ = std::to_string(len_a) + "_" + std::to_string(len_b) + "_" + std::to_string(len_c) +
                 "_" + std::to_string(range) + suffix;

    std::cout << "Data: length of set A = " << len_a << ", length of set B = " << len_b
              << ", length of set C = " << len_c << ", element range = " << range << std::endl;
}
//* for varying distribution
void Data::Generator(uint32_t len, double skew_ratio, double selectivity, double density,
                     double lambda) {
    assert(b_sets_.size() == 1U);
    assert(selectivity >= 0 && selectivity <= 1.0);
    uint32_t len_a = len;
    uint32_t len_b = static_cast<uint32_t>(len / skew_ratio);
    uint32_t len_c = static_cast<uint32_t>(len * selectivity);
    assert(len_c <= len_a && len_c <= len_b);

    // std::exponential_distribution<double> distribution(lambda);
    uint32_t range = std::min(static_cast<uint32_t>((len_a + len_b - len_c) / density),
                              static_cast<uint32_t>(RAND_MAX));
    std::cout << "range: " << range << "\n";

    std::default_random_engine generator;
    // double mean=1.0*range/2;
    // double std=1.0*range/10000;
    // std::normal_distribution<double> distribution(mean,std);
    std::exponential_distribution<double> distribution(lambda);

    // generate a set of length len_a
    std::vector<uint32_t> tmp(len_a);
    std::unordered_set<uint32_t> element_set;

    for (int i = 0; i < len_a; i += 1) {
        int x = 0;
        do {
            double number = distribution(generator);
            x = (uint)round(range * number);
            while (x > range) {
                number = distribution(generator);
                x = (uint)round(range * number);
            }
        } while (element_set.find(x) != element_set.end());
        element_set.insert(x);
        tmp[i] = x;
    }
    std::sort(tmp.begin(), tmp.end());
    // generate sets A and B
    for (int i = 0; i < num_of_sets_a_; i += 1)
        a_sets_[i] = tmp; // sets A are all the same
    for (int i = 0; i < num_of_sets_b_; i += 1) {
        b_sets_[i].resize(len_b);
        memcpy(b_sets_[i].data(), tmp.data(), sizeof(uint32_t) * len_c);
        int x = 0;
        std::unordered_set<uint32_t> tmp_set = element_set;
        for (uint32_t j = len_c; j < len_b; j += 1) {
            do {
                double number = distribution(generator);
                x = (uint)round(range * number);
                while (x > range) {
                    number = distribution(generator);
                    x = (uint)round(range * number);
                }
            } while (tmp_set.find(x) != tmp_set.end());
            tmp_set.insert(x);
            b_sets_[i][j] = x;
        }
        std::sort(b_sets_[i].begin(), b_sets_[i].end());
    }
    // offset of sets
    uint num_of_sets = num_of_sets_a_ + num_of_sets_b_;
    set_offsets_.resize(num_of_sets + 1);
    uint offset = 0U;
    for (int i = 0; i < num_of_sets; i += 1) {
        set_offsets_[i] = offset;
        if (i < num_of_sets_a_)
            offset += len_a;
        else
            offset += len_b;
    }
    set_offsets_[num_of_sets] = offset;
    // store data information
    data_info_ = DataInfo{num_of_sets_a_, num_of_sets_b_, len_a, len_b, len_c, range};

    std::string suffix = ".exponential"; //* , .exponential
    data_name_ = std::to_string(lambda) + "_" + std::to_string(len_a) + "_" +
                 std::to_string(len_b) + "_" + std::to_string(len_c) + "_" + std::to_string(range) +
                 suffix;

    std::cout << "Data: length of set A = " << len_a << ", length of set B = " << len_b
              << ", length of set C = " << len_c << ", element range = " << range << std::endl;
}

void Data::Generator(uint32_t len, double skew_ratio, double selectivity,
                     Distribution distribution_type, double lambda) {
    assert(b_sets_.size() >= 1U);
    assert(selectivity >= 0 && selectivity <= 1);
    uint32_t len_a = len;
    uint32_t len_b = static_cast<uint32_t>(len / skew_ratio); // len;
    uint32_t len_c = static_cast<uint32_t>(len * selectivity);
    const double density = 0.01; //*

    uint32_t min_ele = 0U;
    uint32_t max_ele = std::min(static_cast<uint32_t>((len_a + len_b - len_c) / density),
                                static_cast<uint32_t>(RAND_MAX));

    std::default_random_engine generator;

    std::string suffix = ".m_data";
    switch (distribution_type) {
    case Distribution::Uniform:
        UniformGenerator(generator, len_a, len_b, len_c, min_ele, max_ele);
        suffix = "_uni.m_data";
        break;
    case Distribution::Normal:
        NormalGenerator(generator, len_a, len_b, len_c, min_ele, max_ele);
        suffix = "_nor.m_data";
        break;
    case Distribution::Exponential:
        ExponentialGenerator(generator, len_a, len_b, len_c, min_ele, max_ele, lambda);
        suffix = "_" + std::to_string(int(lambda)) + "_exp.m_data";
        break;
    default:
        UniformGenerator(generator, len_a, len_b, len_c, min_ele, max_ele);
        suffix = "_uni.m_data";
    }
    // compute offset
    uint num_of_sets = num_of_sets_a_ + num_of_sets_b_;
    set_offsets_.resize(num_of_sets + 1);
    uint offset = 0U;
    for (int i = 0; i < num_of_sets; i += 1) {
        set_offsets_[i] = offset;
        if (i < num_of_sets_a_)
            offset += len_a;
        else
            offset += len_b;
    }
    set_offsets_[num_of_sets] = offset;
    // store data information
    data_info_ = DataInfo{num_of_sets_a_, num_of_sets_b_, len_a, len_b, len_c, max_ele};

    data_name_ = std::to_string(len_a) + "_" + std::to_string(len_b) + "_" + std::to_string(len_c) +
                 "_" + std::to_string(b_sets_.size()) + "_" + std::to_string(max_ele) + suffix;

    std::cout << "Data: length of set A = " << len_a << ", length of set B = " << len_b
              << ", length of set C = " << len_c << ", number of set B = " << b_sets_.size()
              << ", element range = " << max_ele << std::endl;
}

void Data::UniformGenerator(std::default_random_engine &generator, uint32_t len_a, uint32_t len_b,
                            uint32_t len_c, uint32_t min_ele, uint32_t max_ele) {
    std::uniform_int_distribution<uint32_t> distribution(min_ele, max_ele);
    // generate set A
    std::vector<uint32_t> tmp(len_a);
    std::unordered_set<uint32_t> element_set;
    for (int i = 0; i < len_a; i += 1) {
        int x = 0;
        do {
            x = distribution(generator);
        } while (element_set.find(x) != element_set.end());
        element_set.insert(x);
        tmp[i] = x;
    }
    std::sort(tmp.begin(), tmp.end());
    for (int i = 0; i < num_of_sets_a_; i += 1)
        a_sets_[i] = tmp; // sets A are all the same
    // generate set B
    for (int i = 0; i < num_of_sets_b_; i += 1) {
        b_sets_[i].resize(len_b);
        memcpy(b_sets_[i].data(), tmp.data(), sizeof(uint32_t) * len_c);
        int x = 0;
        std::unordered_set<uint32_t> tmp_set = element_set;
        for (uint32_t j = len_c; j < len_b; j += 1) {
            do {
                x = distribution(generator);
            } while (tmp_set.find(x) != tmp_set.end());
            tmp_set.insert(x);
            b_sets_[i][j] = x;
        }
        std::sort(b_sets_[i].begin(), b_sets_[i].end());
    }
}

void Data::NormalGenerator(std::default_random_engine &generator, uint32_t len_a, uint32_t len_b,
                           uint32_t len_c, uint32_t min_ele, uint32_t max_ele) {
    double mean = 1.0 * (min_ele + max_ele) / 2;
    double std = 1.0 * max_ele / 50;
    std::normal_distribution<double> distribution(mean, std);
    // generate set A
    std::vector<uint32_t> tmp(len_a);
    std::unordered_set<uint32_t> element_set;
    for (int i = 0; i < len_a; i += 1) {
        int x = 0;
        do {
            x = (uint)round(distribution(generator));
            while (x > max_ele) {
                x = (uint)round(distribution(generator));
            }
        } while (element_set.find(x) != element_set.end());
        element_set.insert(x);
        tmp[i] = x;
    }
    std::sort(tmp.begin(), tmp.end());
    for (int i = 0; i < num_of_sets_a_; i += 1)
        a_sets_[i] = tmp; // sets A are all the same
    // generate set B
    for (int i = 0; i < num_of_sets_b_; i += 1) {
        b_sets_[i].resize(len_b);
        memcpy(b_sets_[i].data(), tmp.data(), sizeof(uint32_t) * len_c);
        int x = 0;
        std::unordered_set<uint32_t> tmp_set = element_set;
        for (uint32_t j = len_c; j < len_b; j += 1) {
            do {
                x = (uint)round(distribution(generator));
                while (x > max_ele) {
                    x = (uint)round(distribution(generator));
                }
            } while (tmp_set.find(x) != tmp_set.end());
            tmp_set.insert(x);
            b_sets_[i][j] = x;
        }
        std::sort(b_sets_[i].begin(), b_sets_[i].end());
    }
}

void Data::ExponentialGenerator(std::default_random_engine &generator, uint32_t len_a,
                                uint32_t len_b, uint32_t len_c, uint32_t min_ele, uint32_t max_ele,
                                double lambda) {
    std::exponential_distribution<double> distribution(lambda); //!
    // generate set A
    std::vector<uint32_t> tmp(len_a);
    std::unordered_set<uint32_t> element_set;
    for (int i = 0; i < len_a; i += 1) {
        int x = 0;
        do {
            double number = distribution(generator);
            x = (uint)round(max_ele * number);
            while (x > max_ele) {
                number = distribution(generator);
                x = (uint)round(max_ele * number);
            }
        } while (element_set.find(x) != element_set.end());
        element_set.insert(x);
        tmp[i] = x;
    }
    std::sort(tmp.begin(), tmp.end());
    for (int i = 0; i < num_of_sets_a_; i += 1)
        a_sets_[i] = tmp; // sets A are all the same
    // generate set B
    for (int i = 0; i < num_of_sets_b_; i += 1) {
        b_sets_[i].resize(len_b);
        memcpy(b_sets_[i].data(), tmp.data(), sizeof(uint32_t) * len_c);
        int x = 0;
        std::unordered_set<uint32_t> tmp_set = element_set;
        for (uint32_t j = len_c; j < len_b; j += 1) {
            do {
                double number = distribution(generator);
                while (number >= 1.0) {
                    number = distribution(generator);
                }
                x = (uint)round(max_ele * number);
            } while (tmp_set.find(x) != tmp_set.end());
            tmp_set.insert(x);
            b_sets_[i][j] = x;
        }
        std::sort(b_sets_[i].begin(), b_sets_[i].end());
    }
}

void Data::Writer(std::string dir) {
    struct stat buffer;
    if (stat(dir.c_str(), &buffer) != 0) {
        std::cerr << "directory (" << dir << ") dose not exist" << std::endl;
        exit(1);
    }

    std::string path = dir + data_name_;
    std::ofstream data_file;
    data_file.open(path, std::ios::binary | std::ios::out);
    if (!data_file) {
        std::cerr << "File error" << std::endl;
        exit(1);
    }
    // write
    data_file.write(reinterpret_cast<char *>(&data_info_.num_of_sets_a), sizeof(uint32_t));
    data_file.write(reinterpret_cast<char *>(&data_info_.num_of_sets_b), sizeof(uint32_t));
    data_file.write(reinterpret_cast<char *>(&data_info_.max_element), sizeof(uint32_t));
    data_file.write(reinterpret_cast<char *>(&data_info_.c_length), sizeof(uint32_t));
    data_file.write(reinterpret_cast<char *>(set_offsets_.data()),
                    sizeof(uint32_t) * set_offsets_.size());
    for (int i = 0; i < data_info_.num_of_sets_a; i += 1)
        data_file.write(reinterpret_cast<char *>(a_sets_[i].data()),
                        sizeof(uint32_t) * a_sets_[i].size());
    for (int i = 0; i < data_info_.num_of_sets_b; i += 1)
        data_file.write(reinterpret_cast<char *>(b_sets_[i].data()),
                        sizeof(uint32_t) * b_sets_[i].size());
    data_file.close();
    std::cout << "Data has been written to path: " << path << std::endl;
}