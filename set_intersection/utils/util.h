#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using uint = unsigned int;

// a tile includes multiple sets
struct Tile {
    uint start;
    uint end;
    uint size;
};

using tile_t = Tile;

struct Dataset {
    uint num_of_sets_a; // first
    uint num_of_sets_b; // last
    uint num_of_sets;
    uint max_element;
    uint intersection;
    std::vector<uint> set_offsets; // offset in elements
    // std::vector<uint> intersection_sizes; // A[0]-B[0, ...], A[1]-B[0, ...], ...
    std::vector<uint> elements; //
};

enum Baseline { BinarySearch = 0, Hash = 1, Bitmap = 2, IntersectPath = 3 };

void ReadData(const std::string &path, Dataset &dataset);

void PrintDataInfo(const Dataset &dataset);

void Check(const Dataset &dataset, const std::vector<uint> &results);