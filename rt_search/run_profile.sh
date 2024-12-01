#!/bin/bash

for i in $(seq 0 3 27); do
    data_size=$((1 << i))
    ncu -o profile/q24_d"$i" --section MemoryWorkloadAnalysis ./bin/measure -d "$data_size" --device 3
done

for i in $(seq 0 3 27); do
    data_size=$((1 << i))
    ncu -o profile/q16_d"$i" --section MemoryWorkloadAnalysis ./bin/measure -d "$data_size" --device 3 -q 65536
done