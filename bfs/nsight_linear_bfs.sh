#!/bin/bash

echo "hollywood"
ncu --csv -k ExpandOneLevel --section MemoryWorkloadAnalysis ./bin/linear_bfs datasets/hollywood-2009/hollywood-2009.mtx 4 3 > profiling/linear/linear_memory_hollywood.csv

echo "kron"
ncu --csv -k ExpandOneLevel --section MemoryWorkloadAnalysis ./bin/linear_bfs datasets/kron_g500-logn21/kron_g500-logn21.mtx 2 3 > profiling/linear/linear_memory_kron.csv

echo "soc-lj"
ncu --csv -k ExpandOneLevel --section MemoryWorkloadAnalysis ./bin/linear_bfs datasets/soc-LiveJournal1/soc-LiveJournal1.mtx 2 3 > profiling/linear/linear_memory_soc-lj.csv

echo "soc-orkut"
ncu --csv -k ExpandOneLevel --section MemoryWorkloadAnalysis ./bin/linear_bfs datasets/soc-orkut/soc-orkut.mtx 2 3 > profiling/linear/linear_memory_soc-orkut.csv


