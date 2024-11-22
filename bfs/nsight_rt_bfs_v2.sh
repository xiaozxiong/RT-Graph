#!/bin/bash

# ========================== instructions to global memory ==========================
echo "measure instructions"
# rt_bfs_v2_no_center
echo "hollywood:"
ncu --csv -k regex:optixLaunch --metrics sm__sass_inst_executed_op_global.sum ./bin/rt_bfs_v2 datasets/hollywood-2009/hollywood-2009.mtx 4 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_hollywood.csv
echo "kron_g500:"
ncu --csv -k regex:optixLaunch --metrics sm__sass_inst_executed_op_global.sum ./bin/rt_bfs_v2 datasets/kron_g500-logn21/kron_g500-logn21.mtx 2 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_kron.csv
echo "soc-LiveJournal1:"
ncu --csv -k regex:optixLaunch --metrics sm__sass_inst_executed_op_global.sum ./bin/rt_bfs_v2 datasets/soc-LiveJournal1/soc-LiveJournal1.mtx 2 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_soc-lj.csv
echo "soc-orkut:"
ncu --csv -k regex:optixLaunch --metrics sm__sass_inst_executed_op_global.sum ./bin/rt_bfs_v2 datasets/soc-orkut/soc-orkut.mtx 2 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_soc-orkut.csv
# echo "road_usa"

# ========================== Section: MemoryWorkloadAnalysis ==========================
echo "measure memory workload"
echo "hollywood"
ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_bfs_v2 datasets/hollywood-2009/hollywood-2009.mtx 4 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_memory_hollywood.csv

echo "kron"
ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_bfs_v2 datasets/kron_g500-logn21/kron_g500-logn21.mtx 2 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_memory_kron.csv

echo "soc-lj"
ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_bfs_v2 datasets/soc-LiveJournal1/soc-LiveJournal1.mtx 2 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_memory_soc-lj.csv

echo "soc-orkut"
ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_bfs_v2 datasets/soc-orkut/soc-orkut.mtx 2 2 3 > profiling/rt_v2_no_center/rt_v2_no_center_memory_soc-orkut.csv

