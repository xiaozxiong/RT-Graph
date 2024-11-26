#!/bin/bash

# rt_1A2, rt_2A1, bs_1A2, bs_2A1
method=$1

# these command line maybe useless due to the bug of Nsight

# echo "======================dblp:"
# ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_tc dataset/com-dblp/com-dblp.ungraph.edge.pd 0 > output/profile/$method/dblp.csv

# echo "======================youtube:"
# ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_tc dataset/com-youtube/com-youtube.ungraph.edge.pd 0 > output/profile/$method/youtube.csv

# echo "======================patents:"
# ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_tc dataset/cit-Patents/cit-Patents.edge.pd 0 > output/profile/$method/patents.csv

# echo "======================wiki:"
# ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_tc dataset/wiki-Talk/wiki-Talk.edge.pd 0 > output/profile/$method/wiki.csv

# echo "======================com-lj:"
# ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_tc dataset/com-lj/com-lj.ungraph.edge.pd 0 > output/profile/$method/com-lj.csv

# echo "======================soc-LiveJournal1:"
# ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_tc dataset/soc-LiveJournal1/soc-LiveJournal1.edge.pd 0 > output/profile/$method/soc-LiveJournal1.csv

echo "======================2M_m8:"
ncu --csv -k regex:optixLaunch --section MemoryWorkloadAnalysis ./bin/rt_tc dataset/synthesised/2M_m8.edge 0 > output/profile/$method/2M_m8.csv