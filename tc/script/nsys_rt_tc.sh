#!/bin/bash

#TODO: RT-1A2
# nsys profile -o output/profile/rt_1A2/dblp --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/com-dblp/com-dblp.ungraph.edge.pd 1

# nsys profile -o output/profile/rt_1A2/youtube --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/com-youtube/com-youtube.ungraph.edge.pd 1

# nsys profile -o output/profile/rt_1A2/patents --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/cit-Patents/cit-Patents.edge.pd 1

# nsys profile -o output/profile/rt_1A2/wiki --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/wiki-Talk/wiki-Talk.edge.pd 1

#TODO: RT-2A1
nsys profile -o output/profile/rt_2A1/dblp --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/com-dblp/com-dblp.ungraph.edge.pd 1

nsys profile -o output/profile/rt_2A1/youtube --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/com-youtube/com-youtube.ungraph.edge.pd 1

nsys profile -o output/profile/rt_2A1/patents --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/cit-Patents/cit-Patents.edge.pd 1

nsys profile -o output/profile/rt_2A1/wiki --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/wiki-Talk/wiki-Talk.edge.pd 1

nsys profile -o output/profile/rt_2A1/com-lj --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/com-lj/com-lj.ungraph.edge.pd 1

nsys profile -o output/profile/rt_2A1/ljournal --cuda-memory-usage true --enable nvml_metrics,-i1 ./bin/rt_tc dataset/soc-LiveJournal1/soc-LiveJournal1.edge.pd 1