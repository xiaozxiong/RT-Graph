#!/bin/bash

# Usage: ./rt_tc graph_file device_id

echo "======================dblp:"
./bin/rt_tc dataset/com-dblp/com-dblp.ungraph.edge.pd 0

echo "======================youtube:"
./bin/rt_tc dataset/com-youtube/com-youtube.ungraph.edge.pd 0

echo "======================patents:"
./bin/rt_tc dataset/cit-Patents/cit-Patents.edge.pd 0

echo "======================wiki:"
./bin/rt_tc dataset/wiki-Talk/wiki-Talk.edge.pd 0

echo "======================com-lj:"
./bin/rt_tc dataset/com-lj/com-lj.ungraph.edge.pd 0

echo "======================soc-LiveJournal1:"
./bin/rt_tc dataset/soc-LiveJournal1/soc-LiveJournal1.edge.pd 0

# synthesised power-law graph
echo "======================2M_m8:"
./bin/rt_tc dataset/synthesised/2M_m8.edge 0