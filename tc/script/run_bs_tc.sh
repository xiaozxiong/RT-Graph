#!/bin/bash

# Usage: ./bs_tc graph_file device_id

echo "======================dblp:"
./bin/bs_tc dataset/com-dblp/com-dblp.ungraph.edge.pd 0

echo "======================youtube:"
./bin/bs_tc dataset/com-youtube/com-youtube.ungraph.edge.pd 0

echo "======================patents:"
./bin/bs_tc dataset/cit-Patents/cit-Patents.edge.pd 0

echo "======================wiki:"
./bin/bs_tc dataset/wiki-Talk/wiki-Talk.edge.pd 0

echo "======================com-lj:"
./bin/bs_tc dataset/com-lj/com-lj.ungraph.edge.pd 0

echo "======================soc-LiveJournal1:"
./bin/bs_tc dataset/soc-LiveJournal1/soc-LiveJournal1.edge.pd 0

echo "======================com-orkut:"
./bin/bs_tc dataset/com-orkut/com-orkut.ungraph.edge 0

# synthesised power-law graph
echo "======================2M_m8:"
./bin/bs_tc dataset/synthesised/2M_m8.edge 0