#!/bin/bash
# ============== rt_bfs_v2
# Usage: ./rt_bfs_v2 graph_path chunk_length digit device_id filter
./bin/rt_bfs_v2 dataset/hollywood-2009.mtx 4 2 3 1

./bin/rt_bfs_v2 dataset/kron_g500-logn21.mtx 2 2 3 1

./bin/rt_bfs_v2 dataset/soc-LiveJournal1.mtx 3 2 3 1

./bin/rt_bfs_v2 dataset/soc-orkut.mtx 6 2 3 1
# no filter
./bin/rt_bfs_v2 dataset/road_usa.mtx 1 2 3 0
# no filter
./bin/rt_bfs_v2 dataset/amazon-2008.mtx 2 3 0 0
# no filter
./bin/rt_bfs_v2 dataset/email-Enron.mtx 3 3 0 0

# ============== rt_bfs_v1
# Usage: ./rt_bfs graph_path chunk_length device_id filter
./bin/rt_bfs dataset/hollywood-2009.mtx 4 3 1

./bin/rt_bfs dataset/kron_g500-logn21.mtx 2 3 1

./bin/rt_bfs dataset/soc-LiveJournal1.mtx 3 3 1

./bin/rt_bfs dataset/soc-orkut.mtx 6 3 1
# Note: no filter
./bin/rt_bfs dataset/road_usa.mtx 1 3 0
# Note: no filter
./bin/rt_bfs dataset/amazon-2008.mtx 2 0 0
# Note: no filter
./bin/rt_bfs dataset/email-Enron.mtx 3 0 0