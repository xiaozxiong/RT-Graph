#!/bin/bash
# Usage: ./linear_bfs graph_path chunk_length device_id filter
./bin/linear_bfs dataset/hollywood-2009.mtx 4 3 1

./bin/linear_bfs dataset/kron_g500-logn21.mtx 2 3 1

./bin/linear_bfs dataset/soc-LiveJournal1.mtx 2 3 1

./bin/linear_bfs dataset/soc-LiveJournal1.mtx 6 3 1

./bin/linear_bfs dataset/road_usa.mtx 1 3 0

./bin/linear_bfs dataset/amazon-2008.mtx 2 0 0

./bin/linear_bfs dataset/email-Enron.mtx 3 0 0