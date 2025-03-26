#!/bin/bash

nsys profile -o profile/linear-bfs/hollywood --cuda-memory-usage true ./bin/linear_bfs dataset/hollywood-2009.mtx 4 3 1

nsys profile -o profile/linear-bfs/kron --cuda-memory-usage true ./bin/linear_bfs dataset/kron_g500-logn21.mtx 2 3 1

nsys profile -o profile/linear-bfs/LiveJournal1 --cuda-memory-usage true ./bin/linear_bfs dataset/soc-LiveJournal1.mtx 3 3 1

nsys profile -o profile/linear-bfs/orkut --cuda-memory-usage true ./bin/linear_bfs dataset/soc-orkut.mtx 6 3 1

nsys profile -o profile/linear-bfs/road_usa --cuda-memory-usage true ./bin/linear_bfs dataset/road_usa.mtx 1 3 0

nsys profile -o profile/linear-bfs/twitter --cuda-memory-usage true ./bin/linear_bfs dataset/soc-twitter-2010.mtx 4 1 1