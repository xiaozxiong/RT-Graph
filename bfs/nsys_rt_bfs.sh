#!/bin/bash

# ============== rt-bfs-enc
# Usage: ./rt_bfs_v2 graph_path chunk_length digit device_id filter

nsys profile -o profile/rt-bfs-enc/hollywood --cuda-memory-usage true ./bin/rt_bfs_v2 dataset/hollywood-2009.mtx 4 2 3 1

nsys profile -o profile/rt-bfs-enc/kron --cuda-memory-usage true ./bin/rt_bfs_v2 dataset/kron_g500-logn21.mtx 2 2 3 1

nsys profile -o profile/rt-bfs-enc/LiveJournal1 --cuda-memory-usage true ./bin/rt_bfs_v2 dataset/soc-LiveJournal1.mtx 3 2 3 1

nsys profile -o profile/rt-bfs-enc/orkut --cuda-memory-usage true ./bin/rt_bfs_v2 dataset/soc-orkut.mtx 6 2 3 1

nsys profile -o profile/rt-bfs-enc/road_usa  --cuda-memory-usage true ./bin/rt_bfs_v2 dataset/road_usa.mtx 1 2 3 0

# ============== rt_bfs
# Usage: ./rt_bfs graph_path chunk_length device_id filter

nsys profile -o profile/rt-bfs/hollywood --cuda-memory-usage true ./bin/rt_bfs dataset/hollywood-2009.mtx 4 3 1

nsys profile -o profile/rt-bfs/kron --cuda-memory-usage true ./bin/rt_bfs dataset/kron_g500-logn21.mtx 2 3 1

nsys profile -o profile/rt-bfs/LiveJournal1 --cuda-memory-usage true ./bin/rt_bfs dataset/soc-LiveJournal1.mtx 3 3 1

nsys profile -o profile/rt-bfs/orkut --cuda-memory-usage true ./bin/rt_bfs dataset/soc-orkut.mtx 6 3 1

nsys profile -o profile/rt-bfs/road_usa --cuda-memory-usage true ./bin/rt_bfs dataset/road_usa.mtx 1 3 0