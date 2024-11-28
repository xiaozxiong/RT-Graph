# RT or Not: Can We Accelerate Graph Algorithms with Ray Tracing Cores?

## Getting Started
### Prerequisites
- CUDA 11.8
- OptiX 7.5
- gcc 11

### Build

Download OptiX7.5 SDK from [NVIDIA OptiX™ Legacy Downloads](https://developer.nvidia.com/designworks/optix/downloads/legacy) and set an enviromental variable `OptiX_INSTALL_DIR` which refers to the include directory of OptiX.
```bash
export OptiX_INSTALL_DIR=/home/xzx/OptiX7.5/include
```
Then you can enter a diretory (e.g., bfs) and run the following commands:
```shell
cmake -B build
cd build
make -j
```


## Breadth First Search (BFS)
### Dataset

| Dataset | Nodes | Edges | Avg. Degree | Max Degree |
| :----: | ----:  |  ----:  | ----:  |  ----:  |
|hollywood-2009 | 1,139,905 |113,891,327 |99.91 |11,468 |
|kron_g500-logn21 | 2,097,152 | 182,082,942 | 86.82 | 213,905|
|soc-LiveJournal1 | 4,847,571 |68,993,773 |14.23 |20,293|
|soc-orkut | 2,997,166 |212,698,418 |70.97 |27,466|
|soc-twitter-2010 | 21,297,772 |530,051,354 |24.89 |698,112 |
|road_usa | 23,947,347 |57,708,624 |2.41 |9|
|amazon-2008 | 735,323 | 5,158,388 | 7.02 | 10 |
|email-Enron | 36,692 | 367,662 | 10.02 | 1,383 |

Most datasets can be found in [SuiteSparse Matrix Collection](https://sparse.tamu.edu/), we use the matrix market (.mtx) format for our program. Dataset `soc-orkut` and `soc-twitter-2010` are downloaded from [Network Repository](https://networkrepository.com/networks.php).

Note: All datasets are placed in a directory named `dataset`.

## Triangle Counting (TC)
### Dataset

| Dataset | Nodes | Edges | Triangles |
| :----: | ----:  |  ----:  | ----:  |
|com-dblp | 317,080 | 1,049,866 | 2,224,385 |
|com-youtube | 1,134,890 | 2,987,624 | 3,056,386 |
|cit-Patents | 3,774,768 | 16,518,948 | 7,515,023 |
|wiki-Talk | 2,394,385 | 5,021,410 | 9,203,519 |
|com-lj | 3,997,962 | 34,681,189 | 177,820,130 |
|soc-LiveJournal1 | 4,847,571 | 68,993,773 | 285,730,264 |
|com-orkut | 3,072,441 | 117,185,083 | 627,584,181 |

All datasets can be downloaded from [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/). And for synthesised datasets, we use [NetworkX](https://networkx.org/) to generate graphs following power law which can be found in `tc/script/powerlaw_graph.py`.