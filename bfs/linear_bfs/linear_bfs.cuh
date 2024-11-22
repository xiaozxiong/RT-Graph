#pragma once

double LinearBFS(const int *adjs, const int *offsets, int nodes, int edges, int chunk_length,
                 int source_node = 0, bool filter = true);
