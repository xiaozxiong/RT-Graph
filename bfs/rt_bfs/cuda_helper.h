#pragma once

double GetOriginsByNodes(int nodes, int *d_node_list, int *d_origin_offset, int *d_origin_num,
                         int *d_origin_list, bool filter = true);

void ThrustFill(int *array, int size, int val = 0);