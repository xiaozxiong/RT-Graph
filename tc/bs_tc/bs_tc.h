#pragma once

#define METHOD 0

void CountOnGPU(int *adjs, int *offsets, int nodes, int edges, int device_id = 0);