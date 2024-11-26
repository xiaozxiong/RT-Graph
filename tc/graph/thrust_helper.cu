#include "thrust_helper.h"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <vector>

#define BucketNum 10

void ThrustSortEdges(std::vector<edge_t> &edge_list) {
    thrust::sort(thrust::host, edge_list.begin(), edge_list.end(), thrust::less<edge_t>());
}

void ThrustSortEdgesInBatch(std::vector<edge_t> &edge_list, int nodes) {
    thrust::host_vector<thrust::host_vector<edge_t>> edge_bucket;
    edge_bucket.resize(BucketNum);
    int bucket_step = (nodes + BucketNum - 1) / BucketNum;
    int cnt = 0;
    for (auto &e : edge_list) {
        if (e.src == -1 || e.dst == -1)
            continue;
        int bucket_id = e.src / bucket_step;
        edge_bucket[bucket_id].push_back(e);
        if (e.src == 0 && e.dst == 1)
            cnt += 1;
    }
    for (int i = 0; i < BucketNum; i += 1) {
        thrust::device_vector<edge_t> temp_list(edge_bucket[i].begin(), edge_bucket[i].end());
        thrust::sort(temp_list.begin(), temp_list.end(), thrust::less<edge_t>());
        thrust::copy(temp_list.begin(), temp_list.end(), edge_bucket[i].begin());
    }
    size_t offset = 0;
    for (int i = 0; i < BucketNum; i += 1) {
        size_t edges_size = sizeof(edge_t) * edge_bucket[i].size();
        memcpy(edge_list.data() + offset, edge_bucket[i].data(), edges_size);
        offset += edge_bucket[i].size();
    }
    // for(int i=0;i<10;i+=1) printf("%d - %d, ",edge_list[i].src,edge_list[i].dst);
    // printf("\n");
}