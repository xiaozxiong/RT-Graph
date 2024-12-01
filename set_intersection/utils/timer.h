#pragma once

#include <chrono>
#include <cuda_runtime.h>

struct CPUTimer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop;
    void StartTiming() { start = std::chrono::high_resolution_clock::now(); }
    void StopTiming() { stop = std::chrono::high_resolution_clock::now(); }
    // ms
    double GetElapsedTime() {
        std::chrono::duration<double> duration = stop - start;
        return duration.count() * 1000.0;
    }
};

struct GPUTimer {
    cudaEvent_t start, stop;
    float elapsed_time;
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        elapsed_time = 0.0f;
    }
    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void StartTiming() { cudaEventRecord(start); }
    void StopTiming() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }
    float GetElapsedTime() {
        cudaEventElapsedTime(&elapsed_time, start, stop);
        return elapsed_time;
    }
};
