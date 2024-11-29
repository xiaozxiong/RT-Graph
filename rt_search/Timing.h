#pragma once
#include <chrono>
// ms
struct Timer{
    std::chrono::time_point<std::chrono::high_resolution_clock> start,end;
    std::chrono::duration<double> duration;
    void StartTiming(){
        start=std::chrono::high_resolution_clock::now();
    }
    void EndTiming(){
        end=std::chrono::high_resolution_clock::now();
        duration=end-start;
    }
    double GetTime(){
        return duration.count()*1000.0;
    }
};  