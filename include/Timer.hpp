#pragma once
#include <chrono>

class ExecutionTimer {
public:
    ExecutionTimer();
    void start();
    double getElapsedMilliseconds() const;
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};
