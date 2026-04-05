#include "Timer.hpp"

ExecutionTimer::ExecutionTimer() {
    start();
}

void ExecutionTimer::start() {
    startTime = std::chrono::high_resolution_clock::now();
}

double ExecutionTimer::getElapsedMilliseconds() const {
    auto endTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(endTime - startTime).count();
}
