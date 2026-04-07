/**
 * @file Timer.cpp
 * @brief Implements high-resolution timer methods for performance profiling.
 */

#include "Timer.hpp"

/// Constructor: Initialize timer and start measurement immediately.
ExecutionTimer::ExecutionTimer() {
    start();
}

/// Restart the timer by recording the current time as the new start point.
void ExecutionTimer::start() {
    startTime = std::chrono::high_resolution_clock::now();
}

/// Get elapsed time in milliseconds since construction or last start() call.
/// Uses high_resolution_clock for microsecond-scale precision.
double ExecutionTimer::getElapsedMilliseconds() const {
    auto endTime = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(endTime - startTime).count();
}
