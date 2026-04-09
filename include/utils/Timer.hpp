/**
 * @file Timer.hpp
 * @brief Declares a lightweight chrono-based execution timer for performance measurements.
 */

#pragma once
#include <chrono>

/// @brief High-resolution execution timer for performance measurements.
///
/// Provides microsecond-precision timing using std::chrono::high_resolution_clock.
/// Useful for profiling algorithm stages. Timer starts automatically in constructor.
class ExecutionTimer {
public:
    /// @brief Construct timer and immediately start measurement.
    ExecutionTimer();
    
    /// @brief Restart the timer (reset start point to now).
    void start();
    
    /// @brief Get elapsed time since construction or last start() call.
    /// @return Elapsed time in milliseconds (double precision)
    double getElapsedMilliseconds() const;
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};
