/**
 * @file utils.hpp
 * @brief Utility functions for image processing operations.
 *
 * Architecture
 * ────────────
 * Template bodies live in utils_impl.hpp (included at the bottom of
 * this file).  Non-template definitions live in utils.cpp.
 *
 * Explicit instantiations for <float> and <double> are provided in
 * utils.cpp so the linker can find them without re-instantiating in
 * every translation unit.  If you need a third type, add an explicit
 * instantiation there.
 *
 * Performance notes
 * ─────────────────
 * Convolution uses border-padding + cv::parallel_for_:
 *
 *   1. Pad the source image with reflected borders (copyMakeBorder).
 *   2. Split the row range across threads with cv::parallel_for_.
 *   3. Run the inner kernel loop with NO boundary checks.
 *
 * This gives us:
 *   - Zero branch overhead per pixel (padding guarantees safe access)
 *   - Compiler-visible straight multiply-accumulate → SIMD (SSE/AVX)
 *     via -O3 -march=native
 *   - Multi-core parallelism via OpenCV's built-in thread pool
 *     (backed by TBB or OpenMP depending on your OpenCV build)
 *
 * convolveV additionally transposes the input so the vertical pass
 * becomes a horizontal pass over contiguous memory rows — eliminating
 * the strided column-access pattern that defeats the CPU cache.
 */

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace utils {

    /**
     * @brief Map an out-of-bounds index into [0, size-1] by reflection.
     *
     * Still available for callers that need it outside convolution.
     *
     * @param x    Index to map (may be negative or >= size)
     * @param size Valid range is [0, size-1]
     * @return     Reflected index in [0, size-1]
     */
    int reflectIndex(int x, int size);

    /**
     * @brief 1-D horizontal convolution, parallelised across rows.
     *
     * @tparam T   float or double  (explicit instantiations in utils.cpp)
     * @param src    Input CV_32FC1 / CV_64FC1 matrix
     * @param kernel 1-D kernel (odd length recommended)
     * @return       Output matrix, same size and type as src
     */
    template<typename T>
    cv::Mat convolveH(const cv::Mat& src, const std::vector<T>& kernel);

    /**
     * @brief 1-D vertical convolution, parallelised across rows.
     *
     * Internally transposes → convolveH → transposes back so that both
     * passes benefit from cache-friendly sequential memory access.
     *
     * @tparam T   float or double  (explicit instantiations in utils.cpp)
     * @param src    Input CV_32FC1 / CV_64FC1 matrix
     * @param kernel 1-D kernel (odd length recommended)
     * @return       Output matrix, same size and type as src
     */
    template<typename T>
    cv::Mat convolveV(const cv::Mat& src, const std::vector<T>& kernel);

} // namespace utils

// Template bodies — included here so that every TU that includes
// utils.hpp can still instantiate the templates if needed, while the
// explicit instantiations in utils.cpp avoid duplicate code-gen for
// the common float/double cases.
#include "utils/utils_impl.hpp"