/**
 * @file utils.hpp
 * @brief Declares reusable image processing utilities: convolution, pooling, and normalization.
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