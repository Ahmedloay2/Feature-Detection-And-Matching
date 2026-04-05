/**
 * @file utils.hpp
 * @brief Utility functions for image processing operations.
 *
 * Performance notes
 * ─────────────────
 * Convolution is implemented using border-padding:
 *   1. Pad the source image with reflected borders (copyMakeBorder).
 *   2. Run the convolution inner loop with NO boundary checks.
 *
 * This eliminates:
 *   - reflectIndex() call per kernel element per pixel (~12M calls saved)
 *   - Indirect/non-sequential memory access pattern
 *   - Conditional branches that block compiler auto-vectorization
 *
 * The inner loop becomes a straight multiply-accumulate over contiguous
 * memory — the compiler can emit SIMD (SSE/AVX) instructions for it.
 */

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>   // cv::copyMakeBorder
#include <vector>

namespace utils {

    /**
     * @brief Map out-of-bounds indices using reflection.
     * Still available for use outside convolution if needed.
     */
    int reflectIndex(int x, int size);

    /**
     * @brief Perform 1D horizontal convolution on each row.
     *
     * Pads the image horizontally with reflected borders equal to half
     * the kernel size, then runs a clean inner loop with no boundary checks.
     *
     * @tparam T  float or double
     * @param src    Input CV_32FC1 (or CV_64FC1) matrix
     * @param kernel 1D kernel, odd length
     * @return Output same size and type as src
     */
    template<typename T>
    cv::Mat convolveH(const cv::Mat& src, const std::vector<T>& kernel)
    {
        const int rows = src.rows;
        const int cols = src.cols;
        const int half = static_cast<int>(kernel.size()) / 2;
        const int klen = static_cast<int>(kernel.size());

        // ── Pad left and right with reflected border ──────────────────────────
        // BORDER_REFLECT_101: reflects without duplicating the edge pixel
        // e.g. padding of 1: [a b c d] → [b | a b c d | c]
        cv::Mat padded;
        cv::copyMakeBorder(src, padded,
            0, 0,          // top, bottom — no vertical padding here
            half, half,    // left, right padding
            cv::BORDER_REFLECT_101);

        cv::Mat dst(rows, cols, src.type());

        for (int i = 0; i < rows; ++i)
        {
            // padded row starts at column 0; src pixel j maps to padded column j+half
            const T* paddedRow = padded.ptr<T>(i);
            T* dstRow = dst.ptr<T>(i);

            for (int j = 0; j < cols; ++j)
            {
                T val = static_cast<T>(0);

                // paddedRow + j is the start of the kernel window for pixel j
                // No boundary check needed — padding guarantees valid access
                const T* window = paddedRow + j;   // window[0..klen-1] all valid

                for (int n = 0; n < klen; ++n)
                    val += window[n] * kernel[n];

                dstRow[j] = val;
            }
        }
        return dst;
    }

    /**
     * @brief Perform 1D vertical convolution on each column.
     *
     * Pads the image vertically with reflected borders, then runs a
     * clean inner loop using row pointers — no boundary checks.
     *
     * @tparam T  float or double
     * @param src    Input CV_32FC1 (or CV_64FC1) matrix
     * @param kernel 1D kernel, odd length
     * @return Output same size and type as src
     */
    template<typename T>
    cv::Mat convolveV(const cv::Mat& src, const std::vector<T>& kernel)
    {
        const int rows = src.rows;
        const int cols = src.cols;
        const int half = static_cast<int>(kernel.size()) / 2;
        const int klen = static_cast<int>(kernel.size());

        // ── Pad top and bottom with reflected border ──────────────────────────
        cv::Mat padded;
        cv::copyMakeBorder(src, padded,
            half, half,    // top, bottom padding
            0, 0,          // left, right — no horizontal padding
            cv::BORDER_REFLECT_101);

        cv::Mat dst(rows, cols, src.type());

        // Pre-cache all row pointers into an array for fast access in inner loop
        // Avoids repeated ptr<T>(row) virtual dispatch overhead
        std::vector<const T*> rowPtrs(rows + 2 * half);
        for (int i = 0; i < rows + 2 * half; ++i)
            rowPtrs[i] = padded.ptr<T>(i);

        for (int i = 0; i < rows; ++i)
        {
            T* dstRow = dst.ptr<T>(i);

            for (int j = 0; j < cols; ++j)
            {
                T val = static_cast<T>(0);

                // rowPtrs[i] corresponds to padded row i (which is src row i-half).
                // The kernel window for src pixel (i,j) spans padded rows i..i+klen-1.
                for (int m = 0; m < klen; ++m)
                    val += rowPtrs[i + m][j] * kernel[m];

                dstRow[j] = val;
            }
        }
        return dst;
    }

} // namespace utils