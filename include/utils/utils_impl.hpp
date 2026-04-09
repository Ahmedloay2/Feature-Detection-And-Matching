/**
 * @file utils_impl.hpp
 * @brief Provides template implementations for separable convolution operations (convolveH/convolveV).
 *
 * ## Implementation Details
 *
 * **Optimization Strategy:**
 * - Separable convolution: Use convolveH then convolveV (or vice versa) for O(k) vs O(k²) complexity
 * - Border handling: BORDER_REFLECT_101 avoids boundary checks inside inner loops
 * - Vectorization hints: pragma GCC ivdep declares no loop-carried dependencies for auto-SIMD
 * - Thread parallelism: cv::parallel_for_ (TBB/OpenMP) parallelizes over rows
 * - Cache locality: Transposed vertical convolution for sequential memory access
 *
 * **Template Parameters:**
 * Both functions are templates accepting float or double; explicit instantiations
 * are in utils.cpp to avoid linker issues.
 */

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace utils {

// Horizontal 1-D convolution
template<typename T>
cv::Mat convolveH(const cv::Mat& src, const std::vector<T>& kernel)
{
    const int rows = src.rows;
    const int cols = src.cols;
    const int half = static_cast<int>(kernel.size()) / 2;
    const int klen = static_cast<int>(kernel.size());

    // ── 1. Pad left/right with reflected border ───────────────────────────
    // BORDER_REFLECT_101 mirrors without repeating the edge pixel:
    //   [a b c d]  →  [b | a b c d | c]   (for half=1)
    // After padding, pixel j in the original maps to column j+half in
    // `padded`, and the kernel window [j .. j+klen-1] is always valid.
    cv::Mat padded;
    cv::copyMakeBorder(src, padded,
        0, 0, half, half,
        cv::BORDER_REFLECT_101);

    cv::Mat dst(rows, cols, src.type());

    // 2. Resolve kernel pointer before threads
    // Avoids std::vector indirection inside the hot loop and gives the
    // compiler a plain pointer it can reason about for vectorisation.
    const T* kptr = kernel.data();

    // ── 3. Parallelise the outer (row) loop with OpenCV's thread pool ─────
    // cv::parallel_for_ uses TBB, OpenMP, or pthreads depending on how
    // OpenCV was built.  The lambda receives a sub-range of rows; each
    // sub-range is processed independently (no shared writes).
    cv::parallel_for_(cv::Range(0, rows), [&](const cv::Range& range)
    {
        for (int i = range.start; i < range.end; ++i)
        {
            const T* paddedRow = padded.ptr<T>(i);
            T*       dstRow    = dst.ptr<T>(i);

            for (int j = 0; j < cols; ++j)
            {
                T val = static_cast<T>(0);

                // window[0..klen-1] is always valid — padding guarantees it.
                // No boundary check needed here.
                const T* window = paddedRow + j;

                // 4. Multiply-accumulate loop──────────
                // #pragma GCC ivdep asserts no loop-carried data dependency,
                // enabling auto-vectorisation to SSE/AVX on -O3 -march=native.
#pragma GCC ivdep
                for (int n = 0; n < klen; ++n)
                    val += window[n] * kptr[n];

                dstRow[j] = val;
            }
        }
    });

    return dst;
}

// Vertical 1-D convolution via transpose + convolveH
template<typename T>
cv::Mat convolveV(const cv::Mat& src, const std::vector<T>& kernel)
{
    // 5. Transpose trick─────────────────────────────────────
    // Naïve vertical convolution reads one pixel per row step — each read
    // jumps `cols * sizeof(T)` bytes, thrashing the CPU cache.
    //
    // Fix: transpose so that every original column becomes a contiguous
    // row, run the same cache-friendly horizontal pass, then transpose
    // back.  cv::transpose uses a cache-oblivious tiled algorithm, so
    // the two extra transposes are cheap relative to the cache savings
    // for any image wider than a few cache lines (~64 B).
    cv::Mat transposed;
    cv::transpose(src, transposed);                    // rows ↔ cols

    cv::Mat resultT = convolveH<T>(transposed, kernel); // horizontal pass
                                                        // on transposed data
                                                        // = vertical pass on
                                                        // original

    cv::Mat dst;
    cv::transpose(resultT, dst);                       // rotate back
    return dst;
}

} // namespace utils
