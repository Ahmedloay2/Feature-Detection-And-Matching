/**
 * @file utils_impl.hpp
 * @brief Template bodies for convolveH / convolveV.
 *
 * Do NOT include this file directly — it is pulled in at the bottom of
 * utils.hpp automatically.
 *
 * Optimisation guide for this file
 * ─────────────────────────────────
 * 1. cv::parallel_for_  — splits the outer row loop across all CPU
 *    cores using OpenCV's thread pool (TBB / OpenMP / pthreads,
 *    depending on the OpenCV build).  Rows are independent so this is
 *    embarrassingly parallel.
 *
 * 2. Raw kernel pointer  — kernel.data() is resolved once before the
 *    parallel region, removing the std::vector overhead from the inner
 *    loop and making the memory layout visible to the auto-vectoriser.
 *
 * 3. #pragma GCC ivdep  — tells GCC/Clang "there are no loop-carried
 *    dependencies here; please vectorise".  Combined with -O3
 *    -march=native this lets the compiler emit SSE/AVX
 *    multiply-accumulate instructions for the inner loop.
 *
 * 4. Transpose trick in convolveV  — vertical convolution reads one
 *    element per row (strided access, terrible for cache).  Transposing
 *    first turns every column into a row, so the same convolveH kernel
 *    loop runs over contiguous memory.  cv::transpose is highly
 *    optimised (cache-oblivious tiled algorithm) so the round-trip
 *    transpose cost is small compared with the cache-miss savings.
 */

#pragma once
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace utils {

// ─────────────────────────────────────────────────────────────────────────────
// convolveH — horizontal 1-D convolution
// ─────────────────────────────────────────────────────────────────────────────

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

    // ── 2. Resolve the kernel pointer ONCE before entering threads ────────
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

                // ── 4. Inner multiply-accumulate loop ─────────────────────
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

// ─────────────────────────────────────────────────────────────────────────────
// convolveV — vertical 1-D convolution via transpose + convolveH
// ─────────────────────────────────────────────────────────────────────────────

template<typename T>
cv::Mat convolveV(const cv::Mat& src, const std::vector<T>& kernel)
{
    // ── 5. Transpose trick ────────────────────────────────────────────────
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
