/**
 * @file utils.cpp
 * @brief Non-template implementations + explicit template instantiations.
 *
 * Explicit instantiations
 * ────────────────────────
 * The two lines at the bottom of this file tell the compiler to emit
 * fully-compiled object code for convolveH<float>, convolveH<double>,
 * convolveV<float>, and convolveV<double> exactly once — here, in this
 * translation unit.
 *
 * Without them the compiler would re-instantiate (and re-compile) the
 * template body in every .cpp that includes utils.hpp, bloating compile
 * times and object sizes.  With them, every other TU uses the already-
 * compiled versions from this object file, exactly like a normal
 * function call.
 *
 * If you later need a third type (e.g. cv::float16_t), add:
 *   template cv::Mat utils::convolveH<cv::float16_t>(const cv::Mat&,
 *                                    const std::vector<cv::float16_t>&);
 *   template cv::Mat utils::convolveV<cv::float16_t>(const cv::Mat&,
 *                                    const std::vector<cv::float16_t>&);
 */

#include "utils/utils.hpp"   // pulls in utils_impl.hpp automatically

namespace utils {

    /**
     * @brief Map an out-of-bounds index into [0, size-1] by reflection.
     *
     * Reflection rules (size = N, valid range [0, N-1]):
     *   x = -1  → 1        (reflect left of 0)
     *   x = -2  → 2
     *   x = N   → N-2      (reflect right of N-1)
     *   x = N+1 → N-3
     *
     * Example with N=4:
     *   input : ... -2 -1  0  1  2  3  4  5 ...
     *   output: ...  2  1  0  1  2  3  2  1 ...
     *
     * @param x    Index to map (may be negative or >= size)
     * @param size Valid range is [0, size-1]
     * @return     Reflected index in [0, size-1]
     */
    int reflectIndex(int x, int size)
    {
        if (x < 0)
            return -x;                  // -1 → 1, -2 → 2, …

        if (x >= size)
            return 2 * (size - 1) - x;  // N → N-2, N+1 → N-3, …

        return x;
    }

} // namespace utils

// ── Explicit template instantiations ─────────────────────────────────────────
// These must appear at namespace scope, after the template bodies are visible
// (i.e. after the #include "utils/utils.hpp" above which pulls in
// utils_impl.hpp).
//
// The `extern template` declarations in every other TU that includes
// utils.hpp suppress re-instantiation there, so the linker finds exactly
// one copy of each specialisation.

template cv::Mat utils::convolveH<float>(const cv::Mat&, const std::vector<float>&);
template cv::Mat utils::convolveH<double>(const cv::Mat&, const std::vector<double>&);

template cv::Mat utils::convolveV<float>(const cv::Mat&, const std::vector<float>&);
template cv::Mat utils::convolveV<double>(const cv::Mat&, const std::vector<double>&);