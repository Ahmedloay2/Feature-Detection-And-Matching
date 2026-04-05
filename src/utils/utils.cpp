/**
 * @file utils.cpp
 * @brief Implementation of utility functions for image processing.
 */

#include "utils/utils.hpp"

namespace utils {

    /**
     * @brief Map out-of-bounds indices to valid range using reflection.
     *
     * Implements reflection boundary handling for convolution:
     * - Indices < 0 are reflected: reflectIndex(-1, size) = 1, reflectIndex(-2, size) = 2
     * - Indices >= size are reflected: reflectIndex(size, size) = size-2, etc.
     * - Valid indices [0, size-1] are returned unchanged
     *
     * This approach avoids artifacts at image boundaries during convolution.
     * For example, with size=4: indices [..., -2, -1, 0, 1, 2, 3, 4, 5, ...]
     * map to [2, 1, 0, 1, 2, 3, 2, 1, ...]
     *
     * @param x The index to map (can be negative or >= size)
     * @param size The valid range is [0, size-1]
     * @return Mapped index in range [0, size-1]
     */
    int reflectIndex(int x, int size) {
        // Handle negative indices: reflect left boundary
        if (x < 0)
            return -x;  // -1 maps to 1, -2 maps to 2, etc.

        // Handle indices >= size: reflect right boundary
        if (x >= size)
            return 2 * (size - 1) - x;  // size maps to size-2, size+1 maps to size-3, etc.

        // Valid index: return unchanged
        return x;
    }

} // namespace utils