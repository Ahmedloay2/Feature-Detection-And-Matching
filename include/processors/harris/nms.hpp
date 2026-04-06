/**
 * @file nms.hpp
 * @brief Non-maximum suppression for edge thinning.
 *
 * This module implements non-maximum suppression, which thins edges to a single
 * pixel width by suppressing gradient values that are not local maxima along
 * the gradient direction.
 */

#pragma once
#include "model/image.hpp"

/**
 * @brief Suppress non-maximum gradient values to thin edges.
 *
 * For each pixel, compares its gradient magnitude with two neighbors along
 * the gradient direction (quantized to 4 main directions: horizontal, vertical,
 * and two diagonals). The pixel is kept only if it is a local maximum.
 *
 * Direction thresholds:
 * - 0-22.5°, 157.5-180°: Horizontal (left/right neighbors)
 * - 22.5-67.5°: Diagonal (top-right/bottom-left)
 * - 67.5-112.5°: Vertical (top/bottom neighbors)
 * - 112.5-157.5°: Diagonal (top-left/bottom-right)
 *
 * @param img The image to process. Reads img.cache["gradient_magnitude"] and
 *            img.cache["gradient_angle"], stores to img.cache["nms"]
 */
void applyCornerNMS(Image& img, const std::string& responseKey, int halfWindow);