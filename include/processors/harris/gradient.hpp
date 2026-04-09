/**
 * @file gradient.hpp
 * @brief Declares Sobel-based image gradient computation (magnitude and direction).
 */

#pragma once
#include "model/image.hpp"

/**
 * @brief Compute image gradient magnitude and direction.
 *
 * Applies separable Sobel operators to compute gradient components:
 * - Gx: horizontal gradient (using difference kernel [-1, 0, 1])
 * - Gy: vertical gradient (using difference kernel [-1, 0, 1])
 *
 * From these components, the magnitude and direction are calculated:
 * - magnitude = sqrt(Gx^2 + Gy^2)
 * - angle = atan2(Gy, Gx)
 *
 * @param img The image to process. Reads img.cache["gaussian"], stores:
 *            - img.cache["gradient_magnitude"] (CV_32FC1): Gradient magnitude
 *            - img.cache["gradient_angle"] (CV_32FC1): Gradient direction in radians
 * @throws std::runtime_error if Gaussian stage has not been computed yet
 */
void computeGradient(Image& img);