/**
 * @file gaussian.hpp
 * @brief Gaussian blur filtering for noise reduction.
 *
 * This module applies a Gaussian blur filter to smooth the image and reduce noise,
 * which is the second stage of the Canny edge detection algorithm.
 */

#pragma once
#include "../../../include/model/image.hpp"

/**
 * @brief Apply Gaussian blur to smooth the grayscale image.
 *
 * Applies a separable 1D Gaussian kernel [0.25, 0.5, 0.25] both horizontally
 * and vertically to the grayscale image. This reduces noise and smooths details
 * before gradient computation.
 *
 * @param img The image to blur. Reads img.cache["grayscale"], writes to img.cache["gaussian"]
 * @throws std::runtime_error if grayscale stage has not been computed yet
 */
void applyGaussian(Image& img);