/**
 * @file gaussian.hpp
 * @brief Declares Gaussian blur filtering for noise reduction before corner detection.
 */

#pragma once
#include "../../../include/model/image.hpp"

/// @brief Apply Gaussian blur to smooth the image and reduce noise.
///
/// Uses a separable Gaussian kernel with configurable sigma for blur radius.
/// Separable convolution (horizontal then vertical) is O(n*k) instead of O(n*k²),
/// where k is kernel size. This is typically the second preprocessing stage,
/// applied after grayscale conversion.
///
/// @param img The image to process. Reads img.cache["grayscale"].
///            Stores img.cache["gaussian"] (same type as input)
/// @param sigma Gaussian standard deviation (blur radius, typically 1.0-2.0)
/// @throws std::runtime_error if grayscale stage has not been computed yet
void applyGaussianBlur(Image& img, float sigma = 1.6f);