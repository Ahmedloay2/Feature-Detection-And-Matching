/**
 * @file grayscale.hpp
 * @brief Declares color-to-grayscale conversion using ITU-R BT.601 weighting.
 */

#pragma once
#include "model/image.hpp"

/// @brief Convert RGB or BGR image to grayscale using ITU-R BT.601 weighting.
///
/// Uses the ITU-R BT.601 standard formula:
/// $$Gray = 0.299 \\cdot R + 0.587 \\cdot G + 0.114 \\cdot B$$
///
/// If the image is already grayscale, it is copied without modification.
/// This is the first preprocessing stage for all feature detectors.
///
/// @param img The image to process. Reads img.cache["loaded"] or img.mat (original image).
///            Stores img.cache["grayscale"] (CV_8UC1 or CV_32FC1)
/// @throws std::runtime_error if the image hasn't been loaded yet
void convertToGrayscale(Image& img);