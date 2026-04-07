/**
 * @file threshold.hpp
 * @brief Declares thresholding of corner response maps to binary corner candidates.
 */

#pragma once
#include "model/image.hpp"

/// @brief Apply binary threshold to corner response map and extract corner locations.
///
/// Converts the response map to binary (corners vs non-corners) by comparing
/// each pixel against a threshold value. Pixels with response >= threshold
/// are marked as corner candidates. Corner pixel locations are extracted as
/// a vector of (x, y) coordinates.
///
/// @param img The image to process. Reads img.cache["nms_result"] or 
///            img.cache["harris_response"] / img.cache["shi_tomasi_response"].
///            Stores img.cache["threshold_result"] (CV_8UC1, binary image).
/// @param responseKey Key in img.cache for input response map
/// @param threshold Minimum response value for a pixel to be considered a corner
/// @return Vector of corner pixel locations (cv::Point)
/// @throws std::runtime_error if response stage hasn't been computed yet
std::vector<cv::Point> applyCornerThreshold(Image& img, const std::string& responseKey, float threshold);