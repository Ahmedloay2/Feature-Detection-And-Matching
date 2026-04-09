/**
 * @file nms.hpp
 * @brief Declares non-maximum suppression to thin corner responses to single pixels.
 */

#pragma once
#include "model/image.hpp"

/// @brief Apply non-maximum suppression (NMS) to corner response maps.
///
/// NMS thins corner responses by suppressing pixels that are not local maxima
/// within a window. For each pixel, if its response is lower than the maximum
/// in the surrounding neighborhood, it is suppressed to zero. This ensures
/// that each corner is represented by a single peak pixel.
///
/// @param img The image to process. Reads img.cache["harris_response"] or
///            img.cache["shi_tomasi_response"].
///            Stores img.cache["nms_result"] (CV_32FC1, binary-like: 0 or original value).
/// @param responseKey Key in img.cache for input response map (e.g., "harris_response")
/// @param halfWindow Half-width of NMS neighborhood window (total window is 2*half+1 x 2*half+1)
/// @throws std::runtime_error if response stage hasn't been computed yet
void applyCornerNMS(Image& img, const std::string& responseKey, int halfWindow);