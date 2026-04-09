/**
 * @file harris_main.hpp
 * @brief Declares the orchestrating entry point for Harris and Shi-Tomasi pipelines.
 */

#pragma once
#include <vector>
#include <opencv2/core/types.hpp>
#include <model/image.hpp>

/// @brief Execute complete Harris or Shi-Tomasi corner detection pipeline.
///
/// Orchestrates all processing stages in sequence:
/// 1. Grayscale conversion
/// 2. Gaussian smoothing
/// 3. Gradient computation (Sobel)
/// 4. Structure tensor (first moment matrix)
/// 5. Harris or Shi-Tomasi response computation
/// 6. Non-maximum suppression
/// 7. Thresholding to extract corner locations
///
/// @param image Input/output image. On input, contains the loaded original image;
///              on output, caches all intermediate results.
/// @param k Harris parameter (not used if mode="shi_tomasi")
/// @param mode "harris" for Harris detector, "shi_tomasi" for Shi-Tomasi
/// @param threshold Minimum response value for corner candidates
/// @param halfwindow NMS window half-width (total window = 2*half+1 x 2*half+1)
/// @return Vector of detected corner pixel locations
/// @throws std::runtime_error if image is invalid or processing fails
std::vector<cv::Point> applyHarris(Image& image, float k, const std::string& mode, float threshold, int halfwindow);