/**
 * @file grayscale.hpp
 * @brief Grayscale color space conversion for Canny edge detection.
 *
 * This module handles the conversion of color images to grayscale using
 * the standard luminance formula for proper color weighting.
 */

#pragma once
#include "model/image.hpp"

/**
 * @brief Convert an image to grayscale using the standard luminance formula.
 *
 * Converts the input image from color (BGR, CV_8UC3) or grayscale (CV_8UC1)
 * to grayscale format using the ITU-R BT.601 luma coefficient formula:
 * gray = 0.299*R + 0.587*G + 0.114*B
 *
 * If the input is already grayscale, it is stored as-is.
 * Unsupported channel counts (other than 1 or 3) throw an exception.
 *
 * @param img The image to convert. img.mat is read, result stored in img.cache["grayscale"]
 * @throws std::runtime_error if the input image has an unsupported number of channels
 */
void toGrayscale(Image& img);