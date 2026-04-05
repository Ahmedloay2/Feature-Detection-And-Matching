/**
 * @file hystersis.hpp
 * @brief Hysteresis edge tracking and thresholding.
 *
 * This module implements hysteresis thresholding and edge tracking, which is the
 * final stage of Canny edge detection. It produces a binary edge map.
 */

#pragma once
#include "model/image.hpp"

/**
 * @brief Apply hysteresis thresholding and edge tracking to produce binary edge map.
 *
 * Uses two thresholds and edge connectivity:
 * 1. Pixels above highThreshold are strong edges (255)
 * 2. Pixels between lowThreshold and highThreshold are weak edges (initially 128)
 * 3. Pixels below lowThreshold are non-edges (0)
 * 4. Weak edges connected to strong edges through 8-connectivity are promoted to edges
 * 5. Remaining weak edges are suppressed to 0
 *
 * This process eliminates spurious edges while preserving edges connected to
 * strong responses, resulting in more robust edge detection.
 *
 * @param img The image to process. Reads img.cache["nms"],
 *            stores binary edge map to img.cache["edges"] (CV_8UC1)
 * @param lowThreshold Lower threshold - weak edge candidate threshold
 * @param highThreshold Upper threshold - strong edge threshold
 */
void applyCornerThreshold(Image& img, const std::string& responseKey, float threshold);