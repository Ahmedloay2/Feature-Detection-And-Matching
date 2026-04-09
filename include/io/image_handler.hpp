/**
 * @file image_handler.hpp
 * @brief Declares image file loading and I/O operations for the detection pipeline.
 */

#pragma once
#include "model/image.hpp"
#include <string>

/// @brief Load an image file from disk into an Image container.
///
/// Reads an image from the specified file path using OpenCV's imread function.
/// The image is loaded in BGR color space (CV_8UC3). Automatically validates that
/// the file exists and is readable.
///
/// **Supported Formats:** PNG, JPEG, BMP, TIFF, and others supported by OpenCV
///
/// @param path File path to the image (absolute or relative to working directory)
/// @return Image object with loaded mat in BGR format (CV_8UC3)
/// @throws std::runtime_error if the file doesn't exist or cannot be read
Image loadImage(const std::string& path);
