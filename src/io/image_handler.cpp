/**
 * @file image_handler.cpp
 * @brief Implements image file loading, validation, and error handling.
 */

#include "../../include/model/image.hpp"
#include "../../include/io/image_handler.hpp"
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

/// Load an image from disk and wrap it in an Image container.
///
/// This function serves as the entry point for all image I/O in the application.
/// It handles file validation, format detection (automatic via OpenCV), and error
/// reporting. The image is always loaded in BGR color space (3 channels, 8-bit unsigned).
Image loadImage(const std::string& path) {
    // Attempt to load the image in BGR color format
    cv::Mat mat = cv::imread(path, cv::IMREAD_COLOR);

    // Validate that image was loaded successfully
    // mat.empty() returns true if imread failed
    if (mat.empty())
        throw std::runtime_error("Could not load image: " + path);

    // Create Image object and populate with loaded data
    Image img;
    img.mat = mat;   // Store loaded image (CV_8UC3, BGR channel order)
    return img;
}