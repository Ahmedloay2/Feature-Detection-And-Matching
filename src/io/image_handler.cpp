/**
 * @file image_handler.cpp
 * @brief Implementation of image loading functionality.
 */

#include "../../include/model/image.hpp"
#include "../../include/io/image_handler.hpp"
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

/**
 * @brief Load an image from the specified file path.
 *
 * Uses OpenCV's imread to load the image in BGR color format (IMREAD_COLOR).
 * Validates that the image was loaded successfully before storing in Image object.
 *
 * @param path File path to the image (absolute or relative)
 * @return Image object with loaded mat in BGR format (CV_8UC3)
 * @throws std::runtime_error if the image cannot be loaded or file doesn't exist
 */
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