/**
 * @file image_handler.hpp
 * @brief Image loading and I/O operations.
 *
 * This module provides functionality for loading image files from disk
 * and initializing them as Image objects for processing.
 */

#pragma once
#include "model/image.hpp"
#include <string>

/**
 * @brief Load an image from disk into an Image object.
 *
 * Reads an image file from the specified path using OpenCV's imread function.
 * The image is loaded in BGR color format (CV_8UC3) if it's a color image.
 *
 * @param path The file path to the image to load
 * @return Image object containing the loaded image matrix
 * @throws std::runtime_error if the image file cannot be read or is corrupt
 */
Image loadImage(const std::string& path);
