/**
 * @file gradient.cpp
 * @brief Implementation of gradient computation using Sobel operators.
 */

#include "../../../include/processors/harris/gradient.hpp"
#include "../../../include/model/image.hpp"
#include "../../../include/utils/utils.hpp"
#include <cmath>
#include <opencv2/core.hpp>

/**
 * @brief Compute image gradient magnitude and direction using Sobel operators.
 *
 * Applies separable Sobel operators to compute two orthogonal gradients:
 * - Gx: Horizontal gradient using difference kernel [-1, 0, 1]
 * - Gy: Vertical gradient using difference kernel [-1, 0, 1]
 *
 * For each pixel, calculates:
 * - magnitude = sqrt(Gx^2 + Gy^2) - Represents edge strength
 * - angle = atan2(Gy, Gx) - Represents edge direction in radians
 *
 * This information is used for non-maximum suppression to thin edges.
 *
 * @param img The image to process. Reads img.cache["gaussian"],
 *            stores img.cache["gradient_magnitude"] and img.cache["gradient_angle"]
 * @throws std::runtime_error if Gaussian stage hasn't been computed
 */
void computeGradient(Image& img) {
    // Validate that Gaussian blur has been applied
    if (!img.has("grayscale"))
        throw std::runtime_error("grayscale stage not computed yet.");

    // Convert Gaussian image to float for gradient computation
    cv::Mat src = img.get("grayscale");

    // Define Sobel operator kernels
    // smooth: [1, 2, 1] - Smoothing to reduce noise
    // diff: [-1, 0, 1] - Differentiation to compute gradient
    const std::vector<float> smooth = { 1.f, 2.f,  1.f };
    const std::vector<float> diff = { -1.f, 0.f,  1.f };

    // Compute horizontal gradient (Gx)
    // Apply smoothing vertically, then differentiation horizontally
    cv::Mat Gx = utils::convolveH<float>(utils::convolveV<float>(src, smooth), diff);

    // Compute vertical gradient (Gy)
    // Apply smoothing horizontally, then differentiation vertically
    cv::Mat Gy = utils::convolveH<float>(utils::convolveV<float>(src, diff), smooth);

    img.store("gradient_gx", Gx);
    img.store("gradient_gy", Gy);

    cv::Mat Ix2,Iy2,IxIy;

    cv::multiply(Gx, Gx, Ix2); 
    cv::multiply(Gy, Gy, Iy2);
    cv::multiply(Gx, Gy, IxIy);

    img.store("gradient_xx", Ix2);
    img.store("gradient_yy", Iy2);
    img.store("gradient_xy", IxIy);
}