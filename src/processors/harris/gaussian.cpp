/**
 * @file gaussian.cpp
 * @brief Implements Gaussian smoothing using separable convolution filters.
 */

#include "../../../include/model/image.hpp"
#include "../../../include/processors/harris/gaussian.hpp"
#include <opencv2/core/mat.hpp>
#include "../../../include/utils/utils.hpp"

 /// Applies separable 5-tap Gaussian blur to smooth the grayscale image.
/// Uses kernel [1/16, 4/16, 6/16, 4/16, 1/16] applied horizontally then vertically.
/// Separable convolution reduces complexity from O(n*k²) to O(n*k).
void applyGaussianBlur(Image& img, float sigma) {
    if (!img.has("grayscale"))
        throw std::runtime_error("Grayscale image not found. Run grayscale processor first.");

    // Read the grayscale mat (CV_32F, produced by convertToGrayscale)
    cv::Mat src = img.get("grayscale");

    // 5-tap Gaussian kernel — stays in float to match the rest of the pipeline
    const std::vector<float> g1d = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };

    // Separable 2-D Gaussian: horizontal then vertical
    cv::Mat gaussian = utils::convolveV<float>(
        utils::convolveH<float>(src, g1d), g1d);

    img.store("gaussian", gaussian);
}