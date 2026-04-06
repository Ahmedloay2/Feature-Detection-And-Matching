/**
 * @file gaussian.cpp
 * @brief Implementation of Gaussian blur filtering.
 */

#include "../../../include/model/image.hpp"
#include "../../../include/processors/harris/gaussian.hpp"
#include <opencv2/core/mat.hpp>
#include "../../../include/utils/utils.hpp"

 /**
  * @brief Apply separable Gaussian blur to the grayscale image.
  *
  * Uses a 5-tap 1D Gaussian kernel [1/16, 4/16, 6/16, 4/16, 1/16] applied
  * separably (horizontal then vertical), matching OpenCV's approach of
  * smoothing before gradient computation.
  *
  * @param img  Reads img.cache["grayscale"], stores result to img.cache["gaussian"]
  * @throws std::runtime_error if grayscale stage hasn't been computed
  */
void applyGaussian(Image& img) {
    if (!img.has("grayscale"))
        throw std::runtime_error("Grayscale image not found. Run grayscale processor first.");

    // Read the grayscale mat (CV_32F, produced by toGrayscale)
    cv::Mat src = img.get("grayscale");

    // 5-tap Gaussian kernel — stays in float to match the rest of the pipeline
    const std::vector<float> g1d = { 0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f };

    // Separable 2-D Gaussian: horizontal then vertical
    cv::Mat gaussian = utils::convolveV<float>(
        utils::convolveH<float>(src, g1d), g1d);

    img.store("gaussian", gaussian);
}