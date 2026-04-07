/**
 * @file sift_pyramid.cpp
 * @brief Implements Gaussian and Difference-of-Gaussians pyramid construction for scale-space.
 */

#include "SiftCore.hpp"

#include <opencv2/imgproc.hpp>
#include <omp.h>

namespace cv_assign
{
/// Builds a Gaussian pyramid with multiple octaves and scales.
/// Each octave is a factor of 2 reduction in resolution. Within each octave,
/// numScales layers are created with progressively increasing sigma values
/// (k = 2^(1/numScales)). This creates the scale-space representation needed
/// to detect keypoints at multiple scales.
void SiftProcessor::buildGaussianPyramid(const cv::Mat& baseImage,
                                         std::vector<std::vector<cv::Mat>>& gaussPyramid,
                                         int numOctaves,
                                         int numScales)
{
    gaussPyramid.clear();
    gaussPyramid.resize(numOctaves);

    const float k = std::pow(2.0f, 1.0f / static_cast<float>(numScales));

    for (int oct = 0; oct < numOctaves; ++oct) {
        cv::Mat octBase;
        if (oct == 0) {
            octBase = baseImage;
        } else {
            if (gaussPyramid[oct - 1].size() <= static_cast<size_t>(numScales)) break;
            const cv::Mat& prev = gaussPyramid[oct - 1][numScales];
            if (prev.empty() || prev.cols <= 4 || prev.rows <= 4) break;
            cv::resize(prev, octBase, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        }

        gaussPyramid[oct].resize(numScales + 3);
        const float baseSigma = (oct == 0) ? 0.0f : SIGMA;

#pragma omp parallel for
        for (int s = 0; s < numScales + 3; ++s) {
            const float totalSigma = SIGMA * std::pow(k, static_cast<float>(s));
            const float applySigma = std::sqrt(std::max(0.0f, totalSigma * totalSigma - baseSigma * baseSigma));
            if (applySigma < 0.01f) gaussPyramid[oct][s] = octBase.clone();
            else cv::GaussianBlur(octBase, gaussPyramid[oct][s], cv::Size(0, 0), applySigma);
        }
    }
}

void SiftProcessor::buildDoGPyramid(const std::vector<std::vector<cv::Mat>>& gaussPyramid,
                                    std::vector<std::vector<cv::Mat>>& dogPyramid)
{
    dogPyramid.clear();
    dogPyramid.resize(gaussPyramid.size());

    for (size_t oct = 0; oct < gaussPyramid.size(); ++oct) {
        const int n = static_cast<int>(gaussPyramid[oct].size());
        if (n < 2) continue;

        dogPyramid[oct].resize(n - 1);
#pragma omp parallel for
        for (int s = 0; s < n - 1; ++s) {
            cv::subtract(gaussPyramid[oct][s + 1], gaussPyramid[oct][s], dogPyramid[oct][s]);
        }
    }
}
}
