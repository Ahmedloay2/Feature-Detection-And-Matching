/**
 * @file sift_extrema_orientation.cpp
 * @brief Implements DoG extrema detection, filtering, and dominant orientation assignment.
 */

#include "core/SiftCore.hpp"

#include <omp.h>

namespace cv_assign
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;
}

/// Detects local extrema (minima and maxima) in the Difference-of-Gaussians (DoG) pyramid.
/// For each scale within an octave, examines 3x3x3 neighborhoods across different scales
/// to find pixels that are local extrema. Filters based on contrast threshold to remove
/// low-contrast keypoints. Uses thread-level local storage for efficiency.
void SiftProcessor::detectExtrema(const std::vector<std::vector<cv::Mat>>& dogPyramid,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  float contrastThreshold,
                                  int numOctaves,
                                  int numScales)
{
    keypoints.clear();
    std::vector<std::vector<cv::KeyPoint>> local(omp_get_max_threads());

    for (int oct = 0; oct < numOctaves && oct < static_cast<int>(dogPyramid.size()); ++oct) {
        if (dogPyramid[oct].size() <= 2) continue;

        const int rows = dogPyramid[oct][0].rows;
        const int cols = dogPyramid[oct][0].cols;
        const int scales = static_cast<int>(dogPyramid[oct].size());
        if (rows <= 2 || cols <= 2) continue;

        for (int s = 1; s < scales - 1; ++s) {
#pragma omp parallel for collapse(2)
            for (int r = 1; r < rows - 1; ++r) {
                for (int c = 1; c < cols - 1; ++c) {
                    const float val = dogPyramid[oct][s].at<float>(r, c);
                    if (std::abs(val) < contrastThreshold) continue;

                    bool isMax = true;
                    bool isMin = true;
                    for (int ds = -1; ds <= 1 && (isMax || isMin); ++ds) {
                        for (int dr = -1; dr <= 1 && (isMax || isMin); ++dr) {
                            for (int dc = -1; dc <= 1 && (isMax || isMin); ++dc) {
                                if (ds == 0 && dr == 0 && dc == 0) continue;
                                const float nb = dogPyramid[oct][s + ds].at<float>(r + dr, c + dc);
                                if (val <= nb) isMax = false;
                                if (val >= nb) isMin = false;
                            }
                        }
                    }

                    if (isMax || isMin) {
                        const int tid = omp_get_thread_num();
                        cv::KeyPoint kp;
                        kp.pt = cv::Point2f(c * std::pow(2.f, static_cast<float>(oct)), r * std::pow(2.f, static_cast<float>(oct)));
                        kp.size = SIGMA * std::pow(2.0f, static_cast<float>(s) / static_cast<float>(numScales)) * std::pow(2.f, static_cast<float>(oct)) * 2.0f;
                        kp.response = std::abs(val);
                        kp.octave = oct + (s << 8);
                        local[tid].push_back(kp);
                    }
                }
            }
        }
    }

    for (const auto& vec : local) {
        keypoints.insert(keypoints.end(), vec.begin(), vec.end());
    }
}

void SiftProcessor::assignOrientations(const std::vector<std::vector<cv::Mat>>& gaussPyramid,
                                       std::vector<cv::KeyPoint>& keypoints,
                                       int numOctaves)
{
    // Orientation assignment via weighted local gradient histogram.
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(keypoints.size()); ++i) {
        cv::KeyPoint& kp = keypoints[i];
        const int oct = kp.octave & 255;
        const int scale = (kp.octave >> 8) & 255;

        if (oct < 0 || oct >= numOctaves || oct >= static_cast<int>(gaussPyramid.size())) continue;
        if (scale < 0 || scale >= static_cast<int>(gaussPyramid[oct].size())) continue;

        const cv::Mat& img = gaussPyramid[oct][scale];
        const int r = static_cast<int>(std::round(kp.pt.y / std::pow(2.f, static_cast<float>(oct))));
        const int c = static_cast<int>(std::round(kp.pt.x / std::pow(2.f, static_cast<float>(oct))));

        const float octSigma = kp.size / std::pow(2.f, static_cast<float>(oct)) / 2.0f;
        const float weightSigma = 1.5f * octSigma;
        const int radius = static_cast<int>(std::round(3.0f * weightSigma));
        std::vector<float> hist(36, 0.0f);

        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                const int rr = r + dr;
                const int cc = c + dc;
                if (rr <= 0 || rr >= img.rows - 1 || cc <= 0 || cc >= img.cols - 1) continue;

                const float dx = img.at<float>(rr, cc + 1) - img.at<float>(rr, cc - 1);
                const float dy = img.at<float>(rr + 1, cc) - img.at<float>(rr - 1, cc);
                const float mag = std::sqrt(dx * dx + dy * dy);
                float theta = std::atan2(dy, dx) * 180.0f / kPi;
                if (theta < 0.0f) theta += 360.0f;

                const float w = std::exp(-(dr * dr + dc * dc) / (2.0f * weightSigma * weightSigma));
                hist[static_cast<int>(theta / 10.0f) % 36] += mag * w;
            }
        }

        int best = 0;
        float bestVal = hist[0];
        for (int b = 1; b < 36; ++b) {
            if (hist[b] > bestVal) {
                bestVal = hist[b];
                best = b;
            }
        }
        kp.angle = best * 10.0f + 5.0f;
    }
}
}
