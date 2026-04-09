/**
 * @file sift_descriptor.cpp
 * @brief Implements 128-dimensional descriptor generation from local orientation histograms.
 */

#include "core/SiftCore.hpp"

#include <omp.h>

namespace cv_assign
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;

// Helper: L2-normalizes a descriptor vector in-place
void l2Normalize(float* desc, int size)
{
    float norm = 0.0f;
    for (int i = 0; i < size; ++i) norm += desc[i] * desc[i];
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < size; ++i) desc[i] /= norm;
    }
}
}

/// Computes 128-dimensional SIFT descriptors for each keypoint.
/// Divides a 16x16 pixel neighborhood around each keypoint (oriented by its dominant angle)
/// into 4x4 sub-regions. For each sub-region, computes an 8-bin gradient orientation histogram,
/// yielding 4*4*8 = 128 dimensions total. Applies Gaussian weighting radially from keypoint,
/// threshold-clips high values, and L2-normalizes final descriptors for robustness.
void SiftProcessor::computeDescriptors(const std::vector<std::vector<cv::Mat>>& gaussPyramid,
                                       std::vector<cv::KeyPoint>& keypoints,
                                       cv::Mat& descriptors,
                                       int numOctaves)
{
    if (keypoints.empty()) {
        descriptors.release();
        return;
    }

    descriptors = cv::Mat::zeros(static_cast<int>(keypoints.size()), 128, CV_32F);

#pragma omp parallel for
    for (int k = 0; k < static_cast<int>(keypoints.size()); ++k) {
        cv::KeyPoint& kp = keypoints[k];
        const int oct = kp.octave & 255;
        const int scale = (kp.octave >> 8) & 255;

        if (oct < 0 || oct >= numOctaves || oct >= static_cast<int>(gaussPyramid.size())) continue;
        if (scale < 0 || scale >= static_cast<int>(gaussPyramid[oct].size())) continue;

        const cv::Mat& img = gaussPyramid[oct][scale];
        const int kp_r = static_cast<int>(std::round(kp.pt.y / std::pow(2.f, static_cast<float>(oct))));
        const int kp_c = static_cast<int>(std::round(kp.pt.x / std::pow(2.f, static_cast<float>(oct))));

        const float cos_t = std::cos(kp.angle * kPi / 180.0f);
        const float sin_t = std::sin(kp.angle * kPi / 180.0f);
        float* desc = descriptors.ptr<float>(k);

        const int D = CELLS_PER_ROW;
        const int N = NUM_BINS;
        const float sigma_oct = kp.size / std::pow(2.f, static_cast<float>(oct)) / 2.0f;
        const float histWidth = 3.0f * sigma_oct;
        const int radius = static_cast<int>(std::round(histWidth * 1.4142f * (D + 1) * 0.5f));

        for (int dr = -radius; dr <= radius; ++dr) {
            for (int dc = -radius; dc <= radius; ++dc) {
                const float rx = (dc * cos_t + dr * sin_t) / histWidth;
                const float ry = (-dc * sin_t + dr * cos_t) / histWidth;
                const float rbin = ry + D / 2.0f - 0.5f;
                const float cbin = rx + D / 2.0f - 0.5f;
                if (rbin < -1.0f || rbin >= D || cbin < -1.0f || cbin >= D) continue;

                const int img_r = kp_r + dr;
                const int img_c = kp_c + dc;
                if (img_r <= 0 || img_r >= img.rows - 1 || img_c <= 0 || img_c >= img.cols - 1) continue;

                const float dx = img.at<float>(img_r, img_c + 1) - img.at<float>(img_r, img_c - 1);
                const float dy = img.at<float>(img_r + 1, img_c) - img.at<float>(img_r - 1, img_c);
                const float mag = std::sqrt(dx * dx + dy * dy);

                float theta = std::atan2(dy, dx) * 180.0f / kPi - kp.angle;
                while (theta < 0.0f) theta += 360.0f;
                while (theta >= 360.0f) theta -= 360.0f;

                const float obin = theta / (360.0f / N);
                const float gauss = std::exp(-(rx * rx + ry * ry) / (2.0f * (D / 2.0f) * (D / 2.0f)));

                const int r0 = static_cast<int>(std::floor(rbin));
                const int c0 = static_cast<int>(std::floor(cbin));
                const int o0 = static_cast<int>(std::floor(obin));
                const float drf = rbin - r0;
                const float dcf = cbin - c0;
                const float dof = obin - o0;

                for (int ri = 0; ri <= 1; ++ri) {
                    const int rr = r0 + ri;
                    if (rr < 0 || rr >= D) continue;
                    const float wr = ri ? drf : (1.0f - drf);
                    for (int ci = 0; ci <= 1; ++ci) {
                        const int cc = c0 + ci;
                        if (cc < 0 || cc >= D) continue;
                        const float wc = ci ? dcf : (1.0f - dcf);
                        for (int oi = 0; oi <= 1; ++oi) {
                            const int oo = (o0 + oi) % N;
                            const float wo = oi ? dof : (1.0f - dof);
                            desc[(rr * D + cc) * N + oo] += mag * gauss * wr * wc * wo;
                        }
                    }
                }
            }
        }

        l2Normalize(desc, 128);
        for (int i = 0; i < 128; ++i) desc[i] = std::min(desc[i], 0.2f);
        l2Normalize(desc, 128);
    }
}
}
