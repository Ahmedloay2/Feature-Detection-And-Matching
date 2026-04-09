/**
 * @file threshold.cpp
 * @brief Implements binary thresholding of corner response maps.
 */

#include "../../../include/processors/harris/threshold.hpp"
#include <model/image.hpp>

/// Converts corner response map to binary by normalizing to [0,255] range
/// and thresholding against a user-provided value. Finds global min/max,
/// normalizes, then suppresses pixels below the threshold to 0.
/// Stores binary output with key "responseKey_threshold" in image cache.
void applyCornerThreshold(Image& img, const std::string& responseKey, float threshold) {
    if (!img.has(responseKey))
        throw std::runtime_error("response stage not computed yet.");

    const cv::Mat& response = img.get(responseKey);
    const int rows = response.rows;
    const int cols = response.cols;

    // Manual min/max scan
    float minVal = response.at<float>(0, 0);
    float maxVal = response.at<float>(0, 0);
    for (int i = 0; i < rows; ++i) {
        const float* row = response.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            if (row[j] < minVal) minVal = row[j];
            if (row[j] > maxVal) maxVal = row[j];
        }
    }

    const float range = (maxVal - minVal) > 1e-6f ? (maxVal - minVal) : 1.f;

    // Manual normalize to [0, 255] and manual threshold
    cv::Mat output = cv::Mat::zeros(response.size(), CV_32FC1);
    for (int i = 0; i < rows; ++i) {
        const float* src = response.ptr<float>(i);
        float* dst = output.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            float normalized = (src[j] - minVal) / range * 255.f;
            if (normalized > threshold)
                dst[j] = normalized;
            // else stays 0
        }
    }

    img.store(responseKey + "_threshold", output);
}