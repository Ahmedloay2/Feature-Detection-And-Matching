#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace cv_assign
{
    namespace matching
    {

        /// @brief Match descriptors using Sum-of-Squared-Differences (SSD) with ratio test.
        /// @param desc1 Descriptor matrix from image 1 (rows=count, cols=128, type=CV_32F)
        /// @param desc2 Descriptor matrix from image 2
        /// @param ratio Lowe's ratio threshold for ambiguous matches (default ~0.8)
        /// @return Vector of DMatch with indices into the descriptor arrays
        std::vector<cv::DMatch> matchSSD(const cv::Mat &desc1,
                                         const cv::Mat &desc2,
                                         float ratio);

        /// @brief Match descriptors using Normalized Cross-Correlation (NCC) similarity.
        /// @param desc1 Descriptor matrix from image 1
        /// @param desc2 Descriptor matrix from image 2
        /// @param minCorr Minimum correlation threshold for a valid match
        /// @return Vector of DMatch with indices into the descriptor arrays
        std::vector<cv::DMatch> matchNCC(const cv::Mat &desc1,
                                         const cv::Mat &desc2,
                                         float minCorr);

    } // namespace matching
} // namespace cv_assign
