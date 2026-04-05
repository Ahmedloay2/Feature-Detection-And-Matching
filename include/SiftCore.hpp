#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace cv_assign {
    class SiftProcessor {
    public:
        // Main unified SIFT pipeline extraction (detection + description)
        static void extractFeatures(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
        
    private:
        // Parameters for Pyramids
        static constexpr int NUM_OCTAVES = 3;
        static constexpr int NUM_SCALES = 3;
        static constexpr float SIGMA = 1.6f;
        static constexpr float CONTRAST_THRESHOLD = 0.04f;
        
        static void buildGaussianPyramid(const cv::Mat& baseImage, std::vector<std::vector<cv::Mat>>& gaussPyramid);
        static void buildDoGPyramid(const std::vector<std::vector<cv::Mat>>& gaussPyramid, std::vector<std::vector<cv::Mat>>& dogPyramid);
        static void detectExtrema(const std::vector<std::vector<cv::Mat>>& dogPyramid, std::vector<cv::KeyPoint>& keypoints);
        static void assignOrientations(const std::vector<std::vector<cv::Mat>>& gaussPyramid, std::vector<cv::KeyPoint>& keypoints);
        static void computeDescriptors(const std::vector<std::vector<cv::Mat>>& gaussPyramid, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);
    };
}
