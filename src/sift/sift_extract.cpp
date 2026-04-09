/**
 * @file sift_extract.cpp
 * @brief Implements the complete SIFT feature extraction pipeline coordination.
 */

#include "core/SiftCore.hpp"

#include <opencv2/imgproc.hpp>

namespace cv_assign
{
/// Main orchestration function for SIFT extraction pipeline.
/// Converts input to grayscale, normalizes to [0,1] float, builds pyramids,
/// detects extrema, assigns orientations, and computes 128-D descriptors.
/// Returns detected keypoints and their descriptors for matching.
void SiftProcessor::extractFeatures(const cv::Mat& image,
                                    std::vector<cv::KeyPoint>& keypoints,
                                    cv::Mat& descriptors,
                                    float contrastThreshold,
                                    int numOctaves,
                                    int numScales)
{
    if (image.empty()) return;

    numOctaves = std::max(1, numOctaves);
    numScales = std::max(2, numScales);

    cv::Mat gray;
    if (image.channels() == 3) cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else gray = image.clone();

    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F, 1.0 / 255.0);

    std::vector<std::vector<cv::Mat>> gaussPyramid;
    std::vector<std::vector<cv::Mat>> dogPyramid;

    buildGaussianPyramid(grayFloat, gaussPyramid, numOctaves, numScales);
    buildDoGPyramid(gaussPyramid, dogPyramid);
    detectExtrema(dogPyramid, keypoints, contrastThreshold, numOctaves, numScales);
    assignOrientations(gaussPyramid, keypoints, numOctaves);
    computeDescriptors(gaussPyramid, keypoints, descriptors, numOctaves);
}
}
