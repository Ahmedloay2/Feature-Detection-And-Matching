#include "SiftCore.hpp"

#include <opencv2/imgproc.hpp>

namespace cv_assign
{
// Entry point that orchestrates the full SIFT pipeline.
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
