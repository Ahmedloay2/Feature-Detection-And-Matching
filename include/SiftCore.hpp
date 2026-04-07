#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace cv_assign
{
    class SiftProcessor
    {
    public:
        // contrastThreshold is exposed so the UI slider can pass it in.
        // Range:  0.001 = extreme-low (many noisy KPs),
        //         0.007 = relaxed default,
        //         0.013 = Lowe's standard (CONTRAST_THRESHOLD/NUM_SCALES),
        //         0.030 = strict (few, strong KPs only).
        static void extractFeatures(const cv::Mat &image,
                                    std::vector<cv::KeyPoint> &keypoints,
                                    cv::Mat &descriptors,
                                    float contrastThreshold = 0.007f,
                                    int numOctaves = 4,    
                                    int numScales = 3);

    private:
        static constexpr int CELLS_PER_ROW = 4;
        static constexpr int NUM_BINS = 8;
        static constexpr float SIGMA = 1.6f;
        static constexpr float CONTRAST_THRESHOLD = 0.04f; // kept for reference

        static void buildGaussianPyramid(const cv::Mat &baseImage,
                                         std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                         int numOctaves,
                                         int numScales);
        static void buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                    std::vector<std::vector<cv::Mat>> &dogPyramid);
        static void detectExtrema(const std::vector<std::vector<cv::Mat>> &dogPyramid,
                                  std::vector<cv::KeyPoint> &keypoints,
                                  float contrastThreshold,
                                  int numOctaves,
                                  int numScales);
        static void assignOrientations(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                       std::vector<cv::KeyPoint> &keypoints,
                                       int numOctaves);
        static void computeDescriptors(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                       std::vector<cv::KeyPoint> &keypoints,
                                       cv::Mat &descriptors,
                                       int numOctaves);
    };
}