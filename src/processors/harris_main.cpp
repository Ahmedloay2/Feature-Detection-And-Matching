/**
 * @file harris_main.cpp
 * @brief Implements the pipeline orchestration for Harris and Shi-Tomasi corner detection.
 */

#include "../../include/processors/harris_main.hpp"
#include <model/image.hpp>
#include <processors/harris/grayscale.hpp>
#include <processors/harris/gradient.hpp>
#include <processors/harris/shi_tomasi.hpp>
#include <processors/harris/harris_response.hpp>
#include <processors/harris/threshold.hpp>
#include <processors/harris/nms.hpp>
#include <processors/harris/strcutre_tensor.hpp>
#include <processors/harris/gaussian.hpp>

#include <chrono>
#include <iostream>
using Clock = std::chrono::high_resolution_clock;

/// Main orchestration function that runs the complete corner detection pipeline.
/// Executes stages in order: grayscale -> gradient -> structure tensor ->
/// response computation (Harris or Shi-Tomasi) -> threshold -> NMS.
/// Times each stage and prints to stdout. Returns detected corner pixel coordinates.
std::vector<cv::Point> applyHarris(Image& image, float k,
    const std::string& mode, float threshold, int halfWindow)
{
    auto t = [](auto t0) {
        return std::chrono::duration<double, std::milli>(
            Clock::now() - t0).count();
        };

    auto t0 = Clock::now();
    if (!image.has("grayscale")) {
        toGrayscale(image);
    }
    std::cout << "  grayscale:        " << t(t0) << " ms\n";

    t0 = Clock::now();
    if (!image.has("gradient_xx")) {
        computeGradient(image);
    }
    std::cout << "  gradient:         " << t(t0) << " ms\n";

    t0 = Clock::now();
    if (!image.has("structure_xx")) {
        applyStructureTensor(image);
    }
    std::cout << "  structure tensor: " << t(t0) << " ms\n";

    t0 = Clock::now();
    const std::string responseKey = mode + "_response";
    if (mode == "shi_tomasi") computeShiTomasi(image);
    else                      computeHarrisResponse(image, k);
    std::cout << "  response:         " << t(t0) << " ms\n";

    t0 = Clock::now();
    applyCornerThreshold(image, responseKey, threshold);
    std::cout << "  threshold:        " << t(t0) << " ms\n";

    t0 = Clock::now();
    applyCornerNMS(image, responseKey, halfWindow);
    std::cout << "  NMS:              " << t(t0) << " ms\n";

    std::vector<cv::Point> corners;
    const cv::Mat& result = image.get(responseKey + "_corner");
    for (int i = 0; i < result.rows; ++i) {
        const float* row = result.ptr<float>(i);
        for (int j = 0; j < result.cols; ++j)
            if (row[j] > 0.f)
                corners.push_back(cv::Point(j, i));
    }
    return corners;
}