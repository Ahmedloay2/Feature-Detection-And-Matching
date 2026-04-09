/**
 * @file grayscale.cpp
 * @brief Implements RGB-to-grayscale conversion with standard luminance weighting.
 */

#include "../include/processors/harris/grayscale.hpp"
#include <stdexcept>
#include "../include/model/image.hpp"

/// Converts color or grayscale images to float32 grayscale format.
/// For 3-channel (BGR) images, applies ITU-R BT.601 luminance formula.
/// For single-channel images, simply converts to CV_32FC1 (float).
/// Throws exception if image has unsupported channel count.
void convertToGrayscale(Image& img)
{
    if (img.mat.channels() == 3)
    {
        const int rows = img.mat.rows;
        const int cols = img.mat.cols;

        cv::Mat gray(rows, cols, CV_32FC1);

        for (int i = 0; i < rows; ++i)
        {
            // uchar pointer into the BGR source row
            const uchar* srcRow = img.mat.ptr<uchar>(i);
            float* dstRow = gray.ptr<float>(i);

            for (int j = 0; j < cols; ++j)
            {
                // BGR channel order: srcRow[3j+0]=B, [3j+1]=G, [3j+2]=R
                const float B = static_cast<float>(srcRow[j * 3 + 0]);
                const float G = static_cast<float>(srcRow[j * 3 + 1]);
                const float R = static_cast<float>(srcRow[j * 3 + 2]);

                // ITU-R BT.601 luma coefficients
                dstRow[j] = 0.114f * B + 0.587f * G + 0.299f * R;
            }
        }
        img.store("grayscale", gray);
    }
    else if (img.mat.channels() == 1)
    {
        // Already grayscale — just convert to float
        cv::Mat gray;
        img.mat.convertTo(gray, CV_32F);
        img.store("grayscale", gray);
    }
    else
    {
        throw std::runtime_error("Unsupported number of channels: " +
            std::to_string(img.mat.channels()));
    }
}