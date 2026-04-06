#include "SiftCore.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <omp.h>

namespace cv_assign
{
    const float PI = 3.14159265358979323846f;

    void SiftProcessor::extractFeatures(const cv::Mat &image,
                                        std::vector<cv::KeyPoint> &keypoints,
                                        cv::Mat &descriptors,
                                        float contrastThreshold)
    {
        if (image.empty())
            return;
        cv::Mat gray;
        if (image.channels() == 3)
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        else
            gray = image.clone();
        cv::Mat grayFloat;
        gray.convertTo(grayFloat, CV_32F, 1.0 / 255.0);
        std::vector<std::vector<cv::Mat>> gaussPyramid, dogPyramid;
        buildGaussianPyramid(grayFloat, gaussPyramid);
        buildDoGPyramid(gaussPyramid, dogPyramid);
        detectExtrema(dogPyramid, keypoints, contrastThreshold);
        assignOrientations(gaussPyramid, keypoints);
        computeDescriptors(gaussPyramid, keypoints, descriptors);
    }

    void SiftProcessor::buildGaussianPyramid(const cv::Mat &baseImage,
                                             std::vector<std::vector<cv::Mat>> &gaussPyramid)
    {
        gaussPyramid.resize(NUM_OCTAVES);
        float k = std::pow(2.0f, 1.0f / NUM_SCALES);
        for (int oct = 0; oct < NUM_OCTAVES; oct++)
        {
            cv::Mat octBase;
            if (oct == 0)
                octBase = baseImage;
            else
            {
                if (gaussPyramid[oct - 1].size() <= (size_t)NUM_SCALES ||
                    gaussPyramid[oct - 1][NUM_SCALES].cols <= 4 ||
                    gaussPyramid[oct - 1][NUM_SCALES].rows <= 4)
                    break;
                cv::resize(gaussPyramid[oct - 1][NUM_SCALES], octBase, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
            }
            gaussPyramid[oct].resize(NUM_SCALES + 3);
            float base_sigma = (oct == 0) ? 0.0f : SIGMA;
#pragma omp parallel for
            for (int s = 0; s < NUM_SCALES + 3; s++)
            {
                float total_sigma = SIGMA * std::pow(k, s);
                float apply_sigma = std::sqrt(std::max(0.0f, total_sigma * total_sigma - base_sigma * base_sigma));
                if (apply_sigma < 0.01f)
                    gaussPyramid[oct][s] = octBase.clone();
                else
                    cv::GaussianBlur(octBase, gaussPyramid[oct][s], cv::Size(0, 0), apply_sigma);
            }
        }
    }

    void SiftProcessor::buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                        std::vector<std::vector<cv::Mat>> &dogPyramid)
    {
        dogPyramid.resize(NUM_OCTAVES);
        for (int oct = 0; oct < NUM_OCTAVES; oct++)
        {
            int n = (int)gaussPyramid[oct].size();
            dogPyramid[oct].resize(n - 1);
#pragma omp parallel for
            for (int s = 0; s < n - 1; s++)
                cv::subtract(gaussPyramid[oct][s + 1], gaussPyramid[oct][s], dogPyramid[oct][s]);
        }
    }

    // Slider range printed in the UI:
    //  0.001 = extreme-low  | 0.007 = relaxed default
    //  0.013 = Lowe standard | 0.030 = strict
    void SiftProcessor::detectExtrema(const std::vector<std::vector<cv::Mat>> &dogPyramid,
                                      std::vector<cv::KeyPoint> &keypoints,
                                      float contrastThreshold)
    {
        keypoints.clear();
        std::vector<std::vector<cv::KeyPoint>> local_kps(omp_get_max_threads());
        for (int oct = 0; oct < NUM_OCTAVES; oct++)
        {
            if (dogPyramid[oct].empty())
                continue;
            int rows = dogPyramid[oct][0].rows;
            int cols = dogPyramid[oct][0].cols;
            int scales = (int)dogPyramid[oct].size();
            for (int s = 1; s < scales - 1; s++)
            {
#pragma omp parallel for collapse(2)
                for (int r = 1; r < rows - 1; r++)
                {
                    for (int c = 1; c < cols - 1; c++)
                    {
                        float val = dogPyramid[oct][s].at<float>(r, c);
                        if (std::abs(val) < contrastThreshold)
                            continue;
                        bool isMax = true, isMin = true;
                        for (int ds = -1; ds <= 1 && (isMax || isMin); ds++)
                            for (int dr = -1; dr <= 1 && (isMax || isMin); dr++)
                                for (int dc = -1; dc <= 1 && (isMax || isMin); dc++)
                                {
                                    if (ds == 0 && dr == 0 && dc == 0)
                                        continue;
                                    float nb = dogPyramid[oct][s + ds].at<float>(r + dr, c + dc);
                                    if (val <= nb)
                                        isMax = false;
                                    if (val >= nb)
                                        isMin = false;
                                }
                        if (isMax || isMin)
                        {
                            int tid = omp_get_thread_num();
                            cv::KeyPoint kp;
                            kp.pt = cv::Point2f(c * std::pow(2.f, oct), r * std::pow(2.f, oct));
                            kp.size = SIGMA * std::pow(2.0, (float)s / NUM_SCALES) * std::pow(2, oct) * 2.0f;
                            kp.response = std::abs(val);
                            kp.octave = oct + (s << 8);
                            local_kps[tid].push_back(kp);
                        }
                    }
                }
            }
        }
        for (const auto &lv : local_kps)
            keypoints.insert(keypoints.end(), lv.begin(), lv.end());
    }

    void SiftProcessor::assignOrientations(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                           std::vector<cv::KeyPoint> &keypoints)
    {
#pragma omp parallel for
        for (int k = 0; k < (int)keypoints.size(); k++)
        {
            cv::KeyPoint &kp = keypoints[k];
            int oct = kp.octave & 255;
            int scale = (kp.octave >> 8) & 255;
            if (oct >= NUM_OCTAVES || scale >= (int)gaussPyramid[oct].size())
                continue;
            const cv::Mat &img = gaussPyramid[oct][scale];
            int r = (int)std::round(kp.pt.y / std::pow(2.f, oct));
            int c = (int)std::round(kp.pt.x / std::pow(2.f, oct));
            float oct_sigma = kp.size / std::pow(2.f, oct) / 2.0f;
            float weight_sigma = 1.5f * oct_sigma;
            int radius = (int)std::round(3.0f * weight_sigma);
            std::vector<float> hist(36, 0.0f);
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    if (r + i <= 0 || r + i >= img.rows - 1 || c + j <= 0 || c + j >= img.cols - 1)
                        continue;
                    float dx = img.at<float>(r + i, c + j + 1) - img.at<float>(r + i, c + j - 1);
                    float dy = img.at<float>(r + i + 1, c + j) - img.at<float>(r + i - 1, c + j);
                    float mag = std::sqrt(dx * dx + dy * dy);
                    float theta = std::atan2(dy, dx) * 180.0f / PI;
                    if (theta < 0)
                        theta += 360.0f;
                    float w = std::exp(-(i * i + j * j) / (2.0f * weight_sigma * weight_sigma));
                    hist[(int)(theta / 10.0f) % 36] += mag * w;
                }
            }
            int max_bin = 0;
            float max_val = hist[0];
            for (int b = 1; b < 36; b++)
                if (hist[b] > max_val)
                {
                    max_val = hist[b];
                    max_bin = b;
                }
            kp.angle = max_bin * 10.0f + 5.0f;
        }
    }

    static void l2NormalizeVector(float* desc, int size) {
        float norm = 0.0f;
        for (int i = 0; i < size; i++)
            norm += desc[i] * desc[i];
        norm = std::sqrt(norm);
        if (norm > 1e-6f)
            for (int i = 0; i < size; i++)
                desc[i] /= norm;
    }

    void SiftProcessor::computeDescriptors(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                           std::vector<cv::KeyPoint> &keypoints,
                                           cv::Mat &descriptors)
    {
        if (keypoints.empty())
            return;
        descriptors = cv::Mat::zeros((int)keypoints.size(), 128, CV_32F);
#pragma omp parallel for
        for (int k = 0; k < (int)keypoints.size(); k++)
        {
            cv::KeyPoint &kp = keypoints[k];
            int oct = kp.octave & 255;
            int scale = (kp.octave >> 8) & 255;
            if (oct >= NUM_OCTAVES || scale >= (int)gaussPyramid[oct].size())
                continue;
            const cv::Mat &img = gaussPyramid[oct][scale];
            int kp_r = (int)std::round(kp.pt.y / std::pow(2.f, oct));
            int kp_c = (int)std::round(kp.pt.x / std::pow(2.f, oct));
            float cos_t = std::cos(kp.angle * PI / 180.0f);
            float sin_t = std::sin(kp.angle * PI / 180.0f);
            float *desc_ptr = descriptors.ptr<float>(k);
            const int D = CELLS_PER_ROW;
            const int N = NUM_BINS;
            float sigma_oct = kp.size / std::pow(2.f, oct) / 2.0f;
            float hist_width = 3.0f * sigma_oct;
            int radius = (int)std::round(hist_width * 1.4142f * (D + 1) * 0.5f);
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    float rx = (j * cos_t + i * sin_t) / hist_width;
                    float ry = (-j * sin_t + i * cos_t) / hist_width;
                    float rbin = ry + D / 2.0f - 0.5f;
                    float cbin = rx + D / 2.0f - 0.5f;
                    if (rbin < -1.0f || rbin >= D || cbin < -1.0f || cbin >= D)
                        continue;
                    int img_r = kp_r + i, img_c = kp_c + j;
                    if (img_r <= 0 || img_r >= img.rows - 1 || img_c <= 0 || img_c >= img.cols - 1)
                        continue;
                    float dx = img.at<float>(img_r, img_c + 1) - img.at<float>(img_r, img_c - 1);
                    float dy = img.at<float>(img_r + 1, img_c) - img.at<float>(img_r - 1, img_c);
                    float mag = std::sqrt(dx * dx + dy * dy);
                    float theta = std::atan2(dy, dx) * 180.0f / PI - kp.angle;
                    while (theta < 0.0f)
                        theta += 360.0f;
                    while (theta >= 360.0f)
                        theta -= 360.0f;
                    float obin = theta / (360.0f / N);
                    float gauss = std::exp(-(rx * rx + ry * ry) / (2.0f * (D / 2.0f) * (D / 2.0f)));
                    int r0 = (int)std::floor(rbin), c0 = (int)std::floor(cbin), o0 = (int)std::floor(obin);
                    float dr = rbin - r0, dc = cbin - c0, dth = obin - o0;
                    for (int ri = 0; ri <= 1; ri++)
                    {
                        int rr = r0 + ri;
                        if (rr < 0 || rr >= D)
                            continue;
                        float wr = ri ? dr : (1.0f - dr);
                        for (int ci = 0; ci <= 1; ci++)
                        {
                            int cc = c0 + ci;
                            if (cc < 0 || cc >= D)
                                continue;
                            float wc = ci ? dc : (1.0f - dc);
                            for (int oi = 0; oi <= 1; oi++)
                            {
                                int oo = (o0 + oi) % N;
                                float wo = oi ? dth : (1.0f - dth);
                                desc_ptr[(rr * D + cc) * N + oo] += mag * gauss * wr * wc * wo;
                            }
                        }
                    }
                }
            }
            
            l2NormalizeVector(desc_ptr, 128);

            for (int i = 0; i < 128; i++)
            {
                desc_ptr[i] = std::min(desc_ptr[i], 0.2f);
            }
            
            l2NormalizeVector(desc_ptr, 128);
        }
    }
} // namespace cv_assign