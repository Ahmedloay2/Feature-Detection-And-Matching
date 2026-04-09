#include "processors/feature_matcher.hpp"
#include <cmath>

namespace cv_assign
{
    namespace matching
    {

        std::vector<cv::DMatch> matchSSD(const cv::Mat &desc1,
                                         const cv::Mat &desc2,
                                         float ratio)
        {
            std::vector<cv::DMatch> good;
            if (desc1.empty() || desc2.empty())
                return good;

            for (int i = 0; i < desc1.rows; ++i)
            {
                const float *q = desc1.ptr<float>(i);
                float best1 = 1e30f, best2 = 1e30f;
                int idx1 = -1;

                for (int j = 0; j < desc2.rows; ++j)
                {
                    const float *t = desc2.ptr<float>(j);
                    float ssd = 0.f;
                    for (int d = 0; d < 128; ++d)
                    {
                        const float diff = q[d] - t[d];
                        ssd += diff * diff;
                    }
                    if (ssd < best1)
                    {
                        best2 = best1;
                        best1 = ssd;
                        idx1 = j;
                    }
                    else if (ssd < best2)
                        best2 = ssd;
                }

                if (idx1 >= 0 && std::sqrt(best1) < ratio * std::sqrt(best2))
                    good.emplace_back(i, idx1, std::sqrt(best1));
            }
            return good;
        }

        std::vector<cv::DMatch> matchNCC(const cv::Mat &desc1,
                                         const cv::Mat &desc2,
                                         float minCorr)
        {
            std::vector<cv::DMatch> good;
            if (desc1.empty() || desc2.empty())
                return good;

            for (int i = 0; i < desc1.rows; ++i)
            {
                const float *q = desc1.ptr<float>(i);
                float qMean = 0.f;
                for (int d = 0; d < 128; ++d)
                    qMean += q[d];
                qMean /= 128.f;

                float qNorm = 0.f;
                for (int d = 0; d < 128; ++d)
                {
                    const float v = q[d] - qMean;
                    qNorm += v * v;
                }
                if (qNorm <= 1e-12f)
                    continue;
                qNorm = std::sqrt(qNorm);

                float best = -2.f;
                int bestIdx = -1;

                for (int j = 0; j < desc2.rows; ++j)
                {
                    const float *t = desc2.ptr<float>(j);
                    float tMean = 0.f;
                    for (int d = 0; d < 128; ++d)
                        tMean += t[d];
                    tMean /= 128.f;

                    float tNorm = 0.f, dot = 0.f;
                    for (int d = 0; d < 128; ++d)
                    {
                        const float qv = q[d] - qMean;
                        const float tv = t[d] - tMean;
                        dot += qv * tv;
                        tNorm += tv * tv;
                    }
                    if (tNorm <= 1e-12f)
                        continue;

                    const float ncc = dot / (qNorm * std::sqrt(tNorm));
                    if (ncc > best)
                    {
                        best = ncc;
                        bestIdx = j;
                    }
                }

                if (bestIdx >= 0 && best >= minCorr)
                    good.emplace_back(i, bestIdx, 1.0f - best);
            }
            return good;
        }

    } // namespace matching
} // namespace cv_assign
