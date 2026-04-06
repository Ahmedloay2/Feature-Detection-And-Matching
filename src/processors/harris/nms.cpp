#include "../../../include/processors/harris/nms.hpp"

void applyCornerNMS(Image& img, const std::string& responseKey, int halfWindow) {
    std::string threshKey = responseKey + "_threshold";
    if (!img.has(threshKey))
        throw std::runtime_error("threshold stage not computed yet.");

    cv::Mat input = img.get(threshKey).clone();
    cv::Mat output = cv::Mat::zeros(input.size(), CV_32FC1);

    const int rows = input.rows;
    const int cols = input.cols;

    for (int i = 0; i < rows; ++i) {
        const float* rowPtr = input.ptr<float>(i);
        for (int j = 0; j < cols; ++j) {
            float val = rowPtr[j];
            if (val == 0.f) continue;

            bool isMax = true;
            int rMin = std::max(0, i - halfWindow);
            int rMax = std::min(rows - 1, i + halfWindow);
            int cMin = std::max(0, j - halfWindow);
            int cMax = std::min(cols - 1, j + halfWindow);

            for (int ni = rMin; ni <= rMax && isMax; ++ni) {
                const float* nRowPtr = input.ptr<float>(ni);
                for (int nj = cMin; nj <= cMax && isMax; ++nj) {
                    if (ni == i && nj == j) continue;
                    if (nRowPtr[nj] > val) isMax = false;
                }
            }

            if (isMax)
                output.at<float>(i, j) = val;
        }
    }

    img.store(responseKey + "_corner", output);
}