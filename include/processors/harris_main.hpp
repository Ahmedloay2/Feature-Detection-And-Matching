#include <vector>
#include <opencv2/core/types.hpp>
#include <model/image.hpp>
std::vector<cv::Point> applyHarris(Image& image, float k, const std::string& mode, float threshold, int halfwindow);