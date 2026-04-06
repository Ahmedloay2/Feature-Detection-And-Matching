#include <model/image.hpp>
#include "../../../include/processors/harris/harris_response.hpp"

void computeHarrisResponse(Image& img, float k)
{
	if (!img.has("structure_xx")|| !img.has("structure_yy") || !img.has("structure_xy"))
		throw std::runtime_error("structure stage not computed yet.");

	cv::Mat Sxx = img.get("structure_xx");
	cv::Mat Syy = img.get("structure_yy");
	cv::Mat Sxy = img.get("structure_xy");
	cv::Mat trace = Sxx + Syy;
	cv::Mat R = Sxx.mul(Syy) - Sxy.mul(Sxy) - k * trace.mul(trace);

	img.store("harris_response", R);
}
