#include <model/image.hpp>
#include "../include/processors/harris/strcutre_tensor.hpp"
#include <utils/utils.hpp>
void applyStructureTensor(Image& img) {

	if (!img.has("gradient_xx") || !img.has("gradient_yy") || !img.has("gradient_xy"))
		throw std::runtime_error("gradient stage not computed yet.");

	cv::Mat Ix2 = img.get("gradient_xx");
	cv::Mat Iy2 = img.get("gradient_yy");
	cv::Mat IxIy = img.get("gradient_xy");

	const std::vector<float> kernel = { 1.f / 16, 4.f / 16, 6.f / 16, 4.f / 16, 1.f / 16 };

	cv::Mat Sxx = utils::convolveH<float>(utils::convolveV<float>(Ix2, kernel), kernel);
	cv::Mat Syy = utils::convolveH<float>(utils::convolveV<float>(Iy2, kernel), kernel);
	cv::Mat Sxy = utils::convolveH<float>(utils::convolveV<float>(IxIy, kernel), kernel);

	img.store("structure_xx", Sxx);
	img.store("structure_yy", Syy);
	img.store("structure_xy", Sxy);
}