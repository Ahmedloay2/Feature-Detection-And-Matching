/**
 * @file shi_tomasi.cpp
 * @brief Implements Shi-Tomasi corner response using minimum eigenvalue of structure tensor.
 */

#include "../../../include/processors/harris/shi_tomasi.hpp"
#include <model/image.hpp>

/// Computes Shi-Tomasi response as min(λ1, λ2) from structure tensor elements.
/// Uses efficient formula: R = (trace(M) - sqrt(trace(M)² - 4*det(M))) / 2
/// which avoids explicit eigenvalue decomposition.
void computeShiTomasi(Image& img)
{
	if (!img.has("structure_xx") || !img.has("structure_yy") || !img.has("structure_xy"))
		throw std::runtime_error("structure stage not computed yet.");

	cv::Mat Sxx = img.get("structure_xx");
	cv::Mat Syy = img.get("structure_yy");
	cv::Mat Sxy = img.get("structure_xy");

	cv::Mat diff, discriminant;
	cv::subtract(Sxx, Syy, diff);
	discriminant =  diff.mul(diff) + 4.f * Sxy.mul(Sxy);
	cv::sqrt(discriminant, discriminant);
	cv::Mat R = 0.5f * ( (Sxx + Syy) - discriminant);

	img.store("shi_tomasi_response", R);
}
