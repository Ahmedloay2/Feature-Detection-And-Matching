/**
 * @file shi_tomasi.hpp
 * @brief Declares Shi-Tomasi corner response using tensor eigenvalues.
 */

#include <model/image.hpp>

/// @brief Compute Shi-Tomasi corner response: R = min(λ1, λ2)
///
/// Shi-Tomasi response uses the smaller eigenvalue of the structure tensor M:
/// $$R = \min(\lambda_1, \lambda_2)$$
///
/// where λ1 and λ2 are the eigenvalues of M. This is equivalent to:
/// $$R = \frac{\mathrm{trace}(M) - \sqrt{\mathrm{trace}(M)^2 - 4\det(M)}}{2}$$
///
/// Shi-Tomasi corners are often more stable than Harris for tracking applications.
/// High response indicates a corner (two large eigenvalues),
/// low response indicates an edge or flat region (one small eigenvalue).
///
/// @param img The image to process. Reads img.cache["structure_xx"],
///            ["structure_yy"], ["structure_xy"].
///            Stores img.cache["shi_tomasi_response"] (CV_32FC1).
/// @throws std::runtime_error if structure tensor stage hasn't been computed yet
void computeShiTomasi(Image& img);