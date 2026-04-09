/**
 * @file strcutre_tensor.hpp
 * @brief Declares structure tensor construction and windowed smoothing for corner detection.
 */

#include <model/image.hpp>

/// @brief Compute and smooth the structure tensor matrix M from image gradients.
///
/// For each pixel, computes the structure tensor (also called second moment matrix):
/// $$M = \begin{pmatrix} I_x^2 & I_xI_y \\ I_xI_y & I_y^2 \end{pmatrix}$$
/// 
/// where Ix and Iy are image gradients. Then applies Gaussian smoothing to each
/// component to integrate local gradient information. This tensor encodes the
/// local image "structure" around each pixel - corners have two large eigenvalues,
/// edges have one, and flat regions have small eigenvalues.
///
/// @param img The image to process. Reads img.cache["gradient_xx"], ["gradient_yy"],
///            ["gradient_xy"]. Stores:
///            - img.cache["structure_xx"] (CV_32FC1): Smoothed Ixx component
///            - img.cache["structure_yy"] (CV_32FC1): Smoothed Iyy component
///            - img.cache["structure_xy"] (CV_32FC1): Smoothed Ixy component
/// @throws std::runtime_error if gradient stage has not been computed yet
void applyStructureTensor(Image& img);