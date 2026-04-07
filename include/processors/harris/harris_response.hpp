/**
 * @file harris_response.hpp
 * @brief Declares Harris corner response computation from structure tensor.
 */

#include "model/image.hpp"

/// @brief Compute Harris corner response: R = det(M) - k*trace(M)^2
///
/// Harris response is computed from the structure tensor M:
/// $$R = \det(M) - k \cdot \mathrm{trace}(M)^2$$
/// $$\det(M) = I_{xx} I_{yy} - (I_{xy})^2$$
/// $$\mathrm{trace}(M) = I_{xx} + I_{yy}$$
///
/// The response R is high at corners (two large eigenvalues),
/// low at edges (one large eigenvalue), and near zero on flat surfaces.
/// Parameter k (typically 0.04-0.06) balances the two terms.
///
/// @param img The image to process. Reads img.cache["structure_xx"],
///            ["structure_yy"], ["structure_xy"].
///            Stores img.cache["harris_response"] (CV_32FC1).
/// @param k Harris parameter (typically 0.04; range 0.01-0.1)
/// @throws std::runtime_error if structure tensor stage hasn't been computed yet
void computeHarrisResponse(Image& img, float k);