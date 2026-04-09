/**
 * @file SiftCore.hpp
 * @brief Declares the SIFT processor class and scale-space feature extraction API.
 */

#pragma once
#include <opencv2/core.hpp>
#include <vector>

namespace cv_assign
{
    /// @class SiftProcessor
    /// @brief Implements the complete SIFT (Scale-Invariant Feature Transform) algorithm.
    /// Provides static methods for extracting scale-invariant keypoints and 128-D descriptors
    /// from images, useful for feature matching and image recognition tasks.
    class SiftProcessor
    {
    public:
        /// @brief Extract SIFT keypoints and descriptors from an image.
        /// 
        /// Performs all SIFT stages:
        /// 1. Build Gaussian scale-space pyramid across octaves
        /// 2. Compute Difference-of-Gaussians (DoG) pyramid
        /// 3. Detect scale-space extrema (candidate keypoints)
        /// 4. Filter by contrast to eliminate weak features
        /// 5. Assign dominant orientation(s) to each keypoint
        /// 6. Compute 128-dimensional descriptor vectors
        ///
        /// @param image Input image (any size; will be internally resized if needed)
        /// @param keypoints [OUT] Detected SIFT keypoints (location, scale, orientation)
        /// @param descriptors [OUT] 128-D descriptor for each keypoint (rows=count, type=CV_32F)
        /// @param contrastThreshold Controls sensitivity of extrema detection.
        ///        - 0.001  = very permissive (many noisy keypoints)
        ///        - 0.007  = relaxed default (good balance)
        ///        - 0.013  = Lowe's standard (fewer, stronger keypoints)
        ///        - 0.030  = strict mode (very few, high-quality keypoints only)
        /// @param numOctaves Number of scale octaves (pyramid depth, typically 4)
        /// @param numScales Number of scales per octave (typically 3-5, more=better localization)
        ///
        /// @note Execution time scales roughly with image area * numOctaves * numScales.
        /// @note Keypoints and descriptors are NOT pre-allocated; they are cleared and refilled.
        static void extractFeatures(const cv::Mat &image,
                                    std::vector<cv::KeyPoint> &keypoints,
                                    cv::Mat &descriptors,
                                    float contrastThreshold = 0.007f,
                                    int numOctaves = 4,    
                                    int numScales = 3);

    private:
        // ─── SIFT Algorithm Constants ───────────────────────────────────────
        static constexpr int CELLS_PER_ROW = 4;        ///< 4x4 spatial cells in descriptor region
        static constexpr int NUM_BINS = 8;             ///< 8 orientation bins per cell
        static constexpr float SIGMA = 1.6f;           ///< Base Gaussian sigma for scale-space
        static constexpr float CONTRAST_THRESHOLD = 0.04f;  ///< Lowe's original threshold (kept for reference)

        // ─── Private SIFT Pipeline Stages ───────────────────────────────────
        
        /// @brief Build multi-scale Gaussian blur pyramid.
        /// Creates numOctaves pyramid levels, each with numScales+3 Gaussian blurs
        /// with progressively increasing sigma values.
        /// @param baseImage Input image for the pyramid (will be downsampled for each octave)
        /// @param gaussPyramid [OUT] 2D vector: [octave][scale] = blurred image
        /// @param numOctaves Number of octaves (pyramid levels)
        /// @param numScales Number of scales per octave
        static void buildGaussianPyramid(const cv::Mat &baseImage,
                                         std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                         int numOctaves,
                                         int numScales);
        
        /// @brief Build Difference-of-Gaussians pyramid for extrema detection.
        /// Computes DoG[octave][scale] = Gauss[octave][scale+1] - Gauss[octave][scale]
        /// extrema (local minima/maxima) in this pyramid are keypoint candidates.
        /// @param gaussPyramid Input Gaussian pyramid from buildGaussianPyramid
        /// @param dogPyramid [OUT] 2D vector: DoG images for extrema detection
        static void buildDoGPyramid(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                    std::vector<std::vector<cv::Mat>> &dogPyramid);
        
        /// @brief Detect scale-space extrema and apply contrast thresholding.
        /// Finds local minima and maxima in the DoG pyramid using 3D neighborhood checks
        /// (same scale, neighboring scales). Filters by contrast to eliminate weak features.
        /// @param dogPyramid Input DoG pyramid from buildDoGPyramid
        /// @param keypoints [OUT] Detected candidate keypoints at sub-pixel precision
        /// @param contrastThreshold Minimum response magnitude to retain keypoint
        /// @param numOctaves Octave count (for scale calculation)
        /// @param numScales Scales per octave
        static void detectExtrema(const std::vector<std::vector<cv::Mat>> &dogPyramid,
                                  std::vector<cv::KeyPoint> &keypoints,
                                  float contrastThreshold,
                                  int numOctaves,
                                  int numScales);
        
        /// @brief Assign dominant orientation(s) to each keypoint.
        /// Computes a histogram of image gradient directions in a circular region
        /// around each keypoint. The peak orientation becomes the keypoint's canonical orientation,
        /// enabling rotation-invariance in descriptor matching.
        /// @param gaussPyramid Input Gaussian pyramid (for gradient computation)
        /// @param keypoints [IN/OUT] Keypoints to augment with orientation information
        /// @param numOctaves Octave count
        static void assignOrientations(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                       std::vector<cv::KeyPoint> &keypoints,
                                       int numOctaves);
        
        /// @brief Compute 128-dimensional SIFT descriptors for all keypoints.
        /// For each keypoint, extracts a 16x16 patch (aligned to canonical orientation),
        /// divides it into 4x4 cells, computes gradient orientation histograms in each cell,
        /// and concatenates them into a 128-D vector. Descriptor is normalized to unit length.
        /// @param gaussPyramid Input Gaussian pyramid (for gradient computation)
        /// @param keypoints Input keypoints with location, scale, and orientation already set
        /// @param descriptors [OUT] 128-D descriptor for each keypoint (rows=count, cols=128, type=CV_32F)
        /// @param numOctaves Octave count
        static void computeDescriptors(const std::vector<std::vector<cv::Mat>> &gaussPyramid,
                                       std::vector<cv::KeyPoint> &keypoints,
                                       cv::Mat &descriptors,
                                       int numOctaves);
    };
}