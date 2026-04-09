/**
 * @file mainwindow.h
 * @brief Main window class managing three tabs: corner detection, SIFT extraction, and feature matching.
 */

#pragma once
#include <QMainWindow>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <mutex>

#include "ui/interactive_label.h"
#include "model/image.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

/// @struct CornerResult
/// @brief Encapsulates corner detection results: points, visualization, and parameters.
struct CornerResult {
    std::vector<cv::Point> points;         ///< Detected corner pixel locations
    cv::Mat                annotated;      ///< Annotated image with corner markers drawn
    int                    count   = 0;    ///< Number of detected corners
    double                 timeMs  = 0.0;  ///< Execution time in milliseconds
    bool                   valid   = false; ///< True if detection succeeded
    float                  thresholdUsed = 0.0f; ///< Response threshold that was applied
    float                  kUsed         = 0.0f; ///< Harris k parameter (only for Harris mode)
    int                    halfWinUsed   = 0;    ///< NMS window half-width
};

/// @struct SiftResult
/// @brief Encapsulates SIFT feature extraction results: keypoints, descriptors, and parameters.
struct SiftResult {
    std::vector<cv::KeyPoint> keypoints;      ///< Detected SIFT keypoints (location, scale, orientation)
    cv::Mat                   descriptors;    ///< 128-D descriptor for each keypoint (rows=count, type=CV_32F)
    cv::Mat                   annotated;      ///< Annotated image with keypoint circles drawn
    double                    timeMs = 0.0;   ///< Execution time in milliseconds
    bool                      valid  = false; ///< True if extraction succeeded
    float                     contrastThreshold = 0.0f;  ///< DoG contrast threshold used
    int                       numOctaves = 0; ///< Number of scale octaves (pyramid depth)
    int                       numScales  = 0; ///< Number of scales per octave
};

/// @struct MatchResult
/// @brief Encapsulates feature matching results between two images.
struct MatchResult {
    std::vector<cv::DMatch> matches;      ///< Matched feature pairs (indices into descriptor arrays)
    cv::Mat                 composite;    ///< Side-by-side composite image with match lines drawn
    cv::Mat                 loadedAnnotated; ///< Loaded reference image annotation
    int                     inliers = 0;  ///< Number of geometrically consistent matches
    double                  timeMs  = 0.0; ///< Matching computation time in milliseconds
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    /// @brief Construct the main window and initialize all UI components.
    /// @param parent Parent widget (typically nullptr for top-level window)
    explicit MainWindow(QWidget* parent = nullptr);
    
    /// @brief Clean up resources and destroy the window.
    ~MainWindow() override;

private slots:
    // Shared utilities
    /// @brief Prompt user to select an image file and load it into img1.
    void onLoadImage1();
    
    /// @brief Prompt user to select an image file and load it into img2.
    void onLoadImage2();

    // Tab 1: Corner Detection
    /// @brief Launch corner detection (Harris or Shi-Tomasi) on img1 asynchronously.
    void onDetectCorners();
    
    /// @brief Handle completion of corner detection: update UI with results.
    void onCornerDetectionFinished();
    
    /// @brief Update corner detector mode when user changes combo selection.
    /// @param index 0=Harris, 1=Shi-Tomasi
    void onCornerModeChanged(int index);
    
    /// @brief Update cornerresponse threshold value from slider.
    /// @param v Threshold value from slider
    void onThresholdChanged(int v);
    
    /// @brief Update Harris k parameter from slider.
    /// @param v Harris k parameter
    void onHarrisKChanged(int v);
    
    /// @brief Update NMS window half-width from slider.
    /// @param v NMS half-window size
    void onNmsWindowChanged(int v);

    // ── Tab 2: SIFT Extraction ─────────────────────────────────────────────
    /// @brief Launch SIFT extraction on img1 asynchronously.
    void onRunSift();
    
    /// @brief Handle completion of SIFT extraction: update keypoint visualization.
    void onSiftExtractionFinished();
    
    /// @brief Update SIFT DoG contrast threshold from slider.
    /// @param v Contrast threshold value
    void onSiftContrastChanged(int v);
    
    /// @brief Update number of SIFT scale octaves from slider.
    /// @param v Number of octaves (1-5)
    void onSiftOctavesChanged(int v);
    
    /// @brief Update number of scales per octave from slider.
    /// @param v Scales per octave (2-5)
    void onSiftScalesChanged(int v);

    // ── Tab 3: Feature Matching ────────────────────────────────────────────
    /// @brief Launch feature matching between img1 and img2 asynchronously.
    void onMatchFeatures();
    
    /// @brief Handle completion of matching: update match visualization.
    void onMatchingFinished();
    
    /// @brief Update Lowe's ratio test threshold from slider.
    /// @param v Ratio threshold value
    void onMatchRatioChanged(int v);
    
    /// @brief Clear all ROI rectangles from the match label.
    void onClearROI();
    
    /// @brief Remove the most recently drawn ROI rectangle.
    void onRemoveLastROI();

private:
    // ── Setup ──────────────────────────────────────────────────────────────
    /// @brief Wire all signal-slot connections after UI initialization.
    void setupConnections();
    
    /// @brief Build Tab 1 (corner detection) widgets and layout.
    void setupTab1();
    
    /// @brief Build Tab 2 (SIFT extraction) widgets and layout.
    void setupTab2();
    
    /// @brief Build Tab 3 (feature matching) widgets and layout.
    void setupTab3();
    
    /// @brief Promote the match result label to an InteractiveLabel widget.
    void promoteMatchLabel();

    // Shared utilities───────
    /// @brief Convert OpenCV Mat to QPixmap for display.
    /// @param mat OpenCV image (any type and channels)
    /// @return Converted QPixmap ready for QLabel display
    static QPixmap matToPixmap(const cv::Mat& mat);
    
    /// @brief Downscale image if either dimension exceeds maxDim; modifies in-place.
    /// @param img Image to potentially downscale
    /// @param maxDim Target maximum dimension (default: 1024 pixels)
    void downscaleIfNeeded(cv::Mat& img, int maxDim = 1024);
    
    /// @brief Update the status bar message.
    /// @param msg Status text to display
    void setStatus(const QString& msg);
    
    /// @brief Convert image to grayscale (handles both color and already-grayscale inputs).
    /// @param src Source image
    /// @return Grayscale image (CV_8UC1)
    cv::Mat toGray(const cv::Mat& src);

    // Tab 1 helpers──────────
    /// @brief Execute Harris corner detection pipeline on img1.
    /// @param threshold Minimum response value for corner candidates
    /// @param k Harris k parameter (typically ~0.04)
    /// @param halfWin NMS half-window size
    /// @param out Result struct filled with corners, annotated image, and metadata
    void runHarris(float threshold, float k,
                   int halfWin, CornerResult& out);
    
    /// @brief Execute Shi-Tomasi corner detection pipeline on img1.
    /// @param threshold Minimum response value for corner candidates
    /// @param halfWin NMS half-window size
    /// @param out Result struct filled with corners, annotated image, and metadata
    void runShiTomasi(float threshold,
                      int halfWin, CornerResult& out);
    
    /// @brief Display corner detection results and timing on Tab 1.
    /// @param result CornerResult struct from detection
    /// @param modeName "Harris" or "Shi-Tomasi" for UI label
    void displayCornerResult(const CornerResult& result, const QString& modeName);

    // Tab 2 helpers──────────
    /// @brief Get or compute SIFT results for the given detector mode.
    /// @param mode "harris" or "sift" (corner preprocessing may be cached)
    /// @return Reference to cached SiftResult struct
    CornerResult& ensureCornersCached(const std::string& mode);

    // Tab 3 helpers──────────
    /// @brief Match descriptors using Sum-of-Squared-Differences (SSD) with ratio test.
    /// @param desc1 Descriptor matrix from image 1 (rows=count, cols=128, type=CV_32F)
    /// @param desc2 Descriptor matrix from image 2
    /// @param ratio Lowe's ratio threshold for ambiguous matches (default ~0.8)
    /// @return Vector of DMatch with indices into the descriptor arrays
    std::vector<cv::DMatch> matchSSD(const cv::Mat& desc1,
                                     const cv::Mat& desc2,
                                     float ratio) const;
    
    /// @brief Match descriptors using Normalized Cross-Correlation (NCC) similarity.
    /// @param desc1 Descriptor matrix from image 1
    /// @param desc2 Descriptor matrix from image 2
    /// @param minCorr Minimum correlation threshold for a valid match
    /// @return Vector of DMatch with indices into the descriptor arrays
    std::vector<cv::DMatch> matchNCC(const cv::Mat& desc1,
                                     const cv::Mat& desc2,
                                     float minCorr) const;

    // ═══════════════════════════════════
    // State
    // ═══════════════════════════════════
    Ui::MainWindow* ui;

    // Source images
    cv::Mat img1;
    cv::Mat img2;

    // Tab 1 state
    CornerResult harrisResult1;
    CornerResult shiTomasiResult1;
    float  p1_threshold    = 120.f;
    float  p1_harrisK      = 0.04f;
    int    p1_nmsHalfWin   = 3;
    int    p1_modeIndex    = 0;   // 0=Harris, 1=Shi-Tomasi
    int    p1_lastRunMode  = 0;
    Image  p1PipelineImage;
    std::mutex p1PipelineMutex;

    // Tab 2 state
    SiftResult siftResult1;
    SiftResult siftResult2;
    float p2_contrastThresh = 0.007f;
    int   p2_numOctaves     = 4;
    int   p2_numScales      = 3;

    // Tab 3 state
    MatchResult    matchResult;
    float          p3_ratioThresh = 0.80f;
    InteractiveLabel* roiLabel    = nullptr;
    std::vector<cv::DMatch> p3_ssdMatches;
    std::vector<cv::DMatch> p3_nccMatches;
    int p3_lastComputedMode = -1;
    int p3_ssdInliers = 0;
    int p3_nccInliers = 0;
    double p3_ssdTimeMs = 0.0;
    double p3_nccTimeMs = 0.0;
    cv::Mat p3_featureSSD;
    cv::Mat p3_featureNCC;
    cv::Mat p3_loadedSSD;
    cv::Mat p3_loadedNCC;

    // Async
    QFutureWatcher<void> watcherCorners;
    QFutureWatcher<void> watcherSift;
    QFutureWatcher<void> watcherMatch;
};
