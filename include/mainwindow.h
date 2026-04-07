#pragma once
/**
 * @file mainwindow.h
 * @brief MainWindow — 3-tab Vision Lab controller.
 *
 * Separation of Concerns
 * ──────────────────────
 *  • UI wiring lives only in setupConnections().
 *  • Each tab has its own private setup + slot group.
 *  • All parameters are driven by dynamic sliders — nothing hardcoded.
 *  • Heavy work runs on QtConcurrent threads; UI updates only in *Finished slots.
 *  • ZoomableImageLabel handles all zoom (mouse wheel) — no scroll-pan side effects.
 *  • Harris and Shi-Tomasi are independent: run only the chosen mode.
 */

#include <QMainWindow>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <mutex>

#include "widgets/interactive_label.h"
#include "model/image.hpp"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

// ─── Result bundles ───────────────────────────────────────────────────────────
struct CornerResult {
    std::vector<cv::Point> points;
    cv::Mat                annotated;
    int                    count   = 0;
    double                 timeMs  = 0.0;
    bool                   valid   = false;
    float                  thresholdUsed = 0.0f;
    float                  kUsed         = 0.0f;
    int                    halfWinUsed   = 0;
};

struct SiftResult {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat                   descriptors;
    cv::Mat                   annotated;
    double                    timeMs = 0.0;
    bool                      valid  = false;
    float                     contrastThreshold = 0.0f;
    int                       numOctaves = 0;
    int                       numScales  = 0;
};

struct MatchResult {
    std::vector<cv::DMatch> matches;
    cv::Mat                 composite;
    cv::Mat                 loadedAnnotated;
    int                     inliers = 0;
    double                  timeMs  = 0.0;
};

// ═════════════════════════════════════════════════════════════════════════════
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

private slots:
    // ── Shared ────────────────────────────────────────────────────────────
    void onLoadImage1();
    void onLoadImage2();

    // ── Tab 1: Corner Detection ────────────────────────────────────────────
    void onDetectCorners();
    void onCornerDetectionFinished();
    void onCornerModeChanged(int index);      // combo: Harris / Shi-Tomasi
    void onThresholdChanged(int v);
    void onHarrisKChanged(int v);
    void onNmsWindowChanged(int v);

    // ── Tab 2: SIFT Extraction ─────────────────────────────────────────────
    void onRunSift();
    void onSiftExtractionFinished();
    void onSiftContrastChanged(int v);
    void onSiftOctavesChanged(int v);         // dynamic: num octaves
    void onSiftScalesChanged(int v);          // dynamic: scales per octave

    // ── Tab 3: Feature Matching ────────────────────────────────────────────
    void onMatchFeatures();
    void onMatchingFinished();
    void onMatchRatioChanged(int v);
    void onClearROI();
    void onRemoveLastROI();

private:
    // ── Setup ──────────────────────────────────────────────────────────────
    void setupConnections();
    void setupTab1();
    void setupTab2();
    void setupTab3();
    void promoteMatchLabel();

    // ── Shared utilities ───────────────────────────────────────────────────
    static QPixmap matToPixmap(const cv::Mat& mat);
    void           downscaleIfNeeded(cv::Mat& img, int maxDim = 1024);
    void           setStatus(const QString& msg);
    cv::Mat        toGray(const cv::Mat& src);

    // ── Tab 1 helpers ──────────────────────────────────────────────────────
    void runHarris(float threshold, float k,
                   int halfWin, CornerResult& out);
    void runShiTomasi(float threshold,
                      int halfWin, CornerResult& out);
    void displayCornerResult(const CornerResult& result, const QString& modeName);

    // ── Tab 2 helpers ──────────────────────────────────────────────────────
    CornerResult& ensureCornersCached(const std::string& mode);

    // ── Tab 3 helpers ──────────────────────────────────────────────────────
    std::vector<cv::DMatch> matchSSD(const cv::Mat& desc1,
                                     const cv::Mat& desc2,
                                     float ratio) const;
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

    // Async
    QFutureWatcher<void> watcherCorners;
    QFutureWatcher<void> watcherSift;
    QFutureWatcher<void> watcherMatch;
};
