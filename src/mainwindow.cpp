/**
 * @file mainwindow.cpp
 * @brief Vision Lab — MainWindow implementation.
 *
 * Tab 1 (Corner Detection)
 *   • Only runs the SELECTED mode (Harris OR Shi-Tomasi), not both.
 *   • Caches result; re-run only invalidates that mode's cache.
 *   • Shows: original | result | stats panel.
 *   • Both modes' stats shown if both have been run.
 *
 * Tab 2 (SIFT Extraction)
 *   • Parameters: contrast threshold, octaves, scales — all from sliders.
 *   • Auto-computes corners if cache missing.
 *   • Shows: original | corners | SIFT features (3-panel).
 *
 * Tab 3 (Feature Matching)
 *   • Shows: feature image (ROI drawable) | loaded image | match result.
 *   • ROI: draw multiple, remove last, clear all.
 *   • Match projects loaded-image region onto feature image via descriptors.
 *
 * Thread safety: all cv::Mat writes inside worker lambdas; all widget writes
 * only in *Finished slots (GUI thread).
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "SiftCore.hpp"
#include "widgets/interactive_label.h"
#include "io/image_handler.hpp"
#include "processors/harris_main.hpp"
#include "model/image.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>

#include <algorithm>

#include <chrono>
#include <cmath>

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

static constexpr int MAX_DIM = 1024;

// ─── Utility: Mat → QPixmap (no widget touches) ──────────────────────────────
QPixmap MainWindow::matToPixmap(const cv::Mat& mat)
{
    if (mat.empty()) return {};
    cv::Mat rgb;
    if (mat.channels() == 1)      cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    else if (mat.channels() == 3) cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else return {};
    QImage qi(rgb.data, rgb.cols, rgb.rows,
               static_cast<int>(rgb.step), QImage::Format_RGB888);
    return QPixmap::fromImage(qi.copy());
}

void MainWindow::downscaleIfNeeded(cv::Mat& img, int maxDim)
{
    if (img.cols > maxDim || img.rows > maxDim) {
        float s = std::min(static_cast<float>(maxDim) / img.cols,
                           static_cast<float>(maxDim) / img.rows);
        cv::resize(img, img, {}, s, s, cv::INTER_AREA);
    }
}

void MainWindow::setStatus(const QString& msg)
{
    statusBar()->showMessage(msg);
}

cv::Mat MainWindow::toGray(const cv::Mat& src)
{
    cv::Mat g;
    if (src.channels() == 3) cv::cvtColor(src, g, cv::COLOR_BGR2GRAY);
    else g = src.clone();
    return g;
}

// =============================================================================
//  Constructor / Destructor
// =============================================================================
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setupTab1();
    setupTab2();
    setupTab3();
    setupConnections();
    setStatus("Ready — load Image 1 to begin.");
}

MainWindow::~MainWindow()
{
    if (watcherCorners.isRunning()) watcherCorners.cancel();
    if (watcherSift.isRunning())    watcherSift.cancel();
    if (watcherMatch.isRunning())   watcherMatch.cancel();
    watcherCorners.waitForFinished();
    watcherSift.waitForFinished();
    watcherMatch.waitForFinished();
    delete ui;
}

// ─── Tab 1 initial state ──────────────────────────────────────────────────────
void MainWindow::setupTab1()
{
    // Sync slider display labels to default values
    ui->cornerThresholdVal->setText(QString::number(static_cast<int>(p1_threshold)));
    ui->harrisKVal->setText(QString::number(p1_harrisK, 'f', 2));
    ui->nmsWindowVal->setText(QString::number(p1_nmsHalfWin));

    ui->cornerThresholdSlider->setValue(static_cast<int>(p1_threshold));
    ui->harrisKSlider->setValue(static_cast<int>(p1_harrisK * 100));
    ui->nmsWindowSlider->setValue(p1_nmsHalfWin);

    ui->btnDetectCorners->setEnabled(false);
    ui->cornerDisplayOriginal->setText("Load Image 1");
    ui->cornerDisplayResult->setText("Run detection");
    ui->cornerDisplayOriginal->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->cornerDisplayResult->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->cornerStatsHarris->setMinimumWidth(280);
    ui->cornerStatsShiTomasi->setMinimumWidth(280);
    ui->cornerStatsHarris->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    ui->cornerStatsShiTomasi->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    if (ui->tab1Root) ui->tab1Root->setStretch(0, 0), ui->tab1Root->setStretch(1, 1);
    if (ui->tab1ImagesLayout) {
        ui->tab1ImagesLayout->setStretch(0, 1);
        ui->tab1ImagesLayout->setStretch(1, 1);
    }
}

// ─── Tab 2 initial state ──────────────────────────────────────────────────────
void MainWindow::setupTab2()
{
    ui->siftContrastVal->setText(QString::number(p2_contrastThresh, 'f', 3));
    ui->siftOctavesVal->setText(QString::number(p2_numOctaves));
    ui->siftScalesVal->setText(QString::number(p2_numScales));

    ui->siftContrastSlider->setValue(static_cast<int>(p2_contrastThresh * 1000));
    ui->siftOctavesSlider->setValue(p2_numOctaves);
    ui->siftScalesSlider->setValue(p2_numScales);

    ui->btnRunSift->setEnabled(false);
    ui->siftDisplayOriginal->setText("Load Image 1");
    ui->siftDisplayCorners->setText("Corners appear here");
    ui->siftDisplayFeatures->setText("SIFT features appear here");
    ui->siftDisplayOriginal->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->siftDisplayCorners->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->siftDisplayFeatures->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->siftInfo->setMinimumWidth(280);
    ui->siftInfo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    if (ui->tab2Root) ui->tab2Root->setStretch(0, 0), ui->tab2Root->setStretch(1, 1);
    if (ui->tab2ImagesLayout) {
        ui->tab2ImagesLayout->setStretch(0, 1);
        ui->tab2ImagesLayout->setStretch(1, 1);
        ui->tab2ImagesLayout->setStretch(2, 1);
    }
}

// ─── Tab 3 initial state ──────────────────────────────────────────────────────
void MainWindow::setupTab3()
{
    ui->matchRatioVal->setText(QString::number(p3_ratioThresh, 'f', 2));
    ui->matchRatioSlider->setValue(static_cast<int>(p3_ratioThresh * 100));

    ui->btnMatchFeatures->setEnabled(false);
    roiLabel = ui->matchDisplayFeature;
    if (roiLabel) {
        roiLabel->setScaledContents(true);
        roiLabel->setAlignment(Qt::AlignCenter);
        roiLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        roiLabel->setMinimumSize(0, 0);
        roiLabel->setText("Run SIFT first (Tab 2)");
    }
    ui->matchDisplayLoaded->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->matchDisplayLoaded->setMinimumSize(0, 0);
    if (ui->matchScrollFeature) {
        ui->matchScrollFeature->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
        ui->matchScrollFeature->setMinimumSize(0, 0);
    }
    ui->matchInfo->setMinimumWidth(320);
    ui->matchInfo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    ui->matchDisplayLoaded->setText("Load Image 2");
    if (ui->matchMethodCombo) {
        ui->matchMethodCombo->setCurrentIndex(0);
    }
    ui->matchROIInfo->setText("Draw ROI on the feature image (left panel)");
    if (ui->tab3Root) ui->tab3Root->setStretch(0, 0), ui->tab3Root->setStretch(1, 1);
    if (ui->tab3ImagesLayout) {
        ui->tab3ImagesLayout->setStretch(0, 1);
        ui->tab3ImagesLayout->setStretch(1, 1);
    }
}

// ─── Wire all signals ─────────────────────────────────────────────────────────
void MainWindow::setupConnections()
{
    // Shared
    connect(ui->btnLoadImage1, &QPushButton::clicked, this, &MainWindow::onLoadImage1);
    connect(ui->btnLoadImage2, &QPushButton::clicked, this, &MainWindow::onLoadImage2);

    // Tab 1
    connect(ui->cornerModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onCornerModeChanged);
    connect(ui->cornerThresholdSlider, &QSlider::valueChanged,
            this, &MainWindow::onThresholdChanged);
    connect(ui->harrisKSlider, &QSlider::valueChanged,
            this, &MainWindow::onHarrisKChanged);
    connect(ui->nmsWindowSlider, &QSlider::valueChanged,
            this, &MainWindow::onNmsWindowChanged);
    connect(ui->btnDetectCorners, &QPushButton::clicked,
            this, &MainWindow::onDetectCorners);
    connect(&watcherCorners, &QFutureWatcher<void>::finished,
            this, &MainWindow::onCornerDetectionFinished);

    // Tab 2
    connect(ui->siftContrastSlider, &QSlider::valueChanged,
            this, &MainWindow::onSiftContrastChanged);
    connect(ui->siftOctavesSlider, &QSlider::valueChanged,
            this, &MainWindow::onSiftOctavesChanged);
    connect(ui->siftScalesSlider, &QSlider::valueChanged,
            this, &MainWindow::onSiftScalesChanged);
    connect(ui->btnRunSift, &QPushButton::clicked,
            this, &MainWindow::onRunSift);
    connect(&watcherSift, &QFutureWatcher<void>::finished,
            this, &MainWindow::onSiftExtractionFinished);

    // Tab 3
    connect(ui->matchRatioSlider, &QSlider::valueChanged,
            this, &MainWindow::onMatchRatioChanged);
    connect(ui->btnMatchFeatures, &QPushButton::clicked,
            this, &MainWindow::onMatchFeatures);
    connect(ui->btnClearROI, &QPushButton::clicked,
            this, &MainWindow::onClearROI);
    connect(ui->btnRemoveLastROI, &QPushButton::clicked,
            this, &MainWindow::onRemoveLastROI);
    connect(&watcherMatch, &QFutureWatcher<void>::finished,
            this, &MainWindow::onMatchingFinished);

    // Tab changes — refresh displays
    connect(ui->tabWidget, QOverload<int>::of(&QTabWidget::currentChanged),
            this, [this](int idx) {
                try {
                    if (idx == 1) {  // Tab 2 (SIFT) selected
                        // Display corners from selected mode
                        if (!ui || !ui->siftCornerModeCombo || !ui->siftDisplayCorners) return;
                        const bool isHarris = (ui->siftCornerModeCombo->currentIndex() == 0);
                        const CornerResult& cr = isHarris ? harrisResult1 : shiTomasiResult1;
                        if (cr.valid && !cr.annotated.empty()) {
                            ui->siftDisplayCorners->setImage(matToPixmap(cr.annotated));
                        }
                    } else if (idx == 2) {  // Tab 3 (Feature Matching) selected
                        // Display SIFT results if available and roiLabel exists
                        if (siftResult1.valid && !siftResult1.annotated.empty()) {
                            if (roiLabel && ui->matchROIInfo) {
                                roiLabel->setPixmap(matToPixmap(siftResult1.annotated));
                                ui->matchROIInfo->setText("Draw ROI on the feature image (left panel)");
                                if (ui->btnMatchFeatures) {
                                    ui->btnMatchFeatures->setEnabled(!img2.empty());
                                }
                            }
                        }
                    }
                } catch (...) {
                    // Silently ignore tab change errors
                }
            });
}

// ─── Promote plain QLabel → InteractiveLabel for ROI drawing ─────────────────
void MainWindow::promoteMatchLabel()
{
    roiLabel = ui->matchDisplayFeature;
    if (roiLabel) {
        roiLabel->setScaledContents(true);
        roiLabel->setAlignment(Qt::AlignCenter);
    }
}

// =============================================================================
//  FILE LOADING
// =============================================================================

void MainWindow::onLoadImage1()
{
    QString fn = QFileDialog::getOpenFileName(
        this, "Open Image 1", {}, "Images (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (fn.isEmpty()) return;

    try {
        img1 = loadImage(fn.toStdString()).mat;
        downscaleIfNeeded(img1);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Load Error", e.what());
        return;
    }

    // Invalidate all cached results that depend on img1
    harrisResult1   = {};
    shiTomasiResult1 = {};
    siftResult1     = {};
    siftResult2     = {};   // descriptors from img1 feed into matching
    matchResult     = {};
    {
        std::lock_guard<std::mutex> lock(p1PipelineMutex);
        p1PipelineImage = {};
        p1PipelineImage.mat = img1.clone();
        p1PipelineImage.clearCache();
    }

    const QPixmap px = matToPixmap(img1);

    // Tab 1 - ensure images fit properly
    ui->cornerDisplayOriginal->setImage(px);
    ui->cornerDisplayResult->setText("Run detection");
    ui->cornerStatsHarris->setText("Harris  —  not run");
    ui->cornerStatsShiTomasi->setText("Shi-Tomasi  —  not run");
    ui->btnDetectCorners->setEnabled(true);

    // Tab 2
    ui->siftDisplayOriginal->setImage(px);
    ui->siftDisplayCorners->setText("Corners appear here");
    ui->siftDisplayFeatures->setText("SIFT features appear here");
    ui->siftInfo->setText("Keypoints: —  |  Time: — ms");
    ui->btnRunSift->setEnabled(true);

    // Display cached corner results if they exist
    if (harrisResult1.valid && !harrisResult1.annotated.empty()) {
        ui->siftDisplayCorners->setImage(matToPixmap(harrisResult1.annotated));
    } else if (shiTomasiResult1.valid && !shiTomasiResult1.annotated.empty()) {
        ui->siftDisplayCorners->setImage(matToPixmap(shiTomasiResult1.annotated));
    }

    // Tab 3 — reset feature panel
    if (roiLabel) {
        roiLabel->setText("Run SIFT first (Tab 2)");
        roiLabel->clearROI();
    }
    ui->btnMatchFeatures->setEnabled(false);

    setStatus(QString("Image 1 loaded  %1 × %2").arg(img1.cols).arg(img1.rows));
}

void MainWindow::onLoadImage2()
{
    QString fn = QFileDialog::getOpenFileName(
        this, "Open Image 2 (to match)", {}, "Images (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (fn.isEmpty()) return;

    try {
        img2 = loadImage(fn.toStdString()).mat;
        downscaleIfNeeded(img2);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Load Error", e.what());
        return;
    }

    // Invalidate sift cache for img2 so it re-runs on next match
    siftResult2 = {};
    matchResult.loadedAnnotated = cv::Mat();

    ui->matchDisplayLoaded->setImage(matToPixmap(img2));

    // Enable matching if SIFT is already done
    if (siftResult1.valid && !img2.empty())
        ui->btnMatchFeatures->setEnabled(true);

    setStatus(QString("Image 2 loaded  %1 × %2").arg(img2.cols).arg(img2.rows));
}

// =============================================================================
//  TAB 1 — CORNER DETECTION
// =============================================================================

void MainWindow::onCornerModeChanged(int index)
{
    p1_modeIndex = index;
}

void MainWindow::onThresholdChanged(int v)
{
    p1_threshold = static_cast<float>(v);
    ui->cornerThresholdVal->setText(QString::number(v));
}

void MainWindow::onHarrisKChanged(int v)
{
    p1_harrisK = v / 100.0f;
    ui->harrisKVal->setText(QString::number(p1_harrisK, 'f', 2));
}

void MainWindow::onNmsWindowChanged(int v)
{
    p1_nmsHalfWin = v;
    ui->nmsWindowVal->setText(QString::number(v));
}

// ─── Harris helper (worker thread) ───────────────────────────────────────────
void MainWindow::runHarris(const cv::Mat& gray, float threshold, float k,
                            int halfWin, CornerResult& out)
{
    auto t0 = Clock::now();
    {
        std::lock_guard<std::mutex> lock(p1PipelineMutex);
        if (p1PipelineImage.mat.empty()) {
            p1PipelineImage.mat = img1.clone();
            p1PipelineImage.clearCache();
        }
        out.points = applyHarris(p1PipelineImage, k, "harris", threshold, halfWin);
    }
    out.count   = static_cast<int>(out.points.size());
    out.timeMs  = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    out.valid   = true;
    out.thresholdUsed = threshold;
    out.kUsed         = k;
    out.halfWinUsed   = halfWin;

    // Draw on original color image if available, otherwise convert from grayscale
    cv::Mat vis;
    if (img1.channels() == 3) {
        vis = img1.clone();
    } else if (img1.channels() == 1) {
        cv::cvtColor(img1, vis, cv::COLOR_GRAY2BGR);
    } else {
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    }
    for (const auto& pt : out.points)
        cv::circle(vis, pt, 4, cv::Scalar(0, 80, 255), 3, cv::LINE_AA);
    out.annotated = vis;
}

// ─── Shi-Tomasi helper (worker thread) ───────────────────────────────────────
void MainWindow::runShiTomasi(const cv::Mat& gray, float threshold,
                               int halfWin, CornerResult& out)
{
    auto t0 = Clock::now();
    {
        std::lock_guard<std::mutex> lock(p1PipelineMutex);
        if (p1PipelineImage.mat.empty()) {
            p1PipelineImage.mat = img1.clone();
            p1PipelineImage.clearCache();
        }
        out.points = applyHarris(p1PipelineImage, 0.f, "shi_tomasi", threshold, halfWin);
    }
    out.count   = static_cast<int>(out.points.size());
    out.timeMs  = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    out.valid   = true;
    out.thresholdUsed = threshold;
    out.kUsed         = 0.f;
    out.halfWinUsed   = halfWin;

    // Draw on original color image if available, otherwise convert from grayscale
    cv::Mat vis;
    if (img1.channels() == 3) {
        vis = img1.clone();
    } else if (img1.channels() == 1) {
        cv::cvtColor(img1, vis, cv::COLOR_GRAY2BGR);
    } else {
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    }
    for (const auto& pt : out.points)
        cv::circle(vis, pt, 4, cv::Scalar(50, 220, 80), 3, cv::LINE_AA);
    out.annotated = vis;
}

// ─── Run only the chosen mode ─────────────────────────────────────────────────
void MainWindow::onDetectCorners()
{
    if (img1.empty()) { setStatus("Load Image 1 first."); return; }

    ui->btnDetectCorners->setEnabled(false);
    ui->cornerDisplayResult->setText("Computing…");
    QApplication::processEvents();

    const float  thresh  = p1_threshold;
    const float  k       = p1_harrisK;
    const int    halfWin = p1_nmsHalfWin;
    const int    mode    = p1_modeIndex;
    const cv::Mat img1_copy = img1.clone();

    p1_lastRunMode = mode;

    auto cacheMatches = [&](const CornerResult& result) {
        if (!result.valid || result.annotated.empty()) return false;
        if (result.thresholdUsed != thresh || result.halfWinUsed != halfWin) return false;
        if (mode == 0) return result.kUsed == k;
        return true;
    };

    if ((mode == 0 && cacheMatches(harrisResult1)) || (mode != 0 && cacheMatches(shiTomasiResult1))) {
        const CornerResult& cached = (mode == 0) ? harrisResult1 : shiTomasiResult1;
        ui->cornerDisplayResult->setImage(matToPixmap(cached.annotated));
        ui->cornerStatsHarris->setText(QString("Harris  |  Corners: %1  |  Time: %2 ms")
                                       .arg(harrisResult1.count)
                                       .arg(harrisResult1.timeMs, 0, 'f', 1));
        ui->cornerStatsShiTomasi->setText(QString("Shi-Tomasi  |  Corners: %1  |  Time: %2 ms")
                                          .arg(shiTomasiResult1.count)
                                          .arg(shiTomasiResult1.timeMs, 0, 'f', 1));
        if (ui->siftDisplayCorners) {
            ui->siftDisplayCorners->setImage(matToPixmap(cached.annotated));
        }
        setStatus(QString("%1 detection — reused cached result").arg(mode == 0 ? "Harris" : "Shi-Tomasi"));
        ui->btnDetectCorners->setEnabled(true);
        return;
    }

    watcherCorners.setFuture(QtConcurrent::run([this, thresh, k, halfWin, mode, img1_copy]() {
        try {
            cv::Mat gray = toGray(img1_copy);
            if (gray.empty()) {
                throw std::runtime_error("Failed to convert to grayscale");
            }

            if (mode == 0) {
                runHarris(gray, thresh, k, halfWin, harrisResult1);
            } else {
                runShiTomasi(gray, thresh, halfWin, shiTomasiResult1);
            }
        } catch (const std::exception& e) {
            qWarning() << "Corner detection error:" << e.what();
            if (mode == 0)
                harrisResult1 = {};
            else
                shiTomasiResult1 = {};
        } catch (...) {
            qWarning() << "Corner detection: unknown error";
            if (mode == 0)
                harrisResult1 = {};
            else
                shiTomasiResult1 = {};
        }
    }));
}

// ─── Update UI after detection ────────────────────────────────────────────────
void MainWindow::onCornerDetectionFinished()
{
    const bool isHarris = (p1_lastRunMode == 0);
    const CornerResult& cr = isHarris ? harrisResult1 : shiTomasiResult1;
    const QString modeName = isHarris ? "Harris" : "Shi-Tomasi";

    if (!cr.annotated.empty())
        ui->cornerDisplayResult->setImage(matToPixmap(cr.annotated));
    else
        ui->cornerDisplayResult->setText("Detection failed");

    // Update stats for the mode that just ran
    auto statsText = [](const QString& name, const CornerResult& r) -> QString {
        if (!r.valid) return name + "  —  not run";
        return QString("%1  |  Corners: %2  |  Time: %3 ms")
            .arg(name).arg(r.count).arg(r.timeMs, 0, 'f', 1);
    };

    ui->cornerStatsHarris->setText(statsText("Harris", harrisResult1));
    ui->cornerStatsShiTomasi->setText(statsText("Shi-Tomasi", shiTomasiResult1));

    // Auto-display corners in Tab 2 if visible
    if (!cr.annotated.empty()) {
        ui->siftDisplayCorners->setImage(matToPixmap(cr.annotated));
    }

    setStatus(QString("%1 detection — %2 corners in %3 ms")
        .arg(modeName).arg(cr.count).arg(cr.timeMs, 0, 'f', 1));

    ui->btnDetectCorners->setEnabled(true);
}

// =============================================================================
//  TAB 2 — SIFT EXTRACTION
// =============================================================================

void MainWindow::onSiftContrastChanged(int v)
{
    p2_contrastThresh = v / 1000.0f;
    ui->siftContrastVal->setText(QString::number(p2_contrastThresh, 'f', 3));
}

void MainWindow::onSiftOctavesChanged(int v)
{
    p2_numOctaves = v;
    ui->siftOctavesVal->setText(QString::number(v));
}

void MainWindow::onSiftScalesChanged(int v)
{
    p2_numScales = v;
    ui->siftScalesVal->setText(QString::number(v));
}

// ─── Ensure corners cached (called inside worker thread) ─────────────────────
CornerResult& MainWindow::ensureCornersCached(const std::string& mode)
{
    if (img1.empty()) {
        throw std::runtime_error("ensureCornersCached: img1 is empty");
    }

    if (mode == "harris") {
        if (!harrisResult1.valid || harrisResult1.thresholdUsed != p1_threshold ||
            harrisResult1.kUsed != p1_harrisK || harrisResult1.halfWinUsed != p1_nmsHalfWin) {
            try {
                cv::Mat gray = toGray(img1);
                if (gray.empty()) {
                    throw std::runtime_error("ensureCornersCached: grayscale conversion failed");
                }
                runHarris(gray, p1_threshold, p1_harrisK, p1_nmsHalfWin, harrisResult1);
            } catch (const std::exception& e) {
                qWarning() << "Harris detection failed:" << e.what();
                harrisResult1 = {};
                throw;
            }
        }
        return harrisResult1;
    } else {
        if (!shiTomasiResult1.valid || shiTomasiResult1.thresholdUsed != p1_threshold ||
            shiTomasiResult1.halfWinUsed != p1_nmsHalfWin) {
            try {
                cv::Mat gray = toGray(img1);
                if (gray.empty()) {
                    throw std::runtime_error("ensureCornersCached: grayscale conversion failed");
                }
                runShiTomasi(gray, p1_threshold, p1_nmsHalfWin, shiTomasiResult1);
            } catch (const std::exception& e) {
                qWarning() << "Shi-Tomasi detection failed:" << e.what();
                shiTomasiResult1 = {};
                throw;
            }
        }
        return shiTomasiResult1;
    }
}

void MainWindow::onRunSift()
{
    if (img1.empty()) { setStatus("Load Image 1 first."); return; }

    ui->btnRunSift->setEnabled(false);
    ui->siftDisplayFeatures->setText("Extracting SIFT features…");
    QApplication::processEvents();

    const int cornerMode = ui->siftCornerModeCombo ? ui->siftCornerModeCombo->currentIndex() : 0;
    try {
        const CornerResult& cachedCorners = ensureCornersCached(cornerMode == 0 ? "harris" : "shi_tomasi");
        if (!cachedCorners.annotated.empty()) {
            ui->siftDisplayCorners->setImage(matToPixmap(cachedCorners.annotated));
        }
    } catch (const std::exception& e) {
        qWarning() << "Corner cache before SIFT failed:" << e.what();
    }

    const float       contrast  = p2_contrastThresh;
    const int   numOctaves = p2_numOctaves;
    const int   numScales  = p2_numScales;
    
    const cv::Mat img1_copy = img1.clone();

    siftResult1 = {};

    watcherSift.setFuture(QtConcurrent::run([this, contrast, numOctaves, numScales, img1_copy]() {
        auto t0 = Clock::now();
        SiftResult tempResult = {};
        try {
            // SIFT extraction only - no corner detection in worker
            tempResult.keypoints.clear();
            tempResult.descriptors.release();
            cv_assign::SiftProcessor::extractFeatures(
                img1_copy, tempResult.keypoints, tempResult.descriptors, contrast, numOctaves, numScales);

            // Annotate keypoints
            cv::drawKeypoints(img1_copy, tempResult.keypoints, tempResult.annotated,
                cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            tempResult.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            tempResult.valid = true;
            tempResult.contrastThreshold = contrast;
            tempResult.numOctaves = numOctaves;
            tempResult.numScales = numScales;
            
            siftResult1 = tempResult;
        } catch (const std::exception& e) {
            qWarning() << "SIFT extraction error:" << e.what();
            siftResult1 = {};
        } catch (...) {
            qWarning() << "SIFT extraction: unknown error";
            siftResult1 = {};
        }
    }));
}

void MainWindow::onSiftExtractionFinished()
{
    // Right panel: SIFT keypoints
    if (siftResult1.valid && !siftResult1.annotated.empty()) {
        ui->siftDisplayFeatures->setImage(matToPixmap(siftResult1.annotated));
        ui->siftInfo->setText(
            QString("Keypoints: %1  |  Time: %2 ms")
            .arg(siftResult1.keypoints.size())
            .arg(siftResult1.timeMs, 0, 'f', 1));

        // Try to update Tab 3 feature panel (safe attempt)
        if (roiLabel) {
            roiLabel->setPixmap(matToPixmap(siftResult1.annotated));
            ui->matchROIInfo->setText("Draw ROI on the feature image (left panel)");
        }
        ui->btnMatchFeatures->setEnabled(!img2.empty());
    } else {
        ui->siftDisplayFeatures->setText("SIFT extraction failed");
        ui->siftInfo->setText("SIFT failed");
    }

    setStatus(siftResult1.valid
        ? QString("SIFT done — %1 keypoints in %2 ms")
              .arg(siftResult1.keypoints.size())
              .arg(siftResult1.timeMs, 0, 'f', 1)
        : "SIFT extraction failed.");

    ui->btnRunSift->setEnabled(true);
}

// =============================================================================
//  TAB 3 — FEATURE MATCHING
// =============================================================================

void MainWindow::onMatchRatioChanged(int v)
{
    p3_ratioThresh = v / 100.0f;
    ui->matchRatioVal->setText(QString::number(p3_ratioThresh, 'f', 2));
}

void MainWindow::onClearROI()
{
    if (roiLabel) roiLabel->clearROI();
    ui->matchROIInfo->setText("ROI cleared — draw a new one.");
}

void MainWindow::onRemoveLastROI()
{
    if (roiLabel) roiLabel->removeLastROI();
    auto rois = roiLabel ? roiLabel->getSelectedROIs() : std::vector<QRect>{};
    ui->matchROIInfo->setText(rois.empty()
        ? "All ROIs removed — draw a new one."
        : QString("ROIs: %1  — draw more, Remove Last, or Clear All").arg(rois.size()));
}

// ─── SSD matching with Lowe ratio test ───────────────────────────────────────
std::vector<cv::DMatch>
MainWindow::matchSSD(const cv::Mat& desc1, const cv::Mat& desc2, float ratio) const
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty()) return good;

    for (int i = 0; i < desc1.rows; ++i) {
        const float* q = desc1.ptr<float>(i);
        float best1 = 1e30f, best2 = 1e30f;
        int   idx1  = -1;

        for (int j = 0; j < desc2.rows; ++j) {
            const float* t = desc2.ptr<float>(j);
            float ssd = 0.f;
            for (int d = 0; d < 128; ++d) { float diff = q[d]-t[d]; ssd += diff*diff; }
            if (ssd < best1) { best2 = best1; best1 = ssd; idx1 = j; }
            else if (ssd < best2) best2 = ssd;
        }

        if (idx1 >= 0 && std::sqrt(best1) < ratio * std::sqrt(best2))
            good.emplace_back(i, idx1, std::sqrt(best1));
    }
    return good;
}

std::vector<cv::DMatch>
MainWindow::matchNCC(const cv::Mat& desc1, const cv::Mat& desc2, float minCorr) const
{
    std::vector<cv::DMatch> good;
    if (desc1.empty() || desc2.empty()) return good;

    for (int i = 0; i < desc1.rows; ++i) {
        const float* q = desc1.ptr<float>(i);
        float qMean = 0.f;
        for (int d = 0; d < 128; ++d) qMean += q[d];
        qMean /= 128.f;

        float qNorm = 0.f;
        for (int d = 0; d < 128; ++d) {
            const float v = q[d] - qMean;
            qNorm += v * v;
        }
        if (qNorm <= 1e-12f) continue;
        qNorm = std::sqrt(qNorm);

        float best = -2.f;
        int bestIdx = -1;

        for (int j = 0; j < desc2.rows; ++j) {
            const float* t = desc2.ptr<float>(j);
            float tMean = 0.f;
            for (int d = 0; d < 128; ++d) tMean += t[d];
            tMean /= 128.f;

            float tNorm = 0.f;
            float dot = 0.f;
            for (int d = 0; d < 128; ++d) {
                const float qv = q[d] - qMean;
                const float tv = t[d] - tMean;
                dot += qv * tv;
                tNorm += tv * tv;
            }
            if (tNorm <= 1e-12f) continue;

            const float ncc = dot / (qNorm * std::sqrt(tNorm));
            if (ncc > best) {
                best = ncc;
                bestIdx = j;
            }
        }

        if (bestIdx >= 0 && best >= minCorr) {
            good.emplace_back(i, bestIdx, 1.0f - best);
        }
    }
    return good;
}

void MainWindow::onMatchFeatures()
{
    if (!siftResult1.valid) { setStatus("Run SIFT (Tab 2) first."); return; }
    if (img2.empty())        { setStatus("Load Image 2 first."); return; }
    if (img1.empty())        { setStatus("Image 1 missing."); return; }
    if (siftResult1.contrastThreshold != p2_contrastThresh ||
        siftResult1.numOctaves != p2_numOctaves ||
        siftResult1.numScales != p2_numScales) {
        setStatus("SIFT cache is stale. Run SIFT again with the current parameters.");
        return;
    }

    ui->btnMatchFeatures->setEnabled(false);
    QApplication::processEvents();

    // Capture state for worker - PASS COPIES to avoid race conditions
    std::vector<QRect> rois = roiLabel ? roiLabel->getSelectedROIs()
                                       : std::vector<QRect>{};
    const float ratio = p3_ratioThresh;
    const int methodIndex = ui->matchMethodCombo ? ui->matchMethodCombo->currentIndex() : 0;
    const cv::Mat img1_copy = img1.clone();
    const cv::Mat img2_copy = img2.clone();
    const float contrastThresh = p2_contrastThresh;
    const std::vector<cv::KeyPoint> sift1_kp = siftResult1.keypoints;
    const cv::Mat sift1_desc = siftResult1.descriptors.clone();

    watcherMatch.setFuture(QtConcurrent::run([this, rois, ratio, methodIndex, img1_copy, img2_copy,
                                              contrastThresh, sift1_kp, sift1_desc]() {
        auto t0 = Clock::now();
        try {
            // Validate input SIFT data before proceeding
            if (sift1_kp.empty() || sift1_desc.empty()) {
                qWarning() << "Feature matching: SIFT data is invalid or empty";
                matchResult.composite = cv::Mat();
                matchResult.loadedAnnotated = cv::Mat();
                matchResult.inliers   = 0;
                matchResult.matches.clear();
                matchResult.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                return;
            }

            // Step 1: SIFT on img2 if not cached
            std::vector<cv::KeyPoint> sift2_kp;
            cv::Mat sift2_desc;
            const bool sift2CacheMatches = siftResult2.valid &&
                                           !siftResult2.descriptors.empty() &&
                                           siftResult2.contrastThreshold == contrastThresh &&
                                           siftResult2.numOctaves == p2_numOctaves &&
                                           siftResult2.numScales == p2_numScales;
            if (!sift2CacheMatches) {
                cv_assign::SiftProcessor::extractFeatures(
                    img2_copy, sift2_kp, sift2_desc,
                    contrastThresh, p2_numOctaves, p2_numScales);
                siftResult2.contrastThreshold = contrastThresh;
                siftResult2.numOctaves = p2_numOctaves;
                siftResult2.numScales = p2_numScales;
            } else {
                sift2_kp = siftResult2.keypoints;
                sift2_desc = siftResult2.descriptors.clone();
            }
            
            // Update siftResult2 safely
            siftResult2.keypoints = sift2_kp;
            siftResult2.descriptors = sift2_desc;
            siftResult2.valid = true;

            // Step 2: filter img1 keypoints to ROI(s)
            cv::Mat                   queryDesc = sift1_desc;
            std::vector<cv::KeyPoint> queryKp   = sift1_kp;

            if (!rois.empty()) {
                std::vector<cv::KeyPoint> filtKp;
                std::vector<int>          filtIdx;
                for (int i = 0; i < static_cast<int>(queryKp.size()); ++i) {
                    const cv::Point2f& pt = queryKp[i].pt;
                    for (const QRect& r : rois) {
                        if (r.contains(static_cast<int>(pt.x), static_cast<int>(pt.y))) {
                            filtKp.push_back(queryKp[i]);
                            filtIdx.push_back(i);
                            break;
                        }
                    }
                }
                if (!filtIdx.empty()) {
                    queryKp  = filtKp;
                    queryDesc = cv::Mat(static_cast<int>(filtIdx.size()), 128, CV_32F);
                    for (int i = 0; i < static_cast<int>(filtIdx.size()); ++i)
                        sift1_desc.row(filtIdx[i]).copyTo(queryDesc.row(i));
                }
            }

            // Step 3: match (feature image → loaded image)
            if (methodIndex == 1) {
                matchResult.matches = matchNCC(queryDesc, sift2_desc, ratio);
            } else {
                matchResult.matches = matchSSD(queryDesc, sift2_desc, ratio);
            }
            matchResult.inliers = static_cast<int>(matchResult.matches.size());

            // Step 4: keep tab panels separate (like SIFT): draw overlays per panel
            auto toBgr = [](const cv::Mat& src) {
                cv::Mat out;
                if (src.empty()) return out;
                if (src.channels() == 3) out = src.clone();
                else cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
                return out;
            };

            cv::Mat featurePreview = toBgr(siftResult1.annotated.empty() ? img1_copy : siftResult1.annotated);
            cv::Mat loadedPreview = toBgr(img2_copy);

            for (const QRect& roi : rois) {
                cv::rectangle(featurePreview,
                              cv::Rect(roi.x(), roi.y(), std::max(1, roi.width()), std::max(1, roi.height())),
                              cv::Scalar(46, 204, 113), 2, cv::LINE_AA);
            }

            for (size_t i = 0; i < matchResult.matches.size(); ++i) {
                const cv::DMatch& match = matchResult.matches[i];
                if (match.queryIdx < 0 || match.queryIdx >= static_cast<int>(queryKp.size())) continue;
                if (match.trainIdx < 0 || match.trainIdx >= static_cast<int>(sift2_kp.size())) continue;

                const cv::Scalar color(70 + (i * 23) % 140, 120 + (i * 41) % 120, 220 - (i * 17) % 120);
                cv::circle(featurePreview, queryKp[match.queryIdx].pt, 4, color, 2, cv::LINE_AA);
                cv::circle(loadedPreview, sift2_kp[match.trainIdx].pt, 4, color, 2, cv::LINE_AA);
            }

            if (matchResult.matches.size() >= 4) {
                std::vector<cv::Point2f> srcPts;
                std::vector<cv::Point2f> dstPts;
                srcPts.reserve(matchResult.matches.size());
                dstPts.reserve(matchResult.matches.size());
                for (const auto& m : matchResult.matches) {
                    if (m.queryIdx < 0 || m.queryIdx >= static_cast<int>(queryKp.size())) continue;
                    if (m.trainIdx < 0 || m.trainIdx >= static_cast<int>(sift2_kp.size())) continue;
                    srcPts.push_back(sift2_kp[m.trainIdx].pt);
                    dstPts.push_back(queryKp[m.queryIdx].pt);
                }

                if (srcPts.size() >= 4) {
                    cv::Mat inlierMask;
                    const cv::Mat H = cv::findHomography(srcPts, dstPts, cv::RANSAC, 3.0, inlierMask);
                    if (!H.empty()) {
                        std::vector<cv::Point2f> corners2 = {
                            cv::Point2f(0.f, 0.f),
                            cv::Point2f(static_cast<float>(img2_copy.cols - 1), 0.f),
                            cv::Point2f(static_cast<float>(img2_copy.cols - 1), static_cast<float>(img2_copy.rows - 1)),
                            cv::Point2f(0.f, static_cast<float>(img2_copy.rows - 1))
                        };
                        std::vector<cv::Point2f> projected;
                        cv::perspectiveTransform(corners2, projected, H);
                        if (projected.size() == 4) {
                            for (int i = 0; i < 4; ++i) {
                                cv::line(featurePreview, projected[i], projected[(i + 1) % 4],
                                         cv::Scalar(255, 190, 80), 2, cv::LINE_AA);
                            }
                        }

                        if (!inlierMask.empty()) {
                            matchResult.inliers = cv::countNonZero(inlierMask);
                        }
                    }
                }
            }

            matchResult.composite = featurePreview;
            matchResult.loadedAnnotated = loadedPreview;

        } catch (const std::exception& e) {
            qWarning() << "Feature matching error:" << e.what();
            matchResult.composite = cv::Mat();
            matchResult.loadedAnnotated = cv::Mat();
            matchResult.inliers   = 0;
        } catch (...) {
            qWarning() << "Feature matching: unknown error";
            matchResult.composite = cv::Mat();
            matchResult.loadedAnnotated = cv::Mat();
            matchResult.inliers   = 0;
        }

        matchResult.timeMs =
            std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }));
}

void MainWindow::onMatchingFinished()
{
    if (!matchResult.loadedAnnotated.empty()) {
        ui->matchDisplayLoaded->setImage(matToPixmap(matchResult.loadedAnnotated));
    } else if (!img2.empty()) {
        ui->matchDisplayLoaded->setImage(matToPixmap(img2));
    }

    if (roiLabel) {
        if (!matchResult.composite.empty()) {
            roiLabel->setPixmap(matToPixmap(matchResult.composite));
        } else {
            roiLabel->setText("Matching failed — check console.");
        }
    }

    const QString methodName = (ui->matchMethodCombo && ui->matchMethodCombo->currentIndex() == 1) ? "NCC" : "SSD";
    ui->matchInfo->setText(
        QString("%1  |  Matches: %2  |  Inliers: %3  |  Time: %4 ms")
        .arg(methodName)
        .arg(matchResult.matches.size())
        .arg(matchResult.inliers)
        .arg(matchResult.timeMs, 0, 'f', 1));

    setStatus(QString("Feature matching — %1 matches in %2 ms")
        .arg(matchResult.matches.size())
        .arg(matchResult.timeMs, 0, 'f', 1));

    ui->btnMatchFeatures->setEnabled(true);
}
