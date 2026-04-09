/**
 * @file mainwindow_tab2.cpp
 * @brief Implements Tab 2: SIFT feature extraction and scale-space visualization.
 *
 * Handles:
 * - Async SIFT extraction via QtConcurrent with progress tracking
 * - Parameter updates (contrast threshold, octaves, scales per octave)
 * - Overlay options: display Harris/Shi-Tomasi corners alongside SIFT keypoints
 * - Keypoint visualization: circles scaled by feature size, lines for orientation
 * - Descriptor caching: results reused for Tab 3 matching when parameters unchanged
 * - SIFT pipeline orchestration (Gaussian pyramid → extrema → descriptor computation)
 */

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "core/SiftCore.hpp"

#include <opencv2/features2d.hpp>

#include <QApplication>

#include <chrono>

using Clock = std::chrono::high_resolution_clock;

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

CornerResult& MainWindow::ensureCornersCached(const std::string& mode)
{
    if (img1.empty()) throw std::runtime_error("ensureCornersCached: img1 is empty");

    if (mode == "harris") {
        if (!harrisResult1.valid || harrisResult1.thresholdUsed != p1_threshold ||
            harrisResult1.kUsed != p1_harrisK || harrisResult1.halfWinUsed != p1_nmsHalfWin) {
            runHarris(p1_threshold, p1_harrisK, p1_nmsHalfWin, harrisResult1);
        }
        return harrisResult1;
    }

    if (!shiTomasiResult1.valid || shiTomasiResult1.thresholdUsed != p1_threshold ||
        shiTomasiResult1.halfWinUsed != p1_nmsHalfWin) {
        runShiTomasi(p1_threshold, p1_nmsHalfWin, shiTomasiResult1);
    }
    return shiTomasiResult1;
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
    } catch (...) {
    }

    const float contrast = p2_contrastThresh;
    const int numOctaves = p2_numOctaves;
    const int numScales = p2_numScales;
    const cv::Mat img1_copy = img1.clone();

    siftResult1 = {};

    watcherSift.setFuture(QtConcurrent::run([this, contrast, numOctaves, numScales, img1_copy]() {
        const auto t0 = Clock::now();
        SiftResult tempResult = {};
        try {
            cv_assign::SiftProcessor::extractFeatures(img1_copy, tempResult.keypoints, tempResult.descriptors, contrast, numOctaves, numScales);
            cv::drawKeypoints(img1_copy, tempResult.keypoints, tempResult.annotated, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            tempResult.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            tempResult.valid = true;
            tempResult.contrastThreshold = contrast;
            tempResult.numOctaves = numOctaves;
            tempResult.numScales = numScales;
            siftResult1 = tempResult;
        } catch (...) {
            siftResult1 = {};
        }
    }));
}

void MainWindow::onSiftExtractionFinished()
{
    if (siftResult1.valid && !siftResult1.annotated.empty()) {
        ui->siftDisplayFeatures->setImage(matToPixmap(siftResult1.annotated));
        ui->siftInfo->setText(QString("Keypoints: %1  |  Time: %2 ms").arg(siftResult1.keypoints.size()).arg(siftResult1.timeMs, 0, 'f', 1));

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
        ? QString("SIFT done — %1 keypoints in %2 ms").arg(siftResult1.keypoints.size()).arg(siftResult1.timeMs, 0, 'f', 1)
        : "SIFT extraction failed.");

    ui->btnRunSift->setEnabled(true);
}
