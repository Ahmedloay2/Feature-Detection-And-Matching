/**
 * @file mainwindow_tab1.cpp
 * @brief Implements Tab 1: Harris/Shi-Tomasi corner detection pipeline and visualization.
 *
 * Handles:
 * - Async corner detection via QtConcurrent::run() with QFutureWatcher
 * - Parameter updates from UI sliders (k, threshold, NMS window)
 * - Detector mode switching (Harris ↔ Shi-Tomasi)
 * - Result caching: reuses cached results when parameters unchanged
 * - Corner visualization: renders detected points as colored circles on images
 * - Per-stage timing measurement and status bar updates
 */

#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include "processors/harris_main.hpp"

#include <opencv2/imgproc.hpp>

#include <QApplication>

#include <chrono>

using Clock = std::chrono::high_resolution_clock;

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

void MainWindow::runHarris(float threshold, float k, int halfWin, CornerResult& out)
{
    const auto t0 = Clock::now();
    {
        std::lock_guard<std::mutex> lock(p1PipelineMutex);
        if (p1PipelineImage.mat.empty()) {
            p1PipelineImage.mat = img1.clone();
            p1PipelineImage.clearCache();
        }
        out.points = applyHarris(p1PipelineImage, k, "harris", threshold, halfWin);
    }

    out.count = static_cast<int>(out.points.size());
    out.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    out.valid = true;
    out.thresholdUsed = threshold;
    out.kUsed = k;
    out.halfWinUsed = halfWin;

    cv::Mat vis;
    if (img1.channels() == 3) vis = img1.clone();
    else if (img1.channels() == 1) cv::cvtColor(img1, vis, cv::COLOR_GRAY2BGR);
    else {
        cv::Mat gray = toGray(img1);
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    }
    for (const auto& pt : out.points) cv::circle(vis, pt, 4, cv::Scalar(0, 80, 255), 3, cv::LINE_AA);
    out.annotated = vis;
}

void MainWindow::runShiTomasi(float threshold, int halfWin, CornerResult& out)
{
    const auto t0 = Clock::now();
    {
        std::lock_guard<std::mutex> lock(p1PipelineMutex);
        if (p1PipelineImage.mat.empty()) {
            p1PipelineImage.mat = img1.clone();
            p1PipelineImage.clearCache();
        }
        out.points = applyHarris(p1PipelineImage, 0.f, "shi_tomasi", threshold, halfWin);
    }

    out.count = static_cast<int>(out.points.size());
    out.timeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    out.valid = true;
    out.thresholdUsed = threshold;
    out.kUsed = 0.f;
    out.halfWinUsed = halfWin;

    cv::Mat vis;
    if (img1.channels() == 3) vis = img1.clone();
    else if (img1.channels() == 1) cv::cvtColor(img1, vis, cv::COLOR_GRAY2BGR);
    else {
        cv::Mat gray = toGray(img1);
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    }
    for (const auto& pt : out.points) cv::circle(vis, pt, 4, cv::Scalar(50, 220, 80), 3, cv::LINE_AA);
    out.annotated = vis;
}

void MainWindow::onDetectCorners()
{
    if (img1.empty()) { setStatus("Load Image 1 first."); return; }

    ui->btnDetectCorners->setEnabled(false);
    ui->cornerDisplayResult->setText("Computing…");
    QApplication::processEvents();

    const float thresh = p1_threshold;
    const float k = p1_harrisK;
    const int halfWin = p1_nmsHalfWin;
    const int mode = p1_modeIndex;
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
        if (ui->siftDisplayCorners) ui->siftDisplayCorners->setImage(matToPixmap(cached.annotated));
        setStatus(QString("%1 detection — reused cached result").arg(mode == 0 ? "Harris" : "Shi-Tomasi"));
        ui->btnDetectCorners->setEnabled(true);
        return;
    }

    watcherCorners.setFuture(QtConcurrent::run([this, thresh, k, halfWin, mode]() {
        try {
            if (mode == 0) runHarris(thresh, k, halfWin, harrisResult1);
            else runShiTomasi(thresh, halfWin, shiTomasiResult1);
        } catch (...) {
            if (mode == 0) harrisResult1 = {};
            else shiTomasiResult1 = {};
        }
    }));
}

void MainWindow::onCornerDetectionFinished()
{
    const bool isHarris = (p1_lastRunMode == 0);
    const CornerResult& cr = isHarris ? harrisResult1 : shiTomasiResult1;
    const QString modeName = isHarris ? "Harris" : "Shi-Tomasi";

    if (!cr.annotated.empty()) ui->cornerDisplayResult->setImage(matToPixmap(cr.annotated));
    else ui->cornerDisplayResult->setText("Detection failed");

    auto statsText = [](const QString& name, const CornerResult& r) -> QString {
        if (!r.valid) return name + "  —  not run";
        return QString("%1  |  Corners: %2  |  Time: %3 ms").arg(name).arg(r.count).arg(r.timeMs, 0, 'f', 1);
    };

    ui->cornerStatsHarris->setText(statsText("Harris", harrisResult1));
    ui->cornerStatsShiTomasi->setText(statsText("Shi-Tomasi", shiTomasiResult1));

    if (!cr.annotated.empty()) ui->siftDisplayCorners->setImage(matToPixmap(cr.annotated));

    setStatus(QString("%1 detection — %2 corners in %3 ms").arg(modeName).arg(cr.count).arg(cr.timeMs, 0, 'f', 1));
    ui->btnDetectCorners->setEnabled(true);
}
