/**
 * @file mainwindow_shared.cpp
 * @brief Implements shared utilities used across all tabs: image I/O, conversions, and helpers.
 *
 * Provides:
 * - Image file loading (file dialog + loadImage wrapper)
 * - OpenCV Mat ↔ Qt QPixmap conversion (color space handling: BGR↔RGB, grayscale)
 * - Image preprocessing: auto-downsampling for large images
 * - Status bar updates and user feedback messages
 * - Image caching and reuse across tabs
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "io/image_handler.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

#include <QFileDialog>
#include <QMessageBox>

QPixmap MainWindow::matToPixmap(const cv::Mat& mat)
{
    if (mat.empty()) return {};

    cv::Mat rgb;
    if (mat.channels() == 1) cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    else if (mat.channels() == 3) cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else return {};

    QImage qi(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    return QPixmap::fromImage(qi.copy());
}

void MainWindow::downscaleIfNeeded(cv::Mat& img, int maxDim)
{
    if (img.cols > maxDim || img.rows > maxDim) {
        const float sx = static_cast<float>(maxDim) / static_cast<float>(img.cols);
        const float sy = static_cast<float>(maxDim) / static_cast<float>(img.rows);
        const float s = std::min(sx, sy);
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

void MainWindow::onLoadImage1()
{
    const QString fn = QFileDialog::getOpenFileName(this, "Open Image 1", {}, "Images (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (fn.isEmpty()) return;

    try {
        img1 = loadImage(fn.toStdString()).mat;
        downscaleIfNeeded(img1);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Load Error", e.what());
        return;
    }

    // Reset all caches that depend on image 1.
    harrisResult1 = {};
    shiTomasiResult1 = {};
    siftResult1 = {};
    siftResult2 = {};
    matchResult = {};
    p3_ssdMatches.clear();
    p3_nccMatches.clear();
    p3_lastComputedMode = -1;
    p3_ssdInliers = 0;
    p3_nccInliers = 0;
    p3_ssdTimeMs = 0.0;
    p3_nccTimeMs = 0.0;
    p3_featureSSD = cv::Mat();
    p3_featureNCC = cv::Mat();
    p3_loadedSSD = cv::Mat();
    p3_loadedNCC = cv::Mat();
    {
        std::lock_guard<std::mutex> lock(p1PipelineMutex);
        p1PipelineImage = {};
        p1PipelineImage.mat = img1.clone();
        p1PipelineImage.clearCache();
    }

    const QPixmap px = matToPixmap(img1);

    ui->cornerDisplayOriginal->setImage(px);
    ui->cornerDisplayResult->setText("Run detection");
    ui->cornerStatsHarris->setText("Harris  —  not run");
    ui->cornerStatsShiTomasi->setText("Shi-Tomasi  —  not run");
    ui->btnDetectCorners->setEnabled(true);

    ui->siftDisplayOriginal->setImage(px);
    ui->siftDisplayCorners->setText("Corners appear here");
    ui->siftDisplayFeatures->setText("SIFT features appear here");
    ui->siftInfo->setText("Keypoints: —  |  Time: — ms");
    ui->btnRunSift->setEnabled(true);

    if (roiLabel) {
        roiLabel->setText("Run SIFT first (Tab 2)");
        roiLabel->clearROI();
    }
    ui->btnMatchFeatures->setEnabled(false);

    setStatus(QString("Image 1 loaded  %1 × %2").arg(img1.cols).arg(img1.rows));
}

void MainWindow::onLoadImage2()
{
    const QString fn = QFileDialog::getOpenFileName(this, "Open Image 2 (to match)", {}, "Images (*.png *.jpg *.jpeg *.bmp *.tiff)");
    if (fn.isEmpty()) return;

    try {
        img2 = loadImage(fn.toStdString()).mat;
        downscaleIfNeeded(img2);
    } catch (const std::exception& e) {
        QMessageBox::critical(this, "Load Error", e.what());
        return;
    }

    siftResult2 = {};
    matchResult.loadedAnnotated = cv::Mat();
    p3_ssdMatches.clear();
    p3_nccMatches.clear();
    p3_lastComputedMode = -1;
    p3_ssdInliers = 0;
    p3_nccInliers = 0;
    p3_ssdTimeMs = 0.0;
    p3_nccTimeMs = 0.0;
    p3_featureSSD = cv::Mat();
    p3_featureNCC = cv::Mat();
    p3_loadedSSD = cv::Mat();
    p3_loadedNCC = cv::Mat();

    ui->matchDisplayLoaded->setImage(matToPixmap(img2));

    if (roiLabel && !img1.empty()) {
        cv::Mat left = (siftResult1.valid && !siftResult1.annotated.empty())
            ? siftResult1.annotated.clone()
            : img1.clone();

        cv::Mat leftBgr;
        if (left.channels() == 3) leftBgr = left;
        else cv::cvtColor(left, leftBgr, cv::COLOR_GRAY2BGR);

        cv::Mat right = cv::Mat::zeros(leftBgr.rows, leftBgr.cols, CV_8UC3);
        if (!img2.empty()) {
            cv::Mat rightSrc;
            if (img2.channels() == 3) rightSrc = img2;
            else cv::cvtColor(img2, rightSrc, cv::COLOR_GRAY2BGR);

            const float sx = static_cast<float>(leftBgr.cols) / std::max(1, rightSrc.cols);
            const float sy = static_cast<float>(leftBgr.rows) / std::max(1, rightSrc.rows);
            const float s = std::min(sx, sy);
            const int fitW = std::max(1, static_cast<int>(std::round(rightSrc.cols * s)));
            const int fitH = std::max(1, static_cast<int>(std::round(rightSrc.rows * s)));
            cv::Mat fitted;
            cv::resize(rightSrc, fitted, cv::Size(fitW, fitH), 0, 0, cv::INTER_AREA);
            const int offX = (right.cols - fitW) / 2;
            const int offY = (right.rows - fitH) / 2;
            fitted.copyTo(right(cv::Rect(offX, offY, fitW, fitH)));
        }

        cv::Mat combined;
        cv::hconcat(leftBgr, right, combined);
        roiLabel->setPixmap(matToPixmap(combined));
        roiLabel->clearROI();
    }

    if (siftResult1.valid && !img2.empty()) ui->btnMatchFeatures->setEnabled(true);

    setStatus(QString("Image 2 loaded  %1 × %2").arg(img2.cols).arg(img2.rows));
}
