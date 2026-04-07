#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "io/image_handler.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>

#include <QFileDialog>
#include <QMessageBox>

// Shared conversion helpers and image-loading slots.

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

    ui->matchDisplayLoaded->setImage(matToPixmap(img2));
    if (siftResult1.valid && !img2.empty()) ui->btnMatchFeatures->setEnabled(true);

    setStatus(QString("Image 2 loaded  %1 × %2").arg(img2.cols).arg(img2.rows));
}
