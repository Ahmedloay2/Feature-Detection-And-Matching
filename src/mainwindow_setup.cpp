#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QApplication>
#include <QTabWidget>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

// UI setup and signal wiring only.

void MainWindow::setupTab1()
{
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
    ui->matchInfo->setMinimumWidth(320);
    ui->matchInfo->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    ui->matchDisplayLoaded->setText("Load Image 2");

    // Single-panel mode for Tab 3: render the combined correspondence canvas
    // in the feature panel only.
    if (ui->labelMatchLoaded) ui->labelMatchLoaded->setVisible(false);
    if (ui->matchDisplayLoaded) ui->matchDisplayLoaded->setVisible(false);

    if (ui->matchMethodCombo) {
        ui->matchMethodCombo->setCurrentIndex(0);
    }
    ui->matchROIInfo->setText("Draw ROI on the feature image (left panel)");

    if (ui->tab3Root) ui->tab3Root->setStretch(0, 0), ui->tab3Root->setStretch(1, 1);
    if (ui->tab3ImagesLayout) {
        ui->tab3ImagesLayout->setStretch(0, 1);
        ui->tab3ImagesLayout->setStretch(1, 0);
    }
}

void MainWindow::setupConnections()
{
    connect(ui->btnLoadImage1, &QPushButton::clicked, this, &MainWindow::onLoadImage1);
    connect(ui->btnLoadImage2, &QPushButton::clicked, this, &MainWindow::onLoadImage2);

    connect(ui->cornerModeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onCornerModeChanged);
    connect(ui->cornerThresholdSlider, &QSlider::valueChanged, this, &MainWindow::onThresholdChanged);
    connect(ui->harrisKSlider, &QSlider::valueChanged, this, &MainWindow::onHarrisKChanged);
    connect(ui->nmsWindowSlider, &QSlider::valueChanged, this, &MainWindow::onNmsWindowChanged);
    connect(ui->btnDetectCorners, &QPushButton::clicked, this, &MainWindow::onDetectCorners);
    connect(&watcherCorners, &QFutureWatcher<void>::finished, this, &MainWindow::onCornerDetectionFinished);

    connect(ui->siftContrastSlider, &QSlider::valueChanged, this, &MainWindow::onSiftContrastChanged);
    connect(ui->siftOctavesSlider, &QSlider::valueChanged, this, &MainWindow::onSiftOctavesChanged);
    connect(ui->siftScalesSlider, &QSlider::valueChanged, this, &MainWindow::onSiftScalesChanged);
    connect(ui->btnRunSift, &QPushButton::clicked, this, &MainWindow::onRunSift);
    connect(&watcherSift, &QFutureWatcher<void>::finished, this, &MainWindow::onSiftExtractionFinished);

    connect(ui->matchRatioSlider, &QSlider::valueChanged, this, &MainWindow::onMatchRatioChanged);
    connect(ui->btnMatchFeatures, &QPushButton::clicked, this, &MainWindow::onMatchFeatures);
    connect(ui->btnClearROI, &QPushButton::clicked, this, &MainWindow::onClearROI);
    connect(ui->btnRemoveLastROI, &QPushButton::clicked, this, &MainWindow::onRemoveLastROI);
    connect(ui->matchMethodCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        const bool haveSelected = (idx == 0 && !p3_featureSSD.empty() && p3_lastComputedMode == 0)
                               || (idx == 1 && !p3_featureNCC.empty() && p3_lastComputedMode == 1);
        if (haveSelected) {
            onMatchingFinished();
            return;
        }

        if (roiLabel && siftResult1.valid && !siftResult1.annotated.empty()) {
            cv::Mat left = siftResult1.annotated.clone();
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
            ui->matchInfo->setText("Mode changed — press Run Match");
        }
    });
    connect(&watcherMatch, &QFutureWatcher<void>::finished, this, &MainWindow::onMatchingFinished);

    connect(ui->tabWidget, QOverload<int>::of(&QTabWidget::currentChanged), this, [this](int idx) {
        try {
            if (idx == 1) {
                if (!ui || !ui->siftCornerModeCombo || !ui->siftDisplayCorners) return;
                const bool isHarris = (ui->siftCornerModeCombo->currentIndex() == 0);
                const CornerResult& cr = isHarris ? harrisResult1 : shiTomasiResult1;
                if (cr.valid && !cr.annotated.empty()) {
                    ui->siftDisplayCorners->setImage(matToPixmap(cr.annotated));
                }
            } else if (idx == 2) {
                if (siftResult1.valid && !siftResult1.annotated.empty()) {
                    if (roiLabel && ui->matchROIInfo) {
                        cv::Mat left = siftResult1.annotated.clone();
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
                        ui->matchROIInfo->setText("Draw ROI on the feature image (left panel)");
                        if (ui->btnMatchFeatures) {
                            ui->btnMatchFeatures->setEnabled(!img2.empty());
                        }
                    }
                }
            }
        } catch (...) {
        }
    });
}

void MainWindow::promoteMatchLabel()
{
    roiLabel = ui->matchDisplayFeature;
    if (roiLabel) {
        roiLabel->setScaledContents(true);
        roiLabel->setAlignment(Qt::AlignCenter);
    }
}
