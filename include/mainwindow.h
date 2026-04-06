#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QFutureWatcher>
#include <QtConcurrent>
#include "interactive_label.h"
#include <opencv2/opencv.hpp>
#include <vector>

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onLoadImage1();
    void onLoadImage2();
    void onRunPipeline();
    void onROISelected();
    void onClearROIs();
    void onImage1SiftFinished();
    void onImage2SiftFinished();
    void onSiftExtractionFinished();

    void onRatioSliderChanged(int value);
    void onContrastSliderChanged(int value);

private:
    void downscaleImageIfNeeded(cv::Mat &img);

    // ── Buttons ──────────────────────────────────────────────────────────────
    QPushButton *btnLoadImage1;
    QPushButton *btnLoadImage2;
    QPushButton *btnRunPipeline;
    QPushButton *btnClearROIs;

    // ── Sliders ───────────────────────────────────────────────────────────────
    // Lowe's ratio test  [50..95] → divided by 100 → [0.50..0.95]
    QSlider *sliderRatio;
    QLabel *lblRatioVal;

    // Contrast threshold [1..30] → divided by 1000 → [0.001..0.030]
    QSlider *sliderContrast;
    QLabel *lblContrastVal;

    // ── Status bar ───────────────────────────────────────────────────────────
    QLabel *lblStatus;

    // ── Image panels ─────────────────────────────────────────────────────────
    QScrollArea *imageScrollArea1;
    QLabel *lblImage1;

    QScrollArea *imageScrollArea2;
    InteractiveLabel *lblImage2;

    QScrollArea *imageScrollArea3;
    QLabel *lblOutputImage;

    // ── CV data ──────────────────────────────────────────────────────────────
    cv::Mat img1, img2;
    std::vector<cv::Rect> roiRects;

    QFutureWatcher<void> watcher1, watcher2, watcherMatch;
    std::vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desc1, desc2;
    std::vector<std::vector<cv::KeyPoint>> roiKps;
    std::vector<cv::Mat> roiDescs;

    // ── Threshold values (kept in sync with sliders) ──────────────────────────
    float ratioThresh;    // default 0.80
    float contrastThresh; // default 0.007
};