#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
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

private:
    void downscaleImageIfNeeded(cv::Mat &img);

    QPushButton *btnLoadImage1;
    QPushButton *btnLoadImage2;
    QPushButton *btnRunPipeline;
    QPushButton *btnClearROIs;

    QLabel *lblStatus;

    QScrollArea *imageScrollArea1;
    QLabel *lblImage1;

    QScrollArea *imageScrollArea2;
    InteractiveLabel *lblImage2;

    QScrollArea *imageScrollArea3;
    QLabel *lblOutputImage;

    cv::Mat img1;
    cv::Mat img2;
    std::vector<cv::Rect> roiRects;

    QFutureWatcher<void> watcher1;
    QFutureWatcher<void> watcher2;
    QFutureWatcher<void> watcherMatch;
    // Data structures for storing async results
    std::vector<cv::KeyPoint> kp1;
    cv::Mat desc1;
    std::vector<cv::KeyPoint> kp2;
    cv::Mat desc2;
    std::vector<std::vector<cv::KeyPoint>> roiKps;
    std::vector<cv::Mat> roiDescs;
};
