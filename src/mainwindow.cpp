#include "mainwindow.h"
#include "SiftCore.hpp"
#include "Timer.hpp"

#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent)
{
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);

    // Top Panel
    QHBoxLayout *buttonsLayout = new QHBoxLayout();
    btnLoadImage1 = new QPushButton("Load Full Scene (Image 1)", this);
    btnLoadImage2 = new QPushButton("Load Target (Image 2)", this);
    btnClearROIs = new QPushButton("Clear Targets", this);
    btnRunPipeline = new QPushButton("Run Custom SIFT & Match", this);

    buttonsLayout->addWidget(btnLoadImage1);
    buttonsLayout->addWidget(btnLoadImage2);
    buttonsLayout->addWidget(btnClearROIs);
    buttonsLayout->addWidget(btnRunPipeline);

    mainLayout->addLayout(buttonsLayout);

    // Middle Panel: 3 identically spaced squashed panels
    QHBoxLayout *imagesLayout = new QHBoxLayout();

    imageScrollArea1 = new QScrollArea(this);
    lblImage1 = new QLabel(this);
    lblImage1->setAlignment(Qt::AlignCenter);
    lblImage1->setScaledContents(true);
    imageScrollArea1->setWidget(lblImage1);
    imageScrollArea1->setWidgetResizable(true);

    imageScrollArea2 = new QScrollArea(this);
    lblImage2 = new InteractiveLabel(this);
    lblImage2->setAlignment(Qt::AlignCenter);
    lblImage2->setScaledContents(true);
    imageScrollArea2->setWidget(lblImage2);
    imageScrollArea2->setWidgetResizable(true);

    imageScrollArea3 = new QScrollArea(this);
    lblOutputImage = new QLabel("Result Array", this);
    lblOutputImage->setAlignment(Qt::AlignCenter);
    lblOutputImage->setScaledContents(true);
    imageScrollArea3->setWidget(lblOutputImage);
    imageScrollArea3->setWidgetResizable(true);

    imagesLayout->addWidget(imageScrollArea1, 1);
    imagesLayout->addWidget(imageScrollArea2, 1);
    imagesLayout->addWidget(imageScrollArea3, 1);

    mainLayout->addLayout(imagesLayout, 1);

    // Bottom Panel
    lblStatus = new QLabel("Execution Timing: Ready. Load Images.", this);
    lblStatus->setStyleSheet("QLabel { color: #81C784; font-weight: bold; font-size: 14px; padding: 10px; }");
    mainLayout->addWidget(lblStatus);

    resize(1400, 700);
    setWindowTitle("Custom Multi-Threaded SIFT Extractor");

    connect(btnLoadImage1, &QPushButton::clicked, this, &MainWindow::onLoadImage1);
    connect(btnLoadImage2, &QPushButton::clicked, this, &MainWindow::onLoadImage2);
    connect(btnClearROIs, &QPushButton::clicked, this, &MainWindow::onClearROIs);
    connect(btnRunPipeline, &QPushButton::clicked, this, &MainWindow::onRunPipeline);
    connect(lblImage2, &InteractiveLabel::roiSelected, this, &MainWindow::onROISelected);
    connect(&watcher1, &QFutureWatcher<void>::finished, this, &MainWindow::onImage1SiftFinished);
    connect(&watcher2, &QFutureWatcher<void>::finished, this, &MainWindow::onImage2SiftFinished);
    connect(&watcherMatch, &QFutureWatcher<void>::finished, this, &MainWindow::onSiftExtractionFinished);

    btnRunPipeline->setEnabled(false);
}

MainWindow::~MainWindow() {}

void MainWindow::downscaleImageIfNeeded(cv::Mat &img)
{
    const int MAX_DIM = 1024;
    if (img.cols > MAX_DIM || img.rows > MAX_DIM)
    {
        float scale = std::min((float)MAX_DIM / img.cols, (float)MAX_DIM / img.rows);
        cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
    }
}

void MainWindow::onLoadImage1()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty())
    {
        img1 = cv::imread(fileName.toStdString());
        downscaleImageIfNeeded(img1);

        cv::Mat rgb;
        cv::cvtColor(img1, rgb, cv::COLOR_BGR2RGB);
        QImage qimg((const unsigned char *)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        lblImage1->setPixmap(QPixmap::fromImage(qimg));

        lblStatus->setText(QString("Image 1 loaded (%1x%2). Extracting features automagically...").arg(img1.cols).arg(img1.rows));

        btnLoadImage1->setEnabled(false);
        btnLoadImage2->setEnabled(false);
        btnRunPipeline->setEnabled(false);
        QApplication::processEvents();

        QFuture<void> future = QtConcurrent::run([this]()
                                                 {
            try {
                kp1.clear();
                desc1.release();
                cv_assign::SiftProcessor::extractFeatures(img1, kp1, desc1);
            } catch (...) {} });

        watcher1.setFuture(future);
    }
}

void MainWindow::onImage1SiftFinished()
{
    if (!kp1.empty())
    {
        cv::Mat displayImg;
        cv::drawKeypoints(img1, kp1, displayImg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::cvtColor(displayImg, displayImg, cv::COLOR_BGR2RGB);
        QImage qimg((const unsigned char *)displayImg.data, displayImg.cols, displayImg.rows, displayImg.step, QImage::Format_RGB888);
        lblImage1->setPixmap(QPixmap::fromImage(qimg));

        lblStatus->setText(QString("Image 1 auto-extracted %1 keypoints. Ready to load Image 2.").arg(kp1.size()));
    }
    else
    {
        lblStatus->setText("Image 1 extraction failed or found no keypoints.");
    }
    btnLoadImage1->setEnabled(true);
    btnLoadImage2->setEnabled(true);
    if (!img1.empty() && !img2.empty() && !roiRects.empty())
        btnRunPipeline->setEnabled(true);
}

void MainWindow::onLoadImage2()
{
    QString fileName = QFileDialog::getOpenFileName(this, "Open Image 2", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (!fileName.isEmpty())
    {
        img2 = cv::imread(fileName.toStdString());
        downscaleImageIfNeeded(img2);

        cv::Mat rgb;
        cv::cvtColor(img2, rgb, cv::COLOR_BGR2RGB);
        QImage qimg((const unsigned char *)rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
        lblImage2->setPixmap(QPixmap::fromImage(qimg));
        lblImage2->clearROI();
        roiRects.clear();

        lblStatus->setText(QString("Image 2 loaded (%1x%2). Extracting features automagically...").arg(img2.cols).arg(img2.rows));

        btnLoadImage1->setEnabled(false);
        btnLoadImage2->setEnabled(false);
        btnRunPipeline->setEnabled(false);
        QApplication::processEvents();

        QFuture<void> future = QtConcurrent::run([this]()
                                                 {
            try {
                kp2.clear();
                desc2.release();
                cv_assign::SiftProcessor::extractFeatures(img2, kp2, desc2);
            } catch (...) {} });

        watcher2.setFuture(future);
    }
}

void MainWindow::onImage2SiftFinished()
{
    // Show the CLEAN image on lblImage2 (no keypoint circles)
    // so the user can draw ROIs on the original image clearly.
    cv::Mat rgb2;
    cv::cvtColor(img2, rgb2, cv::COLOR_BGR2RGB);
    QImage qimg2((const unsigned char *)rgb2.data, rgb2.cols, rgb2.rows, rgb2.step, QImage::Format_RGB888);
    lblImage2->setPixmap(QPixmap::fromImage(qimg2));

    if (!kp2.empty())
        lblStatus->setText(QString("Image 2 auto-extracted %1 keypoints (circles hidden — draw your ROI boxes).").arg(kp2.size()));
    else
        lblStatus->setText("Image 2 extraction failed or found no keypoints.");

    btnLoadImage1->setEnabled(true);
    btnLoadImage2->setEnabled(true);
    if (!img1.empty() && !img2.empty() && !roiRects.empty())
        btnRunPipeline->setEnabled(true);
}

void MainWindow::onROISelected()
{
    auto qrois = lblImage2->getSelectedROIs();
    roiRects.clear();
    for (auto r : qrois)
    {
        roiRects.push_back(cv::Rect(r.x(), r.y(), r.width(), r.height()));
    }
    lblStatus->setText(QString("%1 Targets mapped to subset ROI. Ready for matching.").arg(roiRects.size()));
    if (!img1.empty() && !img2.empty() && !roiRects.empty())
    {
        btnRunPipeline->setEnabled(true);
    }
}

void MainWindow::onClearROIs()
{
    lblImage2->clearROI();
    roiRects.clear();
    lblStatus->setText("Targets cleared. Please drag bounding box ROIs.");
    btnRunPipeline->setEnabled(false);
}

void MainWindow::onRunPipeline()
{
    if (img1.empty() || img2.empty() || roiRects.empty())
        return;

    btnRunPipeline->setText("Processing... Please wait.");
    btnRunPipeline->setEnabled(false);
    btnLoadImage1->setEnabled(false);
    btnLoadImage2->setEnabled(false);
    QApplication::processEvents();

    QFuture<void> future = QtConcurrent::run([this]()
                                             {
        try {
            roiKps.clear();
            roiDescs.clear();
            for (size_t i = 0; i < roiRects.size(); ++i) {
                std::vector<cv::KeyPoint> localKps;
                cv::Mat localDesc;
                const cv::Rect& roi = roiRects[i];
                
                for (size_t j = 0; j < kp2.size(); ++j) {
                    if (roi.contains(kp2[j].pt)) {
                        localKps.push_back(kp2[j]);
                        localDesc.push_back(desc2.row(j));
                    }
                }
                
                roiKps.push_back(localKps);
                if (localDesc.empty()) {
                    roiDescs.push_back(cv::Mat());
                } else {
                    roiDescs.push_back(localDesc.clone());
                }
            }
        } catch (...) {
        } });

    watcherMatch.setFuture(future);
}

void MainWindow::onSiftExtractionFinished()
{
    try
    {
        if (desc1.empty() || kp1.empty() || desc1.type() != CV_32F || desc1.rows < 2)
        {
            throw cv::Exception(cv::Error::StsError, "Image 1 Extrema outputs returned zero variables or rows < 2.", "", __FILE__, __LINE__);
        }

        int totalMatches = 0;

        std::vector<cv::Scalar> colors = {
            cv::Scalar(0, 0, 255),
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 255, 255),
            cv::Scalar(255, 0, 255)};

        cv::BFMatcher matcher(cv::NORM_L2);

        cv::Mat resultImg;
        // Concatenate img2 (with ROIs) and img1 (scene) side-by-side to draw arrows between them
        int maxRows = std::max(img1.rows, img2.rows);
        int totalCols = img1.cols + img2.cols;
        resultImg = cv::Mat::zeros(maxRows, totalCols, CV_8UC3);

        cv::Mat outImg2 = resultImg(cv::Rect(0, 0, img2.cols, img2.rows));
        img2.copyTo(outImg2);
        cv::Mat outImg1 = resultImg(cv::Rect(img2.cols, 0, img1.cols, img1.rows));
        img1.copyTo(outImg1);

        for (size_t index = 0; index < roiRects.size(); ++index)
        {
            const auto &roi_kp = roiKps[index];
            const auto &roi_desc = roiDescs[index];
            const auto &roi = roiRects[index];

            if (roi_desc.empty() || roi_kp.empty() || roi_desc.type() != CV_32F || roi_desc.rows < 2)
                continue;

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(roi_desc, desc1, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            const float ratio_thresh = 0.80f;
            for (size_t i = 0; i < knn_matches.size(); i++)
            {
                if (knn_matches[i].size() > 1 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
                {
                    good_matches.push_back(knn_matches[i][0]);
                }
            }

            totalMatches += good_matches.size();
            cv::Scalar color = colors[index % colors.size()];

            cv::rectangle(resultImg, roi, color, 3);

            for (const auto &match : good_matches)
            {
                cv::Point2f pt1 = roi_kp[match.queryIdx].pt;
                cv::Point2f pt2 = kp1[match.trainIdx].pt + cv::Point2f(img2.cols, 0);
                cv::arrowedLine(resultImg, pt1, pt2, color, 2, cv::LINE_AA, 0, 0.1);
            }
        }

        lblStatus->setText(QString("Processed %1 ROIs and Image 1. Total Matched pairs: %2")
                               .arg(roiRects.size())
                               .arg(totalMatches));

        cv::cvtColor(resultImg, resultImg, cv::COLOR_BGR2RGB);
        QImage qimg((const unsigned char *)resultImg.data, resultImg.cols, resultImg.rows, resultImg.step, QImage::Format_RGB888);
        lblOutputImage->setPixmap(QPixmap::fromImage(qimg));
    }
    catch (const cv::Exception &e)
    {
        lblStatus->setText("Error: No matches found or matrices empty. Extraction pipeline failed limit.");
        lblOutputImage->setText("Error: No matches found.");
    }
    catch (...)
    {
        lblStatus->setText("Error: Extraction pipeline crashed. Catch invoked.");
        lblOutputImage->setText("Error: Pipeline crash.");
    }

    btnRunPipeline->setText("Run Custom SIFT & Match");
    btnRunPipeline->setEnabled(true);
    btnLoadImage1->setEnabled(true);
    btnLoadImage2->setEnabled(true);
}