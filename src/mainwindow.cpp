#include "mainwindow.h"
#include "SiftCore.hpp"
#include "Timer.hpp"
#include "io/image_handler.hpp"

#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QGroupBox>
#include <mutex>
#include <opencv2/calib3d.hpp>

// ── Palette colours matching each ROI index ───────────────────────────────────
static const std::vector<cv::Scalar> ROI_COLORS = {
    cv::Scalar(0, 0, 255),   // red
    cv::Scalar(0, 255, 0),   // green
    cv::Scalar(255, 0, 0),   // blue
    cv::Scalar(0, 255, 255), // yellow
    cv::Scalar(255, 0, 255), // magenta
};

// =============================================================================
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ratioThresh(0.80f), contrastThresh(0.007f)
{
    QWidget *central = new QWidget(this);
    setCentralWidget(central);
    QVBoxLayout *mainLayout = new QVBoxLayout(central);
    mainLayout->setSpacing(4);

    // ── Row 1: Load / Run buttons ────────────────────────────────────────────
    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLoadImage1 = new QPushButton("Load Full Scene (Image 1)", this);
    btnLoadImage2 = new QPushButton("Load Target (Image 2)", this);
    btnClearROIs = new QPushButton("Clear Targets", this);
    btnRunPipeline = new QPushButton("Run Custom SIFT & Match", this);
    btnLayout->addWidget(btnLoadImage1);
    btnLayout->addWidget(btnLoadImage2);
    btnLayout->addWidget(btnClearROIs);
    btnLayout->addWidget(btnRunPipeline);
    mainLayout->addLayout(btnLayout);

    // ── Row 2: Threshold sliders ──────────────────────────────────────────────
    QGroupBox *sliderBox = new QGroupBox("Detection Parameters", this);
    sliderBox->setStyleSheet(
        "QGroupBox { color:#90CAF9; font-weight:bold; border:1px solid #37474F;"
        "            border-radius:4px; margin-top:6px; padding-top:4px; }"
        "QGroupBox::title { subcontrol-origin:margin; left:8px; }");

    QHBoxLayout *sliderLayout = new QHBoxLayout(sliderBox);
    sliderLayout->setContentsMargins(8, 2, 8, 4);
    sliderLayout->setSpacing(12);

    // --- Lowe's Ratio ---
    // 0.50 = very strict (almost no matches)
    // 0.75 = Lowe's paper standard
    // 0.80 = current default (slightly lenient)
    // 0.95 = very lenient (many false matches)
    auto *ratioTitleLabel = new QLabel("Lowe's Ratio Test:", this);
    ratioTitleLabel->setStyleSheet("color:#CFD8DC; font-size:12px;");

    sliderRatio = new QSlider(Qt::Horizontal, this);
    sliderRatio->setRange(50, 95); // 0.50 → 0.95
    sliderRatio->setValue(80);     // default 0.80
    sliderRatio->setTickPosition(QSlider::TicksBelow);
    sliderRatio->setTickInterval(5);
    sliderRatio->setMinimumWidth(180);

    lblRatioVal = new QLabel("0.80", this);
    lblRatioVal->setStyleSheet("color:#80DEEA; font-family:Consolas,monospace; font-size:12px; min-width:36px;");

    auto *ratioHint = new QLabel("← strict  |  lenient →", this);
    ratioHint->setStyleSheet("color:#546E7A; font-size:10px;");

    sliderLayout->addWidget(ratioTitleLabel);
    sliderLayout->addWidget(sliderRatio, 2);
    sliderLayout->addWidget(lblRatioVal);
    sliderLayout->addWidget(ratioHint);
    sliderLayout->addSpacing(20);

    // --- Contrast Threshold ---
    // 0.001 = extreme-low  (flood of noisy KPs)
    // 0.007 = relaxed      (default — good for low-contrast regions)
    // 0.013 = Lowe standard
    // 0.030 = strict       (only very prominent blobs)
    auto *ctTitleLabel = new QLabel("Contrast Threshold:", this);
    ctTitleLabel->setStyleSheet("color:#CFD8DC; font-size:12px;");

    sliderContrast = new QSlider(Qt::Horizontal, this);
    sliderContrast->setRange(1, 30); // 0.001 → 0.030
    sliderContrast->setValue(7);     // default 0.007
    sliderContrast->setTickPosition(QSlider::TicksBelow);
    sliderContrast->setTickInterval(3);
    sliderContrast->setMinimumWidth(180);

    lblContrastVal = new QLabel("0.007", this);
    lblContrastVal->setStyleSheet("color:#80DEEA; font-family:Consolas,monospace; font-size:12px; min-width:42px;");

    auto *ctHint = new QLabel("← many KPs  |  few KPs →", this);
    ctHint->setStyleSheet("color:#546E7A; font-size:10px;");

    sliderLayout->addWidget(ctTitleLabel);
    sliderLayout->addWidget(sliderContrast, 2);
    sliderLayout->addWidget(lblContrastVal);
    sliderLayout->addWidget(ctHint);

    mainLayout->addWidget(sliderBox);

    // ── Row 3: Image panels ──────────────────────────────────────────────────
    QHBoxLayout *imagesLayout = new QHBoxLayout();

    // Panel 1 — Full scene (keypoints shown after extraction)
    imageScrollArea1 = new QScrollArea(this);
    lblImage1 = new QLabel("Scene (Image 1)", this);
    lblImage1->setAlignment(Qt::AlignCenter);
    lblImage1->setScaledContents(true);
    imageScrollArea1->setWidget(lblImage1);
    imageScrollArea1->setWidgetResizable(true);

    // Panel 2 — Target image (ROI drawing)
    imageScrollArea2 = new QScrollArea(this);
    lblImage2 = new InteractiveLabel(this);
    lblImage2->setAlignment(Qt::AlignCenter);
    lblImage2->setScaledContents(true);
    imageScrollArea2->setWidget(lblImage2);
    imageScrollArea2->setWidgetResizable(true);

    // Panel 3 — Full scene (Image 1) annotated with matched keypoints
    imageScrollArea3 = new QScrollArea(this);
    lblOutputImage = new QLabel("Match Results (Image 1)", this);
    lblOutputImage->setAlignment(Qt::AlignCenter);
    lblOutputImage->setScaledContents(true);
    imageScrollArea3->setWidget(lblOutputImage);
    imageScrollArea3->setWidgetResizable(true);

    imagesLayout->addWidget(imageScrollArea1, 1);
    imagesLayout->addWidget(imageScrollArea2, 1);
    imagesLayout->addWidget(imageScrollArea3, 1);
    mainLayout->addLayout(imagesLayout, 1);

    // ── Row 4: Status bar ────────────────────────────────────────────────────
    lblStatus = new QLabel("Ready. Load images to begin.", this);
    lblStatus->setStyleSheet(
        "QLabel { color:#81C784; font-weight:bold; font-size:13px; padding:6px; }");
    mainLayout->addWidget(lblStatus);

    resize(1440, 760);
    setWindowTitle("Custom Multi-Threaded SIFT Extractor");

    // ── Signal connections ───────────────────────────────────────────────────
    connect(btnLoadImage1, &QPushButton::clicked,
            this, &MainWindow::onLoadImage1);
    connect(btnLoadImage2, &QPushButton::clicked,
            this, &MainWindow::onLoadImage2);
    connect(btnClearROIs, &QPushButton::clicked,
            this, &MainWindow::onClearROIs);
    connect(btnRunPipeline, &QPushButton::clicked,
            this, &MainWindow::onRunPipeline);
    connect(lblImage2, &InteractiveLabel::roiSelected,
            this, &MainWindow::onROISelected);
    connect(&watcher1, &QFutureWatcher<void>::finished,
            this, &MainWindow::onImage1SiftFinished);
    connect(&watcher2, &QFutureWatcher<void>::finished,
            this, &MainWindow::onImage2SiftFinished);
    connect(&watcherMatch, &QFutureWatcher<void>::finished,
            this, &MainWindow::onSiftExtractionFinished);

    connect(sliderRatio, &QSlider::valueChanged,
            this, &MainWindow::onRatioSliderChanged);
    connect(sliderContrast, &QSlider::valueChanged,
            this, &MainWindow::onContrastSliderChanged);

    btnRunPipeline->setEnabled(false);
}

MainWindow::~MainWindow() {}

// ── Slider handlers ──────────────────────────────────────────────────────────
void MainWindow::onRatioSliderChanged(int value)
{
    ratioThresh = value / 100.0f;
    lblRatioVal->setText(QString::number(ratioThresh, 'f', 2));
    // Ratio change only affects matching — no re-extraction needed.
}

void MainWindow::onContrastSliderChanged(int value)
{
    contrastThresh = value / 1000.0f;
    lblContrastVal->setText(QString::number(contrastThresh, 'f', 3));
    // Contrast threshold affects extraction — user must reload images to apply.
    if (!img1.empty() || !img2.empty())
        lblStatus->setText(QString(
                               "Contrast threshold changed to %1 — reload images to re-extract with new value.")
                               .arg(contrastThresh, 0, 'f', 3));
}

// ── Helpers ───────────────────────────────────────────────────────────────────
void MainWindow::downscaleImageIfNeeded(cv::Mat &img)
{
    const int MAX_DIM = 1024;
    if (img.cols > MAX_DIM || img.rows > MAX_DIM)
    {
        float scale = std::min((float)MAX_DIM / img.cols, (float)MAX_DIM / img.rows);
        cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_AREA);
    }
}

// ── Load Image 1 ─────────────────────────────────────────────────────────────
void MainWindow::onLoadImage1()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, "Open Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fileName.isEmpty())
        return;

    try
    {
        img1 = loadImage(fileName.toStdString()).mat;
    }
    catch (const std::exception &e)
    {
        QMessageBox::critical(this, "Error loading image", e.what());
        return;
    }
    downscaleImageIfNeeded(img1);

    cv::Mat rgb;
    cv::cvtColor(img1, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    lblImage1->setPixmap(QPixmap::fromImage(qimg.copy()));

    lblStatus->setText(QString("Image 1 loaded (%1×%2). Extracting features…")
                           .arg(img1.cols)
                           .arg(img1.rows));

    btnLoadImage1->setEnabled(false);
    btnLoadImage2->setEnabled(false);
    btnRunPipeline->setEnabled(false);
    QApplication::processEvents();

    float ct = contrastThresh;
    watcher1.setFuture(QtConcurrent::run([this, ct]()
                                         {
        try {
            kp1.clear(); desc1.release();
            cv_assign::SiftProcessor::extractFeatures(img1, kp1, desc1, ct);
        } catch (...) {} }));
}

void MainWindow::onImage1SiftFinished()
{
    if (!kp1.empty())
    {
        cv::Mat display;
        cv::drawKeypoints(img1, kp1, display, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::cvtColor(display, display, cv::COLOR_BGR2RGB);
        QImage qimg(display.data, display.cols, display.rows,
                    (int)display.step, QImage::Format_RGB888);
        lblImage1->setPixmap(QPixmap::fromImage(qimg.copy()));
        lblStatus->setText(
            QString("Image 1: %1 keypoints detected (contrast threshold = %2).")
                .arg(kp1.size())
                .arg(contrastThresh, 0, 'f', 3));
    }
    else
        lblStatus->setText("Image 1: extraction failed or found no keypoints.");

    btnLoadImage1->setEnabled(true);
    btnLoadImage2->setEnabled(true);
    if (!img1.empty() && !img2.empty() && !roiRects.empty())
        btnRunPipeline->setEnabled(true);
}

// ── Load Image 2 ─────────────────────────────────────────────────────────────
void MainWindow::onLoadImage2()
{
    QString fileName = QFileDialog::getOpenFileName(
        this, "Open Image 2", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fileName.isEmpty())
        return;

    try
    {
        img2 = loadImage(fileName.toStdString()).mat;
    }
    catch (const std::exception &e)
    {
        QMessageBox::critical(this, "Error loading image", e.what());
        return;
    }
    downscaleImageIfNeeded(img2);

    // Show clean image immediately so user can draw ROIs
    cv::Mat rgb;
    cv::cvtColor(img2, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, (int)rgb.step, QImage::Format_RGB888);
    lblImage2->setPixmap(QPixmap::fromImage(qimg.copy()));
    lblImage2->clearROI();
    roiRects.clear();

    lblStatus->setText(QString("Image 2 loaded (%1×%2). Extracting features…")
                           .arg(img2.cols)
                           .arg(img2.rows));

    btnLoadImage1->setEnabled(false);
    btnLoadImage2->setEnabled(false);
    btnRunPipeline->setEnabled(false);
    QApplication::processEvents();

    float ct = contrastThresh;
    watcher2.setFuture(QtConcurrent::run([this, ct]()
                                         {
        try {
            kp2.clear(); desc2.release();
            cv_assign::SiftProcessor::extractFeatures(img2, kp2, desc2, ct);
        } catch (...) {} }));
}

void MainWindow::onImage2SiftFinished()
{
    // Keep Image 2 display clean (no keypoint circles) — user needs to draw ROI boxes
    cv::Mat rgb2;
    cv::cvtColor(img2, rgb2, cv::COLOR_BGR2RGB);
    QImage q2(rgb2.data, rgb2.cols, rgb2.rows, (int)rgb2.step, QImage::Format_RGB888);
    lblImage2->setPixmap(QPixmap::fromImage(q2.copy()));

    lblStatus->setText(kp2.empty()
                           ? "Image 2: extraction failed or no keypoints."
                           : QString("Image 2: %1 keypoints detected. Draw ROI box(es) on the target region.")
                                 .arg(kp2.size()));

    btnLoadImage1->setEnabled(true);
    btnLoadImage2->setEnabled(true);
    if (!img1.empty() && !img2.empty() && !roiRects.empty())
        btnRunPipeline->setEnabled(true);
}

// ── ROI selection ─────────────────────────────────────────────────────────────
void MainWindow::onROISelected()
{
    auto qrois = lblImage2->getSelectedROIs();
    roiRects.clear();
    for (auto &r : qrois)
        roiRects.push_back(cv::Rect(r.x(), r.y(), r.width(), r.height()));
    lblStatus->setText(
        QString("%1 target ROI(s) selected. Press \"Run\" to match.")
            .arg(roiRects.size()));
    if (!img1.empty() && !img2.empty() && !roiRects.empty())
        btnRunPipeline->setEnabled(true);
}

void MainWindow::onClearROIs()
{
    lblImage2->clearROI();
    roiRects.clear();
    lblStatus->setText("Targets cleared. Draw new ROI boxes on Image 2.");
    btnRunPipeline->setEnabled(false);
}

// ── Run pipeline ──────────────────────────────────────────────────────────────
void MainWindow::onRunPipeline()
{
    if (img1.empty() || img2.empty() || roiRects.empty())
        return;

    btnRunPipeline->setText("Processing…");
    btnRunPipeline->setEnabled(false);
    btnLoadImage1->setEnabled(false);
    btnLoadImage2->setEnabled(false);
    QApplication::processEvents();

    watcherMatch.setFuture(QtConcurrent::run([this]()
                                             {
        try {
            roiKps.clear(); roiDescs.clear();
            for (size_t i = 0; i < roiRects.size(); ++i)
            {
                std::vector<cv::KeyPoint> localKps;
                cv::Mat localDesc;
                const cv::Rect &roi = roiRects[i];
                for (size_t j = 0; j < kp2.size(); ++j)
                {
                    if (roi.contains(kp2[j].pt))
                    {
                        localKps.push_back(kp2[j]);
                        localDesc.push_back(desc2.row((int)j));
                    }
                }
                roiKps.push_back(localKps);
                roiDescs.push_back(localDesc.empty() ? cv::Mat() : localDesc.clone());
            }
        } catch (...) {} }));
}

// ── Results — Panel 3 shows Image 1 annotated with match locations ─────────────
//
//  Design intent:
//    Panel 2 = Image 2 (target) with the ROI box(es) the user drew
//    Panel 3 = Image 1 (full scene) with coloured circles where each ROI's
//              features were found — one colour per ROI so you can cross-
//              reference panel 2 and panel 3 visually.
//
void MainWindow::onSiftExtractionFinished()
{
    try
    {
        if (desc1.empty() || kp1.empty() || desc1.type() != CV_32F || desc1.rows < 2)
            throw cv::Exception(cv::Error::StsError, "Image 1 descriptors empty.", "", __FILE__, __LINE__);

        // ── Visual Setup: Cross-Panel Composite Image ────────────────────────
        cv::Mat final_composite;
        int totalMatches = 0;

        for (size_t idx = 0; idx < roiRects.size(); ++idx)
        {
            const auto &roi_kp = roiKps[idx];
            const auto &roi_desc = roiDescs[idx];
            const auto &roi = roiRects[idx];
            cv::Scalar color = ROI_COLORS[idx % ROI_COLORS.size()];

            if (roi_desc.empty() || roi_kp.empty() ||
                roi_desc.type() != CV_32F || roi_desc.rows < 2 || desc1.rows < 2)
                continue;

            // 1. Custom SSD matching with thread-safe vector
            std::vector<cv::DMatch> good_matches;

#pragma omp parallel for
            for (int i = 0; i < roi_desc.rows; ++i)
            {
                const float *q_ptr = roi_desc.ptr<float>(i);
                float bestDistSq = 1e30f, secondBestDistSq = 1e30f;
                int bestIdx = -1, secondBestIdx = -1;

                for (int j = 0; j < desc1.rows; ++j)
                {
                    const float *t_ptr = desc1.ptr<float>(j);
                    float ssd = 0.0f;
                    for (int d = 0; d < 128; d++)
                    {
                        float diff = q_ptr[d] - t_ptr[d];
                        ssd += diff * diff;
                    }

                    if (ssd < bestDistSq)
                    {
                        secondBestDistSq = bestDistSq;
                        secondBestIdx = bestIdx;
                        bestDistSq = ssd;
                        bestIdx = j;
                    }
                    else if (ssd < secondBestDistSq)
                    {
                        secondBestDistSq = ssd;
                        secondBestIdx = j;
                    }
                }

                // 2. Lowe's ratio test manually
                if (bestIdx != -1 && secondBestIdx != -1)
                {
                    if (std::sqrt(bestDistSq) < ratioThresh * std::sqrt(secondBestDistSq))
                    {
                        cv::DMatch match(i, bestIdx, std::sqrt(bestDistSq));
#pragma omp critical
                        good_matches.push_back(match);
                    }
                }
            }

            // 3. RANSAC Geometric Verification via Homography
            std::vector<cv::DMatch> inlier_matches;
            if (good_matches.size() >= 4)
            {
                std::vector<cv::Point2f> srcPts, dstPts;
                for (const auto &m : good_matches)
                {
                    srcPts.push_back(roi_kp[m.queryIdx].pt);
                    dstPts.push_back(kp1[m.trainIdx].pt);
                }

                std::vector<uchar> inliersMask;
                cv::Mat H = cv::findHomography(srcPts, dstPts, cv::RANSAC, 3.0, inliersMask);

                for (size_t i = 0; i < inliersMask.size(); ++i)
                {
                    if (inliersMask[i])
                    {
                        inlier_matches.push_back(good_matches[i]);
                    }
                }
            }

            totalMatches += (int)inlier_matches.size();

            // 4. Draw Cross-Panel Connections
            cv::Mat crop = img2(roi).clone();

            int maxRows = std::max(crop.rows, img1.rows);
            int totalCols = crop.cols + img1.cols;
            cv::Mat composite = cv::Mat::zeros(maxRows, totalCols, CV_8UC3);

            cv::Mat outCrop = composite(cv::Rect(0, 0, crop.cols, crop.rows));
            crop.copyTo(outCrop);
            cv::Mat outImg1 = composite(cv::Rect(crop.cols, 0, img1.cols, img1.rows));
            img1.copyTo(outImg1);

            // Highlight the extracted patch boundary
            cv::rectangle(composite, cv::Rect(0, 0, crop.cols, crop.rows), color, 2);

            for (const auto &match : inlier_matches)
            {
                cv::Point2f pt1 = roi_kp[match.queryIdx].pt;
                // Shift src pt to be relative to the cropped ROI box
                pt1.x -= roi.x;
                pt1.y -= roi.y;

                // Shift target pt to the right panel
                cv::Point2f pt2 = kp1[match.trainIdx].pt + cv::Point2f(crop.cols, 0);

                cv::circle(composite, pt1, 4, color, 1, cv::LINE_AA);
                cv::circle(composite, pt2, 4, color, 1, cv::LINE_AA);
                cv::line(composite, pt1, pt2, color, 1, cv::LINE_AA);
            }

            // Stack vertically in final composite if multiple ROIs exist
            if (final_composite.empty())
            {
                final_composite = composite;
            }
            else
            {
                int maxW = std::max(final_composite.cols, composite.cols);
                cv::Mat top = cv::Mat::zeros(final_composite.rows, maxW, CV_8UC3);
                final_composite.copyTo(top(cv::Rect(0, 0, final_composite.cols, final_composite.rows)));
                cv::Mat bot = cv::Mat::zeros(composite.rows, maxW, CV_8UC3);
                composite.copyTo(bot(cv::Rect(0, 0, composite.cols, composite.rows)));
                cv::vconcat(top, bot, final_composite);
            }
        }

        cv::Mat composite = final_composite.empty() ? img1.clone() : final_composite;

        // ── Overlay a compact legend on composite result ────────────────────────
        int ly = 24;
        for (size_t idx = 0; idx < roiRects.size(); ++idx)
        {
            cv::Scalar color = ROI_COLORS[idx % ROI_COLORS.size()];
            std::string label = "ROI " + std::to_string(idx + 1);
            cv::rectangle(composite, cv::Point(8, ly - 12), cv::Point(22, ly + 2), color, -1);
            cv::putText(composite, label, cv::Point(30, ly),
                        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            ly += 24;
        }

        lblStatus->setText(
            QString("Matched %1 geometric pair(s) across %2 ROI(s).  "
                    "Ratio=%.2f  |  Contrast thr=%.3f")
                .arg(totalMatches)
                .arg(roiRects.size())
                .arg((double)ratioThresh)
                .arg((double)contrastThresh));

        // ── Push result to Panel 3 ────────────────────────────────────────────
        cv::cvtColor(composite, composite, cv::COLOR_BGR2RGB);
        QImage qout(composite.data, composite.cols, composite.rows,
                    (int)composite.step, QImage::Format_RGB888);
        lblOutputImage->setPixmap(QPixmap::fromImage(qout.copy()));
    }
    catch (const cv::Exception &e)
    {
        lblStatus->setText("Error: " + QString(e.what()));
        lblOutputImage->setText("Error: " + QString(e.what()));
    }
    catch (...)
    {
        lblStatus->setText("Error: pipeline crashed.");
        lblOutputImage->setText("Error: pipeline crashed.");
    }

    btnRunPipeline->setText("Run Custom SIFT & Match");
    btnRunPipeline->setEnabled(true);
    btnLoadImage1->setEnabled(true);
    btnLoadImage2->setEnabled(true);
}