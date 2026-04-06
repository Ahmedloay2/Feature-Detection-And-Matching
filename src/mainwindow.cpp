#include "mainwindow.h"
#include "SiftCore.hpp"
#include "Timer.hpp"
#include "io/image_handler.hpp"

#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QGroupBox>

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

        // ── Build the annotated Image-1 result ────────────────────────────────
        cv::Mat resultImg = img1.clone(); // Panel 3 = Image 1 only

        cv::BFMatcher matcher(cv::NORM_L2);
        int totalMatches = 0;

        for (size_t idx = 0; idx < roiRects.size(); ++idx)
        {
            const auto &roi_kp = roiKps[idx];
            const auto &roi_desc = roiDescs[idx];
            const auto &roi = roiRects[idx];
            cv::Scalar color = ROI_COLORS[idx % ROI_COLORS.size()];

            if (roi_desc.empty() || roi_kp.empty() ||
                roi_desc.type() != CV_32F || roi_desc.rows < 2)
                continue;

            // kNN match: ROI descriptors (query) vs Image-1 descriptors (train)
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(roi_desc, desc1, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            for (auto &m : knn_matches)
                if (m.size() > 1 && m[0].distance < ratioThresh * m[1].distance)
                    good_matches.push_back(m[0]);

            totalMatches += (int)good_matches.size();

            // ── Draw on Image 1 (Panel 3) ─────────────────────────────────────
            // Filled circle at each matched keypoint location in Image 1
            // Size of circle reflects the keypoint scale for visual clarity
            for (const auto &match : good_matches)
            {
                cv::Point2f pt = kp1[match.trainIdx].pt;
                float r = std::max(4.0f, kp1[match.trainIdx].size / 4.0f);
                cv::circle(resultImg, pt, (int)r, color, 2, cv::LINE_AA);
                cv::circle(resultImg, pt, 2, color, -1, cv::LINE_AA); // centre dot
            }

            // ── Annotate Image 2 panel to show which ROI matched how many ─────
            // (draw the ROI border + small count label onto lblImage2's pixmap)
        }

        // ── Overlay a compact legend on Image 1 result ────────────────────────
        int ly = 18;
        for (size_t idx = 0; idx < roiRects.size(); ++idx)
        {
            cv::Scalar color = ROI_COLORS[idx % ROI_COLORS.size()];
            std::string label = "ROI " + std::to_string(idx + 1);
            cv::rectangle(resultImg, cv::Point(8, ly - 12), cv::Point(22, ly + 2), color, -1);
            cv::putText(resultImg, label, cv::Point(28, ly),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            ly += 20;
        }

        lblStatus->setText(
            QString("Matched %1 pair(s) across %2 ROI(s).  "
                    "Ratio=%.2f  |  Contrast thr=%.3f")
                .arg(totalMatches)
                .arg(roiRects.size())
                .arg((double)ratioThresh)
                .arg((double)contrastThresh));

        // ── Push result to Panel 3 ────────────────────────────────────────────
        cv::cvtColor(resultImg, resultImg, cv::COLOR_BGR2RGB);
        QImage qout(resultImg.data, resultImg.cols, resultImg.rows,
                    (int)resultImg.step, QImage::Format_RGB888);
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