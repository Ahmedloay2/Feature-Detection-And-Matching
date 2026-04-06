#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "SiftCore.hpp"
#include "io/image_handler.hpp"

// OpenCV — each header covers exactly the functions used below
#include <opencv2/imgcodecs.hpp>  // cv::imread
#include <opencv2/imgproc.hpp>    // cv::cvtColor, cv::resize, cv::COLOR_BGR2RGB
#include <opencv2/features2d.hpp> // cv::drawKeypoints, cv::DrawMatchesFlags
#include <opencv2/calib3d.hpp>    // cv::findHomography, cv::RANSAC

#include <QFileDialog>
#include <QApplication>
#include <QResizeEvent>
#include <omp.h>
#include <mutex>

// ── Constructor ───────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // ── Transparent overlay — must be created AFTER setupUi so it is raised
    //    above all child widgets that setupUi created.
    m_overlay = new MatchOverlay(this);
    m_overlay->lines = &matchLines;
    m_overlay->setGeometry(rect()); // cover the full window
    m_overlay->raise();             // sit on top of every other child

    // ── Debounce timer (fires 500 ms after last slider move) ─────────────────
    debounceTimer = new QTimer(this);
    debounceTimer->setSingleShot(true);
    debounceTimer->setInterval(500);
    connect(debounceTimer, &QTimer::timeout, this, &MainWindow::onDebounceTimeout);

    // ── Async watchers ────────────────────────────────────────────────────────
    connect(&watcherSift1, &QFutureWatcher<void>::finished,
            this, &MainWindow::onImg1SiftDone);
    connect(&watcherMatch, &QFutureWatcher<MatchResult>::finished,
            this, &MainWindow::onMatchDone);

    // ── Button connections ────────────────────────────────────────────────────
    connect(ui->btnLoadFullScene, &QPushButton::clicked, this, &MainWindow::onLoadFullScene);
    connect(ui->btnLoadTargetTemplate, &QPushButton::clicked, this, &MainWindow::onLoadTargetTemplate);
    connect(ui->btnRunMatch, &QPushButton::clicked, this, &MainWindow::onExecuteMatch);
    connect(ui->btnUndo, &QPushButton::clicked, this, &MainWindow::onUndoRoi);
    connect(ui->btnRedo, &QPushButton::clicked, this, &MainWindow::onRedoRoi);
    connect(ui->btnReset, &QPushButton::clicked, this, &MainWindow::onResetRoi);

    // ── Slider <-> spin sync ──────────────────────────────────────────────────
    connect(ui->sliderRatio, &QSlider::valueChanged,
            this, &MainWindow::onSiftRatioSlider);
    connect(ui->spinRatio, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onSiftRatioSpin);
    connect(ui->sliderContrast, &QSlider::valueChanged,
            this, &MainWindow::onSiftContrastSlider);
    connect(ui->spinContrast, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onSiftContrastSpin);

    // ── ROI label signals ─────────────────────────────────────────────────────
    connect(ui->lblTemplate, &InteractiveLabel::roiSelected,
            this, [this]()
            {
                matchLines.clear();
                m_overlay->update();
                if (!img1.empty() && !img2.empty())
                    ui->btnRunMatch->setEnabled(true); });
    connect(ui->lblTemplate, &InteractiveLabel::roiHistoryChanged,
            this, &MainWindow::onRoiHistoryChanged);

    ui->lblStatus->setText("Ready. Load full scene and target template to begin.");
}

MainWindow::~MainWindow() { delete ui; }

// ── Keep overlay covering the full window on resize ───────────────────────────

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    if (m_overlay)
    {
        m_overlay->setGeometry(rect());
        m_overlay->raise(); // stay on top after any layout change
    }
}

// ── Parameter sync ────────────────────────────────────────────────────────────

void MainWindow::onSiftRatioSlider(int value)
{
    currentRatioThresh = value / 100.0f;
    ui->spinRatio->blockSignals(true);
    ui->spinRatio->setValue(currentRatioThresh);
    ui->spinRatio->blockSignals(false);
    matchLines.clear();
    m_overlay->update();
}

void MainWindow::onSiftRatioSpin(double value)
{
    currentRatioThresh = (float)value;
    ui->sliderRatio->blockSignals(true);
    ui->sliderRatio->setValue(qRound(value * 100));
    ui->sliderRatio->blockSignals(false);
    matchLines.clear();
    m_overlay->update();
}

void MainWindow::onSiftContrastSlider(int value)
{
    currentContrastThresh = value / 1000.0f;
    ui->spinContrast->blockSignals(true);
    ui->spinContrast->setValue(currentContrastThresh);
    ui->spinContrast->blockSignals(false);
    if (!img1.empty())
    {
        matchLines.clear();
        m_overlay->update();
        debounceTimer->start();
    }
}

void MainWindow::onSiftContrastSpin(double value)
{
    currentContrastThresh = (float)value;
    ui->sliderContrast->blockSignals(true);
    ui->sliderContrast->setValue(qRound(value * 1000));
    ui->sliderContrast->blockSignals(false);
    if (!img1.empty())
    {
        matchLines.clear();
        m_overlay->update();
        debounceTimer->start();
    }
}

// ── Debounce -> re-run SIFT on img1 ──────────────────────────────────────────

void MainWindow::onDebounceTimeout()
{
    if (!img1.empty())
        runSiftOnImg1Async();
}

// ── Load images ───────────────────────────────────────────────────────────────

void MainWindow::downscaleIfNeeded(cv::Mat &img)
{
    const int MAX_DIM = 1024;
    if (img.cols > MAX_DIM || img.rows > MAX_DIM)
    {
        float s = std::min((float)MAX_DIM / img.cols, (float)MAX_DIM / img.rows);
        cv::resize(img, img, cv::Size(), s, s, cv::INTER_AREA);
    }
}

void MainWindow::onLoadFullScene()
{
    QString fn = QFileDialog::getOpenFileName(
        this, "Open Full Scene", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fn.isEmpty())
        return;

    img1 = cv::imread(fn.toStdString());
    if (img1.empty())
    {
        ui->lblStatus->setText("Failed to load image.");
        return;
    }
    downscaleIfNeeded(img1);

    cv::Mat rgb;
    cv::cvtColor(img1, rgb, cv::COLOR_BGR2RGB);
    ui->lblOutputImage->setPixmap(
        QPixmap::fromImage(
            QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step,
                   QImage::Format_RGB888)
                .copy()));

    ui->lblStatus->setText(
        QString("Full scene loaded (%1x%2). Extracting SIFT features...")
            .arg(img1.cols)
            .arg(img1.rows));

    matchLines.clear();
    m_overlay->update();

    runSiftOnImg1Async();
}

void MainWindow::onLoadTargetTemplate()
{
    QString fn = QFileDialog::getOpenFileName(
        this, "Open Target Template", "", "Images (*.png *.jpg *.jpeg *.bmp)");
    if (fn.isEmpty())
        return;

    img2 = cv::imread(fn.toStdString());
    if (img2.empty())
    {
        ui->lblStatus->setText("Failed to load image.");
        return;
    }
    downscaleIfNeeded(img2);

    cv::Mat rgb;
    cv::cvtColor(img2, rgb, cv::COLOR_BGR2RGB);
    ui->lblTemplate->setPixmap(
        QPixmap::fromImage(
            QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step,
                   QImage::Format_RGB888)
                .copy()));

    ui->lblTemplate->clearROI();
    matchLines.clear();
    m_overlay->update();

    ui->lblStatus->setText(
        QString("Target loaded (%1x%2). Draw ROI box(es), then click Run Matches.")
            .arg(img2.cols)
            .arg(img2.rows));

    ui->btnRunMatch->setEnabled(
        !img1.empty() && !ui->lblTemplate->getSelectedROIs().empty());
}

// ── Async SIFT extraction on img1 ─────────────────────────────────────────────

void MainWindow::runSiftOnImg1Async()
{
    if (watcherSift1.isRunning())
    {
        sift1RerunPending = true;
        return;
    }

    sift1RerunPending = false;

    float ct = currentContrastThresh;
    cv::Mat imgCopy = img1.clone();

    pendingKp1 = std::make_shared<std::vector<cv::KeyPoint>>();
    pendingDesc1 = std::make_shared<cv::Mat>();

    auto kpRef = pendingKp1;
    auto descRef = pendingDesc1;

    watcherSift1.setFuture(QtConcurrent::run(
        [imgCopy, ct, kpRef, descRef]()
        {
            cv_assign::SiftProcessor::extractFeatures(imgCopy, *kpRef, *descRef, ct);
        }));
}

void MainWindow::onImg1SiftDone()
{
    if (sift1RerunPending)
    {
        runSiftOnImg1Async();
        return;
    }

    kp1 = *pendingKp1;
    desc1 = *pendingDesc1;

    displayImg1WithKeypoints();

    ui->lblStatus->setText(
        QString("Scene: %1 keypoints detected (contrast = %2).")
            .arg(kp1.size())
            .arg(currentContrastThresh, 0, 'f', 3));
}

void MainWindow::displayImg1WithKeypoints()
{
    if (img1.empty() || kp1.empty())
        return;

    cv::Mat vis;
    cv::drawKeypoints(img1, kp1, vis, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::cvtColor(vis, vis, cv::COLOR_BGR2RGB);
    ui->lblOutputImage->setPixmap(
        QPixmap::fromImage(
            QImage(vis.data, vis.cols, vis.rows,
                   (int)vis.step, QImage::Format_RGB888)
                .copy()));
}

// ── ROI helpers ───────────────────────────────────────────────────────────────

std::vector<cv::Rect> MainWindow::getAllRoisInImg2Space() const
{
    auto qrois = ui->lblTemplate->getSelectedROIs();
    std::vector<cv::Rect> result;
    result.reserve(qrois.size());

    for (const auto &r : qrois)
    {
        int x = std::max(0, r.x());
        int y = std::max(0, r.y());
        int w = std::min(r.width(), img2.cols - x);
        int h = std::min(r.height(), img2.rows - y);
        if (w > 0 && h > 0)
            result.push_back(cv::Rect(x, y, w, h));
    }
    return result;
}

// ── Matching ──────────────────────────────────────────────────────────────────

void MainWindow::onExecuteMatch()
{
    if (img1.empty() || img2.empty())
        return;
    if (kp1.empty() || desc1.empty())
    {
        ui->lblStatus->setText("Re-run SIFT first.");
        return;
    }
    if (watcherMatch.isRunning())
        return;

    auto allRois = getAllRoisInImg2Space();
    if (allRois.empty())
    {
        ui->lblStatus->setText("Draw at least one ROI box on the target first.");
        return;
    }

    ui->btnRunMatch->setEnabled(false);
    ui->btnLoadFullScene->setEnabled(false);
    ui->btnLoadTargetTemplate->setEnabled(false);
    ui->lblStatus->setText(
        QString("Running SIFT matching on %1 ROI box(es)...").arg(allRois.size()));

    matchLines.clear();
    m_overlay->update();

    // ── Assign one distinct color per ROI box ─────────────────────────────────
    static const std::vector<QColor> palette = {
        QColor(255, 80, 80, 200),   // 0 - red
        QColor(80, 200, 255, 200),  // 1 - cyan
        QColor(100, 255, 100, 200), // 2 - green
        QColor(255, 210, 50, 200),  // 3 - yellow
        QColor(200, 100, 255, 200), // 4 - purple
        QColor(255, 140, 0, 200),   // 5 - orange
    };

    pendingRoiColors.clear();
    for (size_t i = 0; i < allRois.size(); ++i)
        pendingRoiColors.push_back(palette[i % palette.size()]);

    // Thread-safe snapshots
    cv::Mat d1copy = desc1.clone();
    std::vector<cv::KeyPoint> kp1copy = kp1;
    cv::Mat img2copy = img2.clone();
    float ratio = currentRatioThresh;
    float ct = currentContrastThresh;

    watcherMatch.setFuture(QtConcurrent::run(
        [allRois, img2copy, d1copy, kp1copy, ratio, ct]() -> MatchResult
        {
            MatchResult combined;
            combined.valid = true;

            int kp2Offset = 0;

            for (int roiIdx = 0; roiIdx < (int)allRois.size(); ++roiIdx)
            {
                const cv::Rect &roi = allRois[roiIdx];
                cv::Mat roiImg = img2copy(roi).clone();

                // Extract SIFT on this ROI sub-image
                std::vector<cv::KeyPoint> kp2roi;
                cv::Mat desc2;
                cv_assign::SiftProcessor::extractFeatures(roiImg, kp2roi, desc2, ct);

                if (desc2.empty() || desc2.rows < 2 || d1copy.empty())
                {
                    kp2Offset += (int)kp2roi.size();
                    continue;
                }

                // Shift keypoints from sub-image space -> full img2 space
                for (auto &kp : kp2roi)
                {
                    kp.pt.x += roi.x;
                    kp.pt.y += roi.y;
                    combined.kp2.push_back(kp);
                }

                // Custom SSD matching with Lowe ratio test
                const int N = desc2.rows;
                const int M = d1copy.rows;

                std::vector<cv::DMatch> raw;
                raw.reserve(N);
                std::mutex mtx;

#pragma omp parallel for schedule(dynamic, 32)
                for (int i = 0; i < N; ++i)
                {
                    const float *q = desc2.ptr<float>(i);
                    float best = 1e30f, second = 1e30f;
                    int bestIdx = -1, secondIdx = -1;

                    for (int j = 0; j < M; ++j)
                    {
                        const float *t = d1copy.ptr<float>(j);
                        float ssd = 0.f;

                        for (int d = 0; d < 128; d += 8)
                        {
                            float d0 = q[d + 0] - t[d + 0], d1a = q[d + 1] - t[d + 1],
                                  d2 = q[d + 2] - t[d + 2], d3 = q[d + 3] - t[d + 3],
                                  d4 = q[d + 4] - t[d + 4], d5 = q[d + 5] - t[d + 5],
                                  d6 = q[d + 6] - t[d + 6], d7 = q[d + 7] - t[d + 7];
                            ssd += d0 * d0 + d1a * d1a + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
                        }

                        if (ssd < best)
                        {
                            second = best;
                            secondIdx = bestIdx;
                            best = ssd;
                            bestIdx = j;
                        }
                        else if (ssd < second)
                        {
                            second = ssd;
                            secondIdx = j;
                        }
                    }

                    if (bestIdx >= 0 && secondIdx >= 0 &&
                        std::sqrt(best) < ratio * std::sqrt(second))
                    {
                        // queryIdx: global index into combined.kp2
                        // imgIdx:   reused as ROI index tag for color lookup
                        cv::DMatch dm(kp2Offset + i, bestIdx, std::sqrt(best));
                        dm.imgIdx = roiIdx;
                        std::lock_guard<std::mutex> lk(mtx);
                        raw.push_back(dm);
                    }
                }

                // RANSAC homography filtering per ROI
                if ((int)raw.size() >= 4)
                {
                    std::vector<cv::Point2f> src, dst;
                    src.reserve(raw.size());
                    dst.reserve(raw.size());
                    for (const auto &m : raw)
                    {
                        src.push_back(combined.kp2[m.queryIdx].pt);
                        dst.push_back(kp1copy[m.trainIdx].pt);
                    }
                    std::vector<uchar> inliers;
                    cv::findHomography(src, dst, cv::RANSAC, 3.0, inliers);
                    for (size_t k = 0; k < inliers.size(); ++k)
                        if (inliers[k])
                            combined.matches.push_back(raw[k]);
                }
                else
                {
                    // Too few for RANSAC — keep all raw matches
                    for (const auto &m : raw)
                        combined.matches.push_back(m);
                }

                kp2Offset += (int)kp2roi.size();
            }

            return combined;
        }));
}

// ── Match done -> build match lines in overlay coordinates ────────────────────

void MainWindow::onMatchDone()
{
    ui->btnRunMatch->setEnabled(true);
    ui->btnLoadFullScene->setEnabled(true);
    ui->btnLoadTargetTemplate->setEnabled(true);

    MatchResult res = watcherMatch.result();
    if (!res.valid || res.matches.empty())
    {
        ui->lblStatus->setText(
            "No matches found. Try lowering the contrast threshold or raising the ratio.");
        return;
    }

    // ── Convert image-space keypoints -> overlay widget coordinates ───────────
    //
    //  The overlay covers the full MainWindow rect (0,0,w,h).
    //  mapTo(this, pt) converts a label-local point to MainWindow coords,
    //  which are identical to overlay coords since the overlay starts at (0,0).
    //
    //  Both labels use scaledContents=true so:
    //    image pixel (kp.x, kp.y) -> label local (kp.x * lblW/imgW, kp.y * lblH/imgH)

    auto toLabel = [](const cv::Point2f &pt, const QLabel *lbl,
                      int imgW, int imgH) -> QPoint
    {
        float sx = (float)lbl->width() / (float)imgW;
        float sy = (float)lbl->height() / (float)imgH;
        return QPoint(qRound(pt.x * sx), qRound(pt.y * sy));
    };

    matchLines.clear();
    matchLines.reserve(res.matches.size());

    for (const auto &m : res.matches)
    {
        QPoint lblSrc = toLabel(res.kp2[m.queryIdx].pt,
                                ui->lblTemplate, img2.cols, img2.rows);
        QPoint lblDst = toLabel(kp1[m.trainIdx].pt,
                                ui->lblOutputImage, img1.cols, img1.rows);

        // mapTo(this) converts label-local -> MainWindow == overlay coords
        QPointF overlaySrc = ui->lblTemplate->mapTo(this, lblSrc);
        QPointF overlayDst = ui->lblOutputImage->mapTo(this, lblDst);

        int roiIdx = m.imgIdx;
        QColor col = (roiIdx >= 0 && roiIdx < (int)pendingRoiColors.size())
                         ? pendingRoiColors[roiIdx]
                         : QColor(50, 220, 120, 200);

        matchLines.push_back({overlaySrc, overlayDst, col});
    }

    m_overlay->raise();  // ensure nothing has been stacked above it since
    m_overlay->update(); // trigger MatchOverlay::paintEvent

    // Status bar with per-ROI breakdown
    int numRois = (int)pendingRoiColors.size();
    QString roiDetail;
    for (int r = 0; r < numRois; ++r)
    {
        int cnt = 0;
        for (const auto &m : res.matches)
            if (m.imgIdx == r)
                ++cnt;
        if (numRois > 1)
            roiDetail += QString(" | ROI %1: %2").arg(r + 1).arg(cnt);
    }

    ui->lblStatus->setText(
        QString("Matched %1 inlier pair(s).  Ratio = %2  |  Contrast = %3%4")
            .arg(res.matches.size())
            .arg(currentRatioThresh, 0, 'f', 2)
            .arg(currentContrastThresh, 0, 'f', 3)
            .arg(roiDetail));
}

// ── ROI history controls ──────────────────────────────────────────────────────

void MainWindow::onUndoRoi()
{
    ui->lblTemplate->undoRoi();
    matchLines.clear();
    m_overlay->update();
}

void MainWindow::onRedoRoi()
{
    ui->lblTemplate->redoRoi();
    matchLines.clear();
    m_overlay->update();
}

void MainWindow::onResetRoi()
{
    ui->lblTemplate->clearROI();
    matchLines.clear();
    pendingRoiColors.clear();
    m_overlay->update();
    ui->btnRunMatch->setEnabled(false);
}

void MainWindow::onRoiHistoryChanged()
{
    ui->btnUndo->setEnabled(ui->lblTemplate->canUndo());
    ui->btnRedo->setEnabled(ui->lblTemplate->canRedo());
}