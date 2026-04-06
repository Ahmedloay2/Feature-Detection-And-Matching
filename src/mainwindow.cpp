#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "SiftCore.hpp"
#include "io/image_handler.hpp"

#include <QFileDialog>
#include <QApplication>
#include <QPainter>
#include <omp.h>
#include <mutex>

// ── Constructor ───────────────────────────────────────────────────────────────
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // ── Debounce timer (fires 500ms after last slider move) ──────────────────
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

    // ── Slider ↔ spin sync ────────────────────────────────────────────────────
    connect(ui->sliderRatio, &QSlider::valueChanged,
            this, &MainWindow::onSiftRatioSlider);
    connect(ui->spinRatio, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onSiftRatioSpin);
    connect(ui->sliderContrast, &QSlider::valueChanged,
            this, &MainWindow::onSiftContrastSlider);
    connect(ui->spinContrast, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &MainWindow::onSiftContrastSpin);

    // ── ROI label signals ─────────────────────────────────────────────────────
    // roiSelected fires when a box is completed → may trigger re-match if run btn was used
    connect(ui->lblTemplate, &InteractiveLabel::roiSelected,
            this, [this]()
            {
                matchLines.clear();
                update();
                // Enable Run button if both images ready
                if (!img1.empty() && !img2.empty())
                    ui->btnRunMatch->setEnabled(true); });
    connect(ui->lblTemplate, &InteractiveLabel::roiHistoryChanged,
            this, &MainWindow::onRoiHistoryChanged);

    ui->lblStatus->setText("Ready. Load full scene and target template to begin.");
}

MainWindow::~MainWindow() { delete ui; }

// ── Parameter sync ────────────────────────────────────────────────────────────

void MainWindow::onSiftRatioSlider(int value)
{
    currentRatioThresh = value / 100.0f;
    ui->spinRatio->blockSignals(true);
    ui->spinRatio->setValue(currentRatioThresh);
    ui->spinRatio->blockSignals(false);
    // Ratio only affects matching — no re-extraction needed.
    matchLines.clear();
    update();
}

void MainWindow::onSiftRatioSpin(double value)
{
    currentRatioThresh = (float)value;
    ui->sliderRatio->blockSignals(true);
    ui->sliderRatio->setValue(qRound(value * 100));
    ui->sliderRatio->blockSignals(false);
    matchLines.clear();
    update();
}

void MainWindow::onSiftContrastSlider(int value)
{
    currentContrastThresh = value / 1000.0f;
    ui->spinContrast->blockSignals(true);
    ui->spinContrast->setValue(currentContrastThresh);
    ui->spinContrast->blockSignals(false);
    // Contrast affects extraction → debounced re-run on img1
    if (!img1.empty())
    {
        matchLines.clear();
        update();
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
        update();
        debounceTimer->start();
    }
}

// ── Debounce → re-run SIFT on img1 ───────────────────────────────────────────
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

    // Show immediately (without circles — they come after SIFT)
    cv::Mat rgb;
    cv::cvtColor(img1, rgb, cv::COLOR_BGR2RGB);
    ui->lblOutputImage->setPixmap(
        QPixmap::fromImage(QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step,
                                  QImage::Format_RGB888)
                               .copy()));

    ui->lblStatus->setText(
        QString("Full scene loaded (%1×%2). Extracting SIFT features…")
            .arg(img1.cols)
            .arg(img1.rows));

    matchLines.clear();
    update();

    // Auto-start async SIFT extraction
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

    // Show clean image — NO SIFT circles on target
    cv::Mat rgb;
    cv::cvtColor(img2, rgb, cv::COLOR_BGR2RGB);
    ui->lblTemplate->setPixmap(
        QPixmap::fromImage(QImage(rgb.data, rgb.cols, rgb.rows, (int)rgb.step,
                                  QImage::Format_RGB888)
                               .copy()));
    ui->lblTemplate->clearROI();
    matchLines.clear();
    update();

    ui->lblStatus->setText(
        QString("Target loaded (%1×%2). Draw ROI box(es), then click Run Matches.")
            .arg(img2.cols)
            .arg(img2.rows));

    // Enable Run if scene also loaded and ROI drawn, otherwise wait
    ui->btnRunMatch->setEnabled(!img1.empty() && !ui->lblTemplate->getSelectedROIs().empty());
}

// ── Async SIFT extraction on img1 ─────────────────────────────────────────────
//
//  Pattern: only one extraction runs at a time.
//  If a new run is requested while one is in flight, we set a flag and
//  re-queue when the current run finishes.

void MainWindow::runSiftOnImg1Async()
{
    if (watcherSift1.isRunning())
    {
        // Mark that another run is needed; onImg1SiftDone will pick it up.
        sift1RerunPending = true;
        return;
    }

    sift1RerunPending = false;

    float ct = currentContrastThresh;
    cv::Mat imgCopy = img1.clone(); // thread-safe copy — main thread keeps img1

    pendingKp1 = std::make_shared<std::vector<cv::KeyPoint>>();
    pendingDesc1 = std::make_shared<cv::Mat>();

    // Capture shared_ptrs by value so the lambda owns them.
    auto kpRef = pendingKp1;
    auto descRef = pendingDesc1;

    watcherSift1.setFuture(QtConcurrent::run([imgCopy, ct, kpRef, descRef]()
                                             { cv_assign::SiftProcessor::extractFeatures(imgCopy, *kpRef, *descRef, ct); }));
}

void MainWindow::onImg1SiftDone()
{
    // If slider moved again while this was running, kick off another extraction
    // and wait — don't update the display with stale parameters.
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
        QPixmap::fromImage(QImage(vis.data, vis.cols, vis.rows,
                                  (int)vis.step, QImage::Format_RGB888)
                               .copy()));
}

// ── ROI helper ────────────────────────────────────────────────────────────────

cv::Rect MainWindow::getRoiInImg2Space() const
{
    auto rois = ui->lblTemplate->getSelectedROIs();
    if (rois.empty() || img2.empty())
        return cv::Rect();

    // Use the last-drawn ROI box. getSelectedROIs() returns rects in pixmap
    // pixel space which == img2 space (since we set the pixmap without pre-scaling).
    QRect r = rois.back();
    int x = std::max(0, r.x());
    int y = std::max(0, r.y());
    int w = std::min(r.width(), img2.cols - x);
    int h = std::min(r.height(), img2.rows - y);
    if (w <= 0 || h <= 0)
        return cv::Rect();
    return cv::Rect(x, y, w, h);
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

    cv::Rect roi = getRoiInImg2Space();
    if (roi.area() <= 0)
    {
        ui->lblStatus->setText("Draw an ROI box on the target first.");
        return;
    }

    ui->btnRunMatch->setEnabled(false);
    ui->btnLoadFullScene->setEnabled(false);
    ui->btnLoadTargetTemplate->setEnabled(false);
    ui->lblStatus->setText("Running SIFT matching…");
    matchLines.clear();
    update();

    // Snapshot everything the worker thread needs — no shared mutable state.
    cv::Mat roiImg = img2(roi).clone();
    int ox = roi.x, oy = roi.y;
    cv::Mat d1copy = desc1.clone();
    std::vector<cv::KeyPoint> kp1copy = kp1;
    float ratio = currentRatioThresh;
    float ct = currentContrastThresh;

    watcherMatch.setFuture(QtConcurrent::run(
        [roiImg, ox, oy, d1copy, kp1copy, ratio, ct]() -> MatchResult
        {
            MatchResult res;

            // ── Extract SIFT on the target ROI sub-image ─────────────────────
            cv::Mat desc2;
            cv_assign::SiftProcessor::extractFeatures(roiImg, res.kp2, desc2, ct);

            if (desc2.empty() || desc2.rows < 2 || d1copy.empty())
                return res;

            // Offset kp2 so they are in img2 space (not sub-image space).
            for (auto &kp : res.kp2)
            {
                kp.pt.x += ox;
                kp.pt.y += oy;
            }

            // ── Custom SSD matching with Lowe ratio test ─────────────────────
            // We use SSD (as required by the assignment) and apply Lowe's
            // ratio test to distinguish genuine from ambiguous matches.
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
                    // Unrolled 128-dim SSD
                    for (int d = 0; d < 128; d += 8)
                    {
                        float d0 = q[d] - t[d], d1 = q[d + 1] - t[d + 1], d2 = q[d + 2] - t[d + 2], d3 = q[d + 3] - t[d + 3];
                        float d4 = q[d + 4] - t[d + 4], d5 = q[d + 5] - t[d + 5], d6 = q[d + 6] - t[d + 6], d7 = q[d + 7] - t[d + 7];
                        ssd += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
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
                    std::lock_guard<std::mutex> lk(mtx);
                    raw.push_back(cv::DMatch(i, bestIdx, std::sqrt(best)));
                }
            }

            // ── RANSAC homography filtering to remove geometric outliers ─────
            if (raw.size() >= 4)
            {
                std::vector<cv::Point2f> src, dst;
                for (auto &m : raw)
                {
                    src.push_back(res.kp2[m.queryIdx].pt);
                    dst.push_back(kp1copy[m.trainIdx].pt);
                }
                std::vector<uchar> inliers;
                cv::findHomography(src, dst, cv::RANSAC, 3.0, inliers);
                for (size_t i = 0; i < inliers.size(); i++)
                    if (inliers[i])
                        res.matches.push_back(raw[i]);
            }
            else
            {
                res.matches = raw;
            }

            res.valid = true;
            return res;
        }));
}

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

    // ── Build match lines in MainWindow widget coordinates ───────────────────
    //
    //  Both labels use scaledContents=true, so:
    //    pixel (kp.x, kp.y) in image → widget pos (kp.x * w/W, kp.y * h/H)
    //  where w/h = label widget size, W/H = image size.
    //  Then mapTo(this, pos) converts to MainWindow space for paintEvent.

    auto toWidget = [](const cv::Point2f &pt, const QLabel *lbl,
                       int imgW, int imgH) -> QPointF
    {
        float sx = (float)lbl->width() / imgW;
        float sy = (float)lbl->height() / imgH;
        return QPointF(pt.x * sx, pt.y * sy);
    };

    matchLines.clear();
    matchLines.reserve(res.matches.size());

    for (const auto &m : res.matches)
    {
        // Start: keypoint in target (lblTemplate), in img2 space
        QPointF wSrc = toWidget(res.kp2[m.queryIdx].pt,
                                ui->lblTemplate, img2.cols, img2.rows);
        // End:   keypoint in full scene (lblOutputImage), in img1 space
        QPointF wDst = toWidget(kp1[m.trainIdx].pt,
                                ui->lblOutputImage, img1.cols, img1.rows);

        QPointF mainSrc = ui->lblTemplate->mapTo(this, wSrc.toPoint());
        QPointF mainDst = ui->lblOutputImage->mapTo(this, wDst.toPoint());
        matchLines.push_back({mainSrc, mainDst});
    }

    update(); // trigger paintEvent to draw arrows

    ui->lblStatus->setText(
        QString("Matched %1 inlier pair(s).  Ratio = %2  |  Contrast = %3")
            .arg(res.matches.size())
            .arg(currentRatioThresh, 0, 'f', 2)
            .arg(currentContrastThresh, 0, 'f', 3));
}

// ── paintEvent — draw match arrows ───────────────────────────────────────────
//
//  Arrows run from the target template (left panel) → full scene (right panel).

void MainWindow::paintEvent(QPaintEvent *event)
{
    QMainWindow::paintEvent(event);
    if (matchLines.empty())
        return;

    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    // Slightly translucent green-yellow arrows
    QPen linePen(QColor(50, 220, 120, 160), 1.2f, Qt::SolidLine, Qt::RoundCap);
    p.setPen(linePen);

    for (const auto &ln : matchLines)
    {
        // Arrow body
        p.drawLine(ln.start, ln.end);

        // Source dot (template side) — cyan
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(0, 200, 255, 200));
        p.drawEllipse(ln.start, 3, 3);

        // Destination dot (scene side) — orange
        p.setBrush(QColor(255, 140, 0, 200));
        p.drawEllipse(ln.end, 3, 3);

        p.setPen(linePen);
    }
}

// ── ROI history controls ──────────────────────────────────────────────────────

void MainWindow::onUndoRoi()
{
    ui->lblTemplate->undoRoi();
    matchLines.clear();
    update();
}

void MainWindow::onRedoRoi()
{
    ui->lblTemplate->redoRoi();
    matchLines.clear();
    update();
}

void MainWindow::onResetRoi()
{
    ui->lblTemplate->clearROI();
    matchLines.clear();
    update();
    ui->btnRunMatch->setEnabled(false);
}

void MainWindow::onRoiHistoryChanged()
{
    ui->btnUndo->setEnabled(ui->lblTemplate->canUndo());
    ui->btnRedo->setEnabled(ui->lblTemplate->canRedo());
}