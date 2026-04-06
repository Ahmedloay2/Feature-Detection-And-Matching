#pragma once

#include <QMainWindow>
#include <QTimer>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrent>
#include <QColor>
#include <QPointF>
#include <QPainter>
#include <QPen>

#include <opencv2/core.hpp>
#include <vector>
#include <memory>

#include "interactive_label.h"

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

// ── Data structures shared between matching thread and UI ─────────────────────

struct MatchResult
{
    std::vector<cv::KeyPoint> kp2;   // all keypoints from all ROIs (img2 space)
    std::vector<cv::DMatch> matches; // inlier matches; DMatch.imgIdx = ROI index
    bool valid = false;
};

struct MatchLine
{
    QPointF start; // point on lblTemplate   (MainWindow coords)
    QPointF end;   // point on lblOutputImage (MainWindow coords)
    QColor color;  // per-ROI color
};

// ── Transparent overlay — drawn on top of all child widgets ───────────────────
//
//  Lives as a child of MainWindow, always covers its full rect, and is always
//  raised above siblings.  Mouse events pass straight through it so the labels
//  underneath still receive clicks for ROI drawing.

class MatchOverlay : public QWidget
{
    Q_OBJECT
public:
    explicit MatchOverlay(QWidget *parent)
        : QWidget(parent)
    {
        setAttribute(Qt::WA_TransparentForMouseEvents); // clicks reach labels below
        setAttribute(Qt::WA_NoSystemBackground);        // skip default fill
        setStyleSheet("background: transparent;");
    }

    // Pointer into MainWindow's matchLines vector — no copies needed
    const std::vector<MatchLine> *lines = nullptr;

protected:
    void paintEvent(QPaintEvent *) override
    {
        if (!lines || lines->empty())
            return;

        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);

        for (const auto &ln : *lines)
        {
            QPen pen(ln.color, 1.5f, Qt::SolidLine, Qt::RoundCap);
            p.setPen(pen);
            p.drawLine(ln.start, ln.end);

            p.setPen(Qt::NoPen);
            p.setBrush(ln.color);
            p.drawEllipse(ln.start, 3, 3); // dot on template side
            p.drawEllipse(ln.end, 3, 3);   // dot on scene side
        }
    }
};

// ── MainWindow ────────────────────────────────────────────────────────────────

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    // Keep the overlay covering the full window whenever it resizes
    void resizeEvent(QResizeEvent *event) override;

private slots:
    // Buttons
    void onLoadFullScene();
    void onLoadTargetTemplate();
    void onExecuteMatch();
    void onUndoRoi();
    void onRedoRoi();
    void onResetRoi();

    // Slider <-> spin sync
    void onSiftRatioSlider(int value);
    void onSiftRatioSpin(double value);
    void onSiftContrastSlider(int value);
    void onSiftContrastSpin(double value);

    // Async callbacks
    void onDebounceTimeout();
    void onImg1SiftDone();
    void onMatchDone();

    // ROI history state -> enable/disable undo/redo buttons
    void onRoiHistoryChanged();

private:
    // ── Helpers ───────────────────────────────────────────────────────────────
    void downscaleIfNeeded(cv::Mat &img);
    void runSiftOnImg1Async();
    void displayImg1WithKeypoints();

    // Returns ALL valid ROI rects in img2 pixel space
    std::vector<cv::Rect> getAllRoisInImg2Space() const;

    // ── UI ────────────────────────────────────────────────────────────────────
    Ui::MainWindow *ui;
    MatchOverlay *m_overlay = nullptr; // transparent drawing surface on top

    // ── Images ────────────────────────────────────────────────────────────────
    cv::Mat img1; // full scene
    cv::Mat img2; // target template

    // ── SIFT data for full scene (img1) ───────────────────────────────────────
    std::vector<cv::KeyPoint> kp1;
    cv::Mat desc1;

    // Pending results while async extraction runs
    std::shared_ptr<std::vector<cv::KeyPoint>> pendingKp1;
    std::shared_ptr<cv::Mat> pendingDesc1;

    // ── Match visualisation ───────────────────────────────────────────────────
    std::vector<MatchLine> matchLines;
    std::vector<QColor> pendingRoiColors; // one entry per drawn ROI box

    // ── Async workers ─────────────────────────────────────────────────────────
    QFutureWatcher<void> watcherSift1;
    QFutureWatcher<MatchResult> watcherMatch;
    bool sift1RerunPending = false;

    // ── Parameters ────────────────────────────────────────────────────────────
    float currentRatioThresh = 0.75f;
    float currentContrastThresh = 0.007f;

    // Debounce timer so slider drags don't spam SIFT re-extraction
    QTimer *debounceTimer = nullptr;
};