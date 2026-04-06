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

// ── Matching algorithm selector ───────────────────────────────────────────────

enum class MatchAlgo
{
    SSD, // Sum of Squared Differences  — lower distance = better
    NCC  // Normalized Cross-Correlation — higher similarity = better
};

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

class MatchOverlay : public QWidget
{
    Q_OBJECT
public:
    explicit MatchOverlay(QWidget *parent)
        : QWidget(parent)
    {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_NoSystemBackground);
        setStyleSheet("background: transparent;");
    }

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
            p.drawEllipse(ln.start, 3, 3);
            p.drawEllipse(ln.end, 3, 3);
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
    void resizeEvent(QResizeEvent *event) override;

private slots:
    void onLoadFullScene();
    void onLoadTargetTemplate();
    void onExecuteMatch();
    void onUndoRoi();
    void onRedoRoi();
    void onResetRoi();

    void onSiftRatioSlider(int value);
    void onSiftRatioSpin(double value);
    void onSiftContrastSlider(int value);
    void onSiftContrastSpin(double value);

    void onMatchAlgoChanged(int index); // combo box selection

    void onDebounceTimeout();
    void onImg1SiftDone();
    void onMatchDone();

    void onRoiHistoryChanged();

private:
    void downscaleIfNeeded(cv::Mat &img);
    void runSiftOnImg1Async();
    void displayImg1WithKeypoints();
    std::vector<cv::Rect> getAllRoisInImg2Space() const;

    // ── UI ────────────────────────────────────────────────────────────────────
    Ui::MainWindow *ui;
    MatchOverlay *m_overlay = nullptr;

    // ── Images ────────────────────────────────────────────────────────────────
    cv::Mat img1;
    cv::Mat img2;

    // ── SIFT data ─────────────────────────────────────────────────────────────
    std::vector<cv::KeyPoint> kp1;
    cv::Mat desc1;

    std::shared_ptr<std::vector<cv::KeyPoint>> pendingKp1;
    std::shared_ptr<cv::Mat> pendingDesc1;

    // ── Match visualisation ───────────────────────────────────────────────────
    std::vector<MatchLine> matchLines;
    std::vector<QColor> pendingRoiColors;

    // ── Async workers ─────────────────────────────────────────────────────────
    QFutureWatcher<void> watcherSift1;
    QFutureWatcher<MatchResult> watcherMatch;
    bool sift1RerunPending = false;

    // ── Parameters ────────────────────────────────────────────────────────────
    float currentRatioThresh = 0.75f;
    float currentContrastThresh = 0.007f;
    MatchAlgo currentMatchAlgo = MatchAlgo::SSD;

    QTimer *debounceTimer = nullptr;
};