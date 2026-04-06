#pragma once

#include <QMainWindow>
#include <QTimer>
#include <QFutureWatcher>
#include <QtConcurrent>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void paintEvent(QPaintEvent *event) override;

private slots:
    // ── Image loading ─────────────────────────────────────────────────────────
    void onLoadFullScene();
    void onLoadTargetTemplate();

    // ── Slider / spin sync ────────────────────────────────────────────────────
    void onSiftRatioSlider(int value);
    void onSiftRatioSpin(double value);
    void onSiftContrastSlider(int value);
    void onSiftContrastSpin(double value);

    // ── Debounce fires → re-extract SIFT on img1 ─────────────────────────────
    void onDebounceTimeout();

    // ── Async SIFT extraction on img1 finishes ────────────────────────────────
    void onImg1SiftDone();

    // ── Match pipeline ────────────────────────────────────────────────────────
    void onExecuteMatch();
    void onMatchDone();

    // ── ROI box controls ──────────────────────────────────────────────────────
    void onUndoRoi();
    void onRedoRoi();
    void onResetRoi();
    void onRoiHistoryChanged(); // enable/disable undo-redo buttons

private:
    // ── Result struct returned from the async match thread ───────────────────
    struct MatchResult
    {
        std::vector<cv::KeyPoint> kp2; // kps in img2 space (roi-offset applied)
        std::vector<cv::DMatch> matches;
        bool valid = false;
    };

    // ── Match lines for paintEvent ────────────────────────────────────────────
    struct MatchLine
    {
        QPointF start;
        QPointF end;
    };

    // ── Helpers ───────────────────────────────────────────────────────────────
    void downscaleIfNeeded(cv::Mat &img);
    void runSiftOnImg1Async();          // start (or queue) bg extraction
    void displayImg1WithKeypoints();    // draw circles on img1 panel
    cv::Rect getRoiInImg2Space() const; // map selected ROI → img2 pixel coords

    // ── Qt / UI ───────────────────────────────────────────────────────────────
    Ui::MainWindow *ui;
    QTimer *debounceTimer;

    // ── Async watchers ────────────────────────────────────────────────────────
    QFutureWatcher<void> watcherSift1;        // img1 feature extraction
    QFutureWatcher<MatchResult> watcherMatch; // img2 extraction + matching

    // ── Result buffers written by background threads via shared_ptr ──────────
    std::shared_ptr<std::vector<cv::KeyPoint>> pendingKp1;
    std::shared_ptr<cv::Mat> pendingDesc1;
    bool sift1RerunPending = false;

    // ── Persistent image/feature state ────────────────────────────────────────
    cv::Mat img1, img2;
    std::vector<cv::KeyPoint> kp1;
    cv::Mat desc1;

    // ── Runtime parameters (kept in sync with sliders) ────────────────────────
    float currentRatioThresh = 0.75f;
    float currentContrastThresh = 0.007f;

    // ── Match lines drawn in paintEvent ───────────────────────────────────────
    std::vector<MatchLine> matchLines;
};