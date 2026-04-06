/**
 * @file main.cpp
 * @brief Corner detection comparison: Custom Harris, Custom Shi-Tomasi,
 *        OpenCV Harris, OpenCV Shi-Tomasi (goodFeaturesToTrack).
 *
 * Set IMAGE_PATH below and run. No command-line arguments needed.
 * Mouse wheel = zoom in/out. Click + drag = pan.
 */

#include <iostream>
#include <chrono>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <algorithm>

 // OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// Qt
#include <QApplication>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QWidget>
#include <QFont>
#include <QString>
#include <QFrame>
#include <QScreen>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QWheelEvent>
#include <QPainter>
#include <QScrollArea>

// Your pipeline
#include "model/image.hpp"
#include "io/image_handler.hpp"
#include "processors/harris_main.hpp"

// =============================================================================
//  CONFIGURE HERE
// =============================================================================
static const std::string IMAGE_PATH = "D:\\sbme\\third year\\Second Term\\Computer Vision\\Tasks\\Feature-Detection-And-Matching\\resources\\images\\1.jpg";

static constexpr float  K_HARRIS = 0.04f;
static constexpr float  THRESHOLD = 120.f;   
static constexpr int    HALF_WINDOW = 3;        

// OpenCV built-in params
static constexpr int    CV_BLOCK_SIZE = 2;
static constexpr int    CV_APERTURE = 3;
static constexpr int    CV_MAX_CORNERS = 200;
static constexpr double CV_QUALITY = 0.01;
static constexpr double CV_MIN_DIST = 10.0;
// =============================================================================

// ── Technique descriptions ────────────────────────────────────────────────────
static const std::string DESC_HARRIS_CUSTOM =
"Custom Harris: R = det(M) - k*trace(M)^2\n"
"R>0=corner  R<0=edge  R~0=flat\n"
"Sobel gradients + box-filter structure tensor + custom NMS.";

static const std::string DESC_SHI_CUSTOM =
"Custom Shi-Tomasi (lambda-): R = min eigenvalue of M\n"
"R = (Sxx+Syy - sqrt((Sxx-Syy)^2 + 4*Sxy^2)) / 2\n"
"More stable than Harris near edges. Same NMS pipeline.";

static const std::string DESC_HARRIS_CV =
"OpenCV Harris (cornerHarris): Same formula, SIMD-optimised.\n"
"Response normalised to [0,255] then thresholded at 180.\n"
"blockSize=2, apertureSize=3, k=0.04.";

static const std::string DESC_SHI_CV =
"OpenCV Shi-Tomasi (goodFeaturesToTrack):\n"
"Top N corners ranked by min eigenvalue.\n"
"Built-in NMS via minDistance. Max=500, quality=0.01.";

// ── Timing ────────────────────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline double elapsedMs(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Zoomable QGraphicsView ────────────────────────────────────────────────────
class ZoomView : public QGraphicsView
{
public:
    explicit ZoomView(QWidget* parent = nullptr) : QGraphicsView(parent) {
        setDragMode(QGraphicsView::ScrollHandDrag);
        setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        setResizeAnchor(QGraphicsView::AnchorUnderMouse);
        setRenderHint(QPainter::Antialiasing);
        setRenderHint(QPainter::SmoothPixmapTransform);
        setBackgroundBrush(QColor("#0d0d1a"));
        setFrameShape(QFrame::NoFrame);
    }

protected:
    void wheelEvent(QWheelEvent* e) override {
        const double factor = e->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
        scale(factor, factor);
    }
};

// ── Draw corners ──────────────────────────────────────────────────────────────
cv::Mat drawCorners(const cv::Mat& src,
    const std::vector<cv::Point>& corners,
    const cv::Scalar& color)
{
    cv::Mat vis = src.clone();
    for (const auto& pt : corners)
        cv::circle(vis, pt, 4, color, -1, cv::LINE_AA);
    return vis;
}

// ── Overlay title + stats on image ────────────────────────────────────────────
void overlayStats(cv::Mat& img,
    const std::string& title,
    int cornerCount,
    double timeMs)
{
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(img.cols, 52),
        cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.6, img, 0.4, 0, img);

    cv::putText(img, title,
        cv::Point(6, 18), cv::FONT_HERSHEY_SIMPLEX,
        0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    std::ostringstream ss;
    ss << "Corners: " << cornerCount
        << "   Time: " << std::fixed << std::setprecision(2) << timeMs << " ms";
    cv::putText(img, ss.str(),
        cv::Point(6, 42), cv::FONT_HERSHEY_SIMPLEX,
        0.48, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
}

// ── cv::Mat (BGR) → QPixmap ───────────────────────────────────────────────────
QPixmap matToPixmap(const cv::Mat& mat) {
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows,
        static_cast<int>(rgb.step),
        QImage::Format_RGB888);
    return QPixmap::fromImage(qimg.copy());
}

// ── Stats panel widget shown below the zoom view ─────────────────────────────
QWidget* makeStatsPanel(const std::string names[4],
    const int counts[4],
    const double times[4],
    const std::string descs[4],
    const cv::Scalar colors[4])   // BGR
{
    QWidget* panel = new QWidget();
    panel->setStyleSheet("background-color: #0d0d1a;");

    QHBoxLayout* hbox = new QHBoxLayout(panel);
    hbox->setContentsMargins(8, 6, 8, 6);
    hbox->setSpacing(6);

    for (int i = 0; i < 4; ++i) {
        QWidget* cell = new QWidget();
        cell->setStyleSheet("background-color: #1a1a2e; border-radius: 4px;");

        // Colored accent bar
        QFrame* bar = new QFrame();
        bar->setFixedHeight(3);
        QString c = QString("rgb(%1,%2,%3)")
            .arg((int)colors[i][2])
            .arg((int)colors[i][1])
            .arg((int)colors[i][0]);
        bar->setStyleSheet(QString("background-color: %1;").arg(c));

        // Detector name
        QLabel* nameLabel = new QLabel(QString::fromStdString(names[i]));
        nameLabel->setStyleSheet(
            "color: #ffffff; font-family: Consolas,monospace; font-size: 11px;"
            "font-weight: bold; padding: 4px 8px 0px 8px;");

        // Stats
        std::ostringstream ss;
        ss << "Corners: " << counts[i]
            << "   |   " << std::fixed << std::setprecision(3) << times[i] << " ms";
        QLabel* statsLabel = new QLabel(QString::fromStdString(ss.str()));
        statsLabel->setStyleSheet(
            "color: #00e5ff; font-family: Consolas,monospace; font-size: 10px;"
            "padding: 2px 8px 0px 8px;");

        // Description
        QLabel* descLabel = new QLabel(QString::fromStdString(descs[i]));
        descLabel->setStyleSheet(
            "color: #9e9e9e; font-family: Consolas,monospace; font-size: 9px;"
            "padding: 2px 8px 6px 8px;");
        descLabel->setWordWrap(true);

        QVBoxLayout* vbox = new QVBoxLayout(cell);
        vbox->setContentsMargins(0, 0, 0, 0);
        vbox->setSpacing(0);
        vbox->addWidget(bar);
        vbox->addWidget(nameLabel);
        vbox->addWidget(statsLabel);
        vbox->addWidget(descLabel);

        hbox->addWidget(cell);
    }

    return panel;
}

// ── OpenCV Harris ─────────────────────────────────────────────────────────────
std::vector<cv::Point> runOpenCVHarris(const cv::Mat& grayF32, double& timeMs) {
    cv::Mat response, responseNorm;
    auto t0 = Clock::now();

    cv::cornerHarris(grayF32, response, CV_BLOCK_SIZE, CV_APERTURE, K_HARRIS);
    cv::normalize(response, responseNorm, 0, 255, cv::NORM_MINMAX, CV_32F);

    std::vector<cv::Point> corners;
    for (int i = 0; i < responseNorm.rows; ++i)
        for (int j = 0; j < responseNorm.cols; ++j)
            if (responseNorm.at<float>(i, j) > 120.f)
                corners.push_back(cv::Point(j, i));

    timeMs = elapsedMs(t0, Clock::now());
    return corners;
}

// ── OpenCV Shi-Tomasi ─────────────────────────────────────────────────────────
std::vector<cv::Point> runOpenCVShiTomasi(const cv::Mat& gray8U, double& timeMs) {
    std::vector<cv::Point2f> pts;
    auto t0 = Clock::now();

    cv::goodFeaturesToTrack(gray8U, pts, CV_MAX_CORNERS, CV_QUALITY, CV_MIN_DIST);

    timeMs = elapsedMs(t0, Clock::now());

    std::vector<cv::Point> corners;
    corners.reserve(pts.size());
    for (const auto& p : pts)
        corners.push_back(cv::Point(static_cast<int>(p.x),
            static_cast<int>(p.y)));
    return corners;
}

// ── Corner agreement analysis ─────────────────────────────────────────────────
bool cornersAgree(const cv::Point& a, const cv::Point& b, int radius) {
    int dx = a.x - b.x, dy = a.y - b.y;
    return (dx * dx + dy * dy) <= (radius * radius);
}

void printAgreement(const std::string names[4],
    const std::vector<cv::Point> corners[4],
    int radius = 5)
{
    const int W = 72;
    std::cout << std::string(W, '=') << "\n";
    std::cout << "      Detector Agreement Analysis  (radius = "
        << radius << " px)\n";
    std::cout << std::string(W, '=') << "\n";
    std::cout
        << "  HOW TO READ THIS TABLE:\n"
        << "  Agree-A = % of A's corners that B also found nearby\n"
        << "  Agree-B = % of B's corners that A also found nearby\n"
        << "  Match-A = how many of A's corners were matched by B\n"
        << "  Match-B = how many of B's corners were matched by A\n";
    std::cout << std::string(W, '-') << "\n";
    std::cout << std::left
        << std::setw(28) << "Pair"
        << std::setw(10) << "Agree-A"
        << std::setw(10) << "Agree-B"
        << std::setw(10) << "Match-A"
        << "Match-B\n";
    std::cout << std::string(W, '-') << "\n";

    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            int matchA = 0, matchB = 0;
            for (const auto& pa : corners[a])
                for (const auto& pb : corners[b])
                    if (cornersAgree(pa, pb, radius)) { ++matchA; break; }
            for (const auto& pb : corners[b])
                for (const auto& pa : corners[a])
                    if (cornersAgree(pa, pb, radius)) { ++matchB; break; }

            int sizeA = (int)corners[a].size();
            int sizeB = (int)corners[b].size();
            std::string pairName =
                names[a].substr(0, 9) + " / " + names[b].substr(0, 12);

            std::cout << std::left << std::setw(28) << pairName
                << std::setw(10) << (sizeA ? std::to_string(matchA * 100 / sizeA) + "%" : "N/A")
                << std::setw(10) << (sizeB ? std::to_string(matchB * 100 / sizeB) + "%" : "N/A")
                << std::setw(10) << matchA
                << matchB << "\n";
        }
    }
    std::cout << std::string(W, '-') << "\n\n";
}

// ── Stdout report ─────────────────────────────────────────────────────────────
void printReport(const std::string names[4],
    const int counts[4],
    const double times[4])
{
    const int W = 62;
    std::cout << "\n" << std::string(W, '=') << "\n";
    std::cout << "      Corner Detection Comparison Report\n";
    std::cout << std::string(W, '=') << "\n";
    std::cout << std::left
        << std::setw(22) << "Detector"
        << std::setw(10) << "Corners"
        << std::setw(14) << "Time (ms)"
        << "ms/corner\n";
    std::cout << std::string(W, '-') << "\n";
    for (int i = 0; i < 4; ++i) {
        double mpc = counts[i] > 0 ? times[i] / counts[i] : 0.0;
        std::cout << std::left
            << std::setw(22) << names[i]
            << std::setw(10) << counts[i]
            << std::setw(14) << std::fixed << std::setprecision(3) << times[i]
            << std::fixed << std::setprecision(4) << mpc << "\n";
    }
    std::cout << std::string(W, '-') << "\n";

    int fastestIdx = 0, mostIdx = 0;
    for (int i = 1; i < 4; ++i) {
        if (times[i] < times[fastestIdx])  fastestIdx = i;
        if (counts[i] > counts[mostIdx])    mostIdx = i;
    }
    std::cout << "  Fastest:       " << names[fastestIdx]
        << " (" << std::fixed << std::setprecision(3)
        << times[fastestIdx] << " ms)\n";
    std::cout << "  Most corners:  " << names[mostIdx]
        << " (" << counts[mostIdx] << ")\n";
    std::cout << std::string(W, '=') << "\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    // ── Load image ────────────────────────────────────────────────────────────
    cv::Mat src = cv::imread(IMAGE_PATH, cv::IMREAD_COLOR);
    if (src.empty()) {
        std::cerr << "Error: could not load image: " << IMAGE_PATH << "\n";
        return 1;
    }
    std::cout << "Loaded: " << IMAGE_PATH
        << "  [" << src.cols << " x " << src.rows << "]\n";

    // Shared grayscale for OpenCV built-ins
    cv::Mat gray8U, grayF32;
    cv::cvtColor(src, gray8U, cv::COLOR_BGR2GRAY);
    gray8U.convertTo(grayF32, CV_32F);
    cv::setNumThreads(cv::getNumberOfCPUs());
    // ── Detector 1: Your Harris ───────────────────────────────────────────────
    Image img1 = loadImage(IMAGE_PATH);
    std::string modeHarris = "harris";
    auto h0 = Clock::now();
    auto harrisCorners = applyHarris(img1, K_HARRIS, modeHarris,
        THRESHOLD, HALF_WINDOW);
    double t_harris = elapsedMs(h0, Clock::now());

    std::string modeShi = "shi_tomasi";
    auto s0 = Clock::now();
    auto shiCorners = applyHarris(img1, 0.f, modeShi,
        THRESHOLD, HALF_WINDOW);
    double t_shi = elapsedMs(s0, Clock::now());

    // ── Detector 3: OpenCV Harris ─────────────────────────────────────────────
    double t_cvHarris;
    auto cvHarrisCorners = runOpenCVHarris(grayF32, t_cvHarris);

    // ── Detector 4: OpenCV Shi-Tomasi ─────────────────────────────────────────
    double t_cvShi;
    auto cvShiCorners = runOpenCVShiTomasi(gray8U, t_cvShi);

    // ── Report data arrays ────────────────────────────────────────────────────
    const std::string names[4] = {
        "Your Harris", "Your Shi-Tomasi",
        "OpenCV Harris", "OpenCV Shi-Tomasi"
    };
    const int counts[4] = {
        (int)harrisCorners.size(), (int)shiCorners.size(),
        (int)cvHarrisCorners.size(), (int)cvShiCorners.size()
    };
    const double times[4] = { t_harris, t_shi, t_cvHarris, t_cvShi };

    const std::vector<cv::Point> allCorners[4] = {
        harrisCorners, shiCorners, cvHarrisCorners, cvShiCorners
    };

    const std::string descs[4] = {
        DESC_HARRIS_CUSTOM, DESC_SHI_CUSTOM,
        DESC_HARRIS_CV, DESC_SHI_CV
    };

    const cv::Scalar colors[4] = {
        cv::Scalar(0,   0, 255),   // red   — Your Harris
        cv::Scalar(0, 255,   0),   // green — Your Shi-Tomasi
        cv::Scalar(255, 0,   0),   // blue  — OpenCV Harris
        cv::Scalar(0, 255, 255)    // yellow— OpenCV Shi-Tomasi
    };

    printReport(names, counts, times);
    printAgreement(names, allCorners);

    // ── Annotate images ───────────────────────────────────────────────────────
    cv::Mat vis[4];
    vis[0] = drawCorners(src, harrisCorners, colors[0]);
    vis[1] = drawCorners(src, shiCorners, colors[1]);
    vis[2] = drawCorners(src, cvHarrisCorners, colors[2]);
    vis[3] = drawCorners(src, cvShiCorners, colors[3]);

    for (int i = 0; i < 4; ++i)
        overlayStats(vis[i], names[i], counts[i], times[i]);

    // ── Scale images to fit screen ────────────────────────────────────────────
    QScreen* screen = QApplication::primaryScreen();
    QSize screenSize = screen->availableSize();

    // Stats panel is ~120px tall; leave room for it and window chrome
    int availH = screenSize.height() - 180;
    int availW = screenSize.width() - 40;

    int targetCellW = (availW - 8) / 2;
    int targetCellH = (availH - 8) / 2;

    float scale = std::min((float)targetCellW / src.cols,
        (float)targetCellH / src.rows);
    cv::Size cellSize((int)(src.cols * scale), (int)(src.rows * scale));

    for (int i = 0; i < 4; ++i)
        cv::resize(vis[i], vis[i], cellSize);

    // ── Build 2x2 grid image ──────────────────────────────────────────────────
    cv::Mat row1, row2, grid;
    cv::hconcat(vis[0], vis[1], row1);
    cv::hconcat(vis[2], vis[3], row2);
    cv::vconcat(row1, row2, grid);

    // ── ZoomView with the grid ────────────────────────────────────────────────
    QGraphicsScene* scene = new QGraphicsScene();
    scene->addPixmap(matToPixmap(grid));

    ZoomView* view = new ZoomView();
    view->setScene(scene);
    view->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);

    // ── Stats panel below ─────────────────────────────────────────────────────
    QWidget* statsPanel = makeStatsPanel(names, counts, times, descs, colors);
    statsPanel->setFixedHeight(120);

    // ── Main window ───────────────────────────────────────────────────────────
    QWidget* window = new QWidget();
    window->setWindowTitle("Corner Detection Comparison  |  scroll=zoom  drag=pan");
    window->setStyleSheet("background-color: #0d0d1a;");

    QVBoxLayout* outerLayout = new QVBoxLayout(window);
    outerLayout->setContentsMargins(4, 4, 4, 4);
    outerLayout->setSpacing(4);
    outerLayout->addWidget(view, 1);          // zoom view takes all remaining space
    outerLayout->addWidget(statsPanel, 0);    // stats panel fixed at bottom

    window->setLayout(outerLayout);
    window->resize(screenSize.width() - 40, screenSize.height() - 60);
    window->show();

    return app.exec();
}