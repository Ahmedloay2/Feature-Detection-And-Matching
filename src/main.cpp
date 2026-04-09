/**
 * @file main.cpp
 * @brief Main Entry Point: Merged SIFT Extractor & Custom Harris Comparison
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
#include <QTabWidget>

// Domain
#include "mainwindow.h"
#include "model/image.hpp"
#include "io/image_handler.hpp"
#include "processors/harris_main.hpp"
#include "processors/harris/harris_response.hpp"
#include "processors/harris/grayscale.hpp"
#include "processors/harris/gradient.hpp"
#include "processors/harris/strcutre_tensor.hpp"
#include "processors/harris/threshold.hpp"

// =============================================================================
//  Harris Comparison Configuration
// =============================================================================
static const std::string IMAGE_PATH = "resources/images/1.jpg";

static constexpr float K_HARRIS = 0.04f;
static constexpr float THRESHOLD = 120.f;
static constexpr int HALF_WINDOW = 3;

// OpenCV built-in params (For Shi-Tomasi)
static constexpr int CV_MAX_CORNERS = 200;
static constexpr double CV_QUALITY = 0.01;
static constexpr double CV_MIN_DIST = 10.0;
// =============================================================================

// Descriptions
static const std::string DESC_HARRIS_CUSTOM =
    "Custom Harris: R = det(M) - k*trace(M)^2\n"
    "R>0=corner  R<0=edge  R~0=flat\n"
    "Sobel gradients + box-filter structure tensor + custom NMS.";

static const std::string DESC_SHI_CUSTOM =
    "Custom Shi-Tomasi (lambda-): R = min eigenvalue of M\n"
    "R = (Sxx+Syy - sqrt((Sxx-Syy)^2 + 4*Sxy^2)) / 2\n"
    "More stable than Harris near edges. Same NMS pipeline.";

static const std::string DESC_HARRIS_REPLACED =
    "Replaced OpenCV Harris: Now effectively runs Custom Harris\n"
    "Computes response (CV_32F raw gradients), then normalized\n"
    "by cv::normalize to [0,255] and thresholded > 120.f.";

static const std::string DESC_SHI_CV =
    "OpenCV Shi-Tomasi (goodFeaturesToTrack):\n"
    "Top N corners ranked by min eigenvalue.\n"
    "Built-in NMS via minDistance. Max=500, quality=0.01.";

// Timing Helpers
using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline double elapsedMs(TimePoint t0, TimePoint t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── Zoomable QGraphicsView ───────────────────────────────────────────────────
class ZoomView : public QGraphicsView
{
public:
    explicit ZoomView(QWidget *parent = nullptr) : QGraphicsView(parent)
    {
        setDragMode(QGraphicsView::ScrollHandDrag);
        setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
        setResizeAnchor(QGraphicsView::AnchorUnderMouse);
        setRenderHint(QPainter::Antialiasing);
        setRenderHint(QPainter::SmoothPixmapTransform);
        setBackgroundBrush(QColor("#0d0d1a"));
        setFrameShape(QFrame::NoFrame);
    }

protected:
    void wheelEvent(QWheelEvent *e) override
    {
        const double factor = e->angleDelta().y() > 0 ? 1.15 : 1.0 / 1.15;
        scale(factor, factor);
    }
};

// ── Utility Functions ────────────────────────────────────────────────────────
cv::Mat drawCorners(const cv::Mat &src, const std::vector<cv::Point> &corners, const cv::Scalar &color)
{
    cv::Mat vis = src.clone();
    for (const auto &pt : corners)
        cv::circle(vis, pt, 4, color, -1, cv::LINE_AA);
    return vis;
}

void overlayStats(cv::Mat &img, const std::string &title, int cornerCount, double timeMs)
{
    cv::Mat overlay = img.clone();
    cv::rectangle(overlay, cv::Point(0, 0), cv::Point(img.cols, 52), cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.6, img, 0.4, 0, img);
    cv::putText(img, title, cv::Point(6, 18), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    std::ostringstream ss;
    ss << "Corners: " << cornerCount << "   Time: " << std::fixed << std::setprecision(2) << timeMs << " ms";
    cv::putText(img, ss.str(), cv::Point(6, 42), cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
}

QPixmap matToPixmap(const cv::Mat &mat)
{
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
    return QPixmap::fromImage(qimg.copy());
}

QWidget *makeStatsPanel(const std::string names[4], const int counts[4], const double times[4], const std::string descs[4], const cv::Scalar colors[4])
{
    QWidget *panel = new QWidget();
    panel->setStyleSheet("background-color: #0d0d1a;");
    QHBoxLayout *hbox = new QHBoxLayout(panel);
    hbox->setContentsMargins(8, 6, 8, 6);
    hbox->setSpacing(6);

    for (int i = 0; i < 4; ++i)
    {
        QWidget *cell = new QWidget();
        cell->setStyleSheet("background-color: #1a1a2e; border-radius: 4px;");

        QFrame *bar = new QFrame();
        bar->setFixedHeight(3);
        QString c = QString("rgb(%1,%2,%3)").arg((int)colors[i][2]).arg((int)colors[i][1]).arg((int)colors[i][0]);
        bar->setStyleSheet(QString("background-color: %1;").arg(c));

        QLabel *nameLabel = new QLabel(QString::fromStdString(names[i]));
        nameLabel->setStyleSheet("color: #ffffff; font-family: Consolas,monospace; font-size: 11px; font-weight: bold; padding: 4px 8px 0px 8px;");

        std::ostringstream ss;
        ss << "Corners: " << counts[i] << "   |   " << std::fixed << std::setprecision(3) << times[i] << " ms";
        QLabel *statsLabel = new QLabel(QString::fromStdString(ss.str()));
        statsLabel->setStyleSheet("color: #00e5ff; font-family: Consolas,monospace; font-size: 10px; padding: 2px 8px 0px 8px;");

        QLabel *descLabel = new QLabel(QString::fromStdString(descs[i]));
        descLabel->setStyleSheet("color: #9e9e9e; font-family: Consolas,monospace; font-size: 9px; padding: 2px 8px 6px 8px;");
        descLabel->setWordWrap(true);

        QVBoxLayout *vbox = new QVBoxLayout(cell);
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

// ── Replaced OpenCV Harris ───────────────────────────────────────────────────
std::vector<cv::Point> runReplacedHarris(Image &imgObj, double &timeMs)
{
    auto t0 = Clock::now();

    // 1. Run the entire pipeline natively using Custom implementations
    toGrayscale(imgObj);
    computeGradient(imgObj);
    applyStructureTensor(imgObj);
    computeHarrisResponse(imgObj, K_HARRIS);

    // 2. Thresholding + Custom Normalization to 0-255 scaling
    applyCornerThreshold(imgObj, "harris_response", 120.f);

    // 3. Extract the corners from the normalized threshold step directly
    std::vector<cv::Point> corners;
    const cv::Mat &responseNorm = imgObj.get("harris_response_threshold");
    for (int i = 0; i < responseNorm.rows; ++i)
    {
        for (int j = 0; j < responseNorm.cols; ++j)
        {
            if (responseNorm.at<float>(i, j) > 0.f)
            {
                corners.push_back(cv::Point(j, i));
            }
        }
    }

    timeMs = elapsedMs(t0, Clock::now());
    return corners;
}

// ── OpenCV Shi-Tomasi ────────────────────────────────────────────────────────
std::vector<cv::Point> runOpenCVShiTomasi(const cv::Mat &gray8U, double &timeMs)
{
    std::vector<cv::Point2f> pts;
    auto t0 = Clock::now();
    cv::goodFeaturesToTrack(gray8U, pts, CV_MAX_CORNERS, CV_QUALITY, CV_MIN_DIST);
    timeMs = elapsedMs(t0, Clock::now());

    std::vector<cv::Point> corners;
    corners.reserve(pts.size());
    for (const auto &p : pts)
        corners.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
    return corners;
}

// ── Harris Comparison Widget Class ───────────────────────────────────────────
class HarrisComparisonWidget : public QWidget
{
public:
    HarrisComparisonWidget(QWidget *parent = nullptr) : QWidget(parent)
    {
        setupPipeline();
    }

private:
    void setupPipeline()
    {
        // Custom IO Handler usage!
        cv::Mat src;
        try
        {
            Image srcImg = loadImage(IMAGE_PATH);
            src = srcImg.mat;
            std::cout << "Loaded: " << IMAGE_PATH << "  [" << src.cols << " x " << src.rows << "]\n";
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: could not load image: " << e.what() << "\n";
            QLabel *errorLabel = new QLabel("Error loading default image: " + QString(e.what()));
            QVBoxLayout *l = new QVBoxLayout(this);
            l->addWidget(errorLabel);
            return;
        }

        // ── Detector 1 & 2: Custom Harris & Shi-Tomasi
        Image img1 = loadImage(IMAGE_PATH); // Load fresh image object for processors

        toGrayscale(img1);
        cv::Mat gray8U = img1.get("grayscale").clone();

        auto h0 = Clock::now();
        auto harrisCorners = applyHarris(img1, K_HARRIS, "harris", THRESHOLD, HALF_WINDOW);
        double t_harris = elapsedMs(h0, Clock::now());

        auto s0 = Clock::now();
        auto shiCorners = applyHarris(img1, 0.f, "shi_tomasi", THRESHOLD, HALF_WINDOW);
        double t_shi = elapsedMs(s0, Clock::now());

        // ── Detector 3: Replaced OpenCV Harris, using custom internals
        double t_cvHarris;
        auto replacedHarrisCorners = runReplacedHarris(img1, t_cvHarris);

        // ── Detector 4: OpenCV Shi-Tomasi
        double t_cvShi;
        auto cvShiCorners = runOpenCVShiTomasi(gray8U, t_cvShi);

        // UI Data Aggregation
        const std::string names[4] = {"Your Harris", "Your Shi-Tomasi", "Replaced Harris", "OpenCV Shi-Tomasi"};
        const int counts[4] = {(int)harrisCorners.size(), (int)shiCorners.size(), (int)replacedHarrisCorners.size(), (int)cvShiCorners.size()};
        const double times[4] = {t_harris, t_shi, t_cvHarris, t_cvShi};
        const std::vector<cv::Point> allCorners[4] = {harrisCorners, shiCorners, replacedHarrisCorners, cvShiCorners};
        const std::string descs[4] = {DESC_HARRIS_CUSTOM, DESC_SHI_CUSTOM, DESC_HARRIS_REPLACED, DESC_SHI_CV};
        const cv::Scalar colors[4] = {
            cv::Scalar(0, 0, 255),  // red
            cv::Scalar(0, 255, 0),  // green
            cv::Scalar(255, 0, 0),  // blue
            cv::Scalar(0, 255, 255) // yellow
        };

        // Annotate images
        cv::Mat vis[4];
        vis[0] = drawCorners(src, harrisCorners, colors[0]);
        vis[1] = drawCorners(src, shiCorners, colors[1]);
        vis[2] = drawCorners(src, replacedHarrisCorners, colors[2]);
        vis[3] = drawCorners(src, cvShiCorners, colors[3]);

        for (int i = 0; i < 4; ++i)
            overlayStats(vis[i], names[i], counts[i], times[i]);

        QScreen *screen = QApplication::primaryScreen();
        QSize screenSize = screen->availableSize();
        int availH = screenSize.height() - 250;
        int availW = screenSize.width() - 40;
        int targetCellW = (availW - 8) / 2;
        int targetCellH = (availH - 8) / 2;

        float scale = std::min((float)targetCellW / src.cols, (float)targetCellH / src.rows);
        cv::Size cellSize((int)(src.cols * scale), (int)(src.rows * scale));
        for (int i = 0; i < 4; ++i)
            cv::resize(vis[i], vis[i], cellSize);

        cv::Mat row1, row2, grid;
        cv::hconcat(vis[0], vis[1], row1);
        cv::hconcat(vis[2], vis[3], row2);
        cv::vconcat(row1, row2, grid);

        QGraphicsScene *scene = new QGraphicsScene(this);
        scene->addPixmap(matToPixmap(grid));
        ZoomView *view = new ZoomView();
        view->setScene(scene);
        view->fitInView(scene->itemsBoundingRect(), Qt::KeepAspectRatio);

        QWidget *statsPanel = makeStatsPanel(names, counts, times, descs, colors);
        statsPanel->setFixedHeight(120);

        QVBoxLayout *outerLayout = new QVBoxLayout(this);
        outerLayout->setContentsMargins(4, 4, 4, 4);
        outerLayout->setSpacing(4);
        outerLayout->addWidget(view, 1);
        outerLayout->addWidget(statsPanel, 0);

        setLayout(outerLayout);
    }
};

// ── Application Entry ────────────────────────────────────────────────────────
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // App-Wide Stylesheet
    a.setStyleSheet(R"(
        QMainWindow { background-color: #1e1e1e; }
        QWidget { color: #ffffff; font-family: Inter, Roboto, Arial; font-size: 14px; }
        QPushButton { background-color: #0D47A1; color: white; border: none; border-radius: 4px; padding: 10px 20px; font-weight: bold; }
        QPushButton:hover { background-color: #1565C0; }
        QPushButton:disabled { background-color: #37474F; color: #78909C; }
        QScrollArea { border: 1px solid #333333; background-color: #2b2b2b; }
        QLabel { color: #e0e0e0; }
        QTabWidget::pane { border: 1px solid #333333; background: #1e1e1e; }
        QTabBar::tab { background: #2b2b2b; color: white; padding: 8px 16px; margin: 2px; }
        QTabBar::tab:selected { background: #0D47A1; font-weight: bold; }
    )");

    // QTabWidget cleanly resolving the two app modalities
    QTabWidget *mainTabs = new QTabWidget();

    MainWindow *siftWindow = new MainWindow();
    HarrisComparisonWidget *harrisWindow = new HarrisComparisonWidget();

    mainTabs->addTab(siftWindow, "SIFT Extractor (Main Workflow)");
    mainTabs->addTab(harrisWindow, "Harris Detector Analysis");

    QScreen *screen = QApplication::primaryScreen();
    QSize screenSize = screen->availableSize();
    mainTabs->resize(screenSize.width() - 80, screenSize.height() - 100);
    mainTabs->setWindowTitle("Feature Detection And Matching");

    mainTabs->show();
    return a.exec();
}
