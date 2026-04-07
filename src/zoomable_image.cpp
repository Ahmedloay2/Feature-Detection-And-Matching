/**
 * @file zoomable_image.cpp
 * @brief Implements smooth zoom scaling, pan offset handling, and display updates.
 */

#include "widgets/zoomable_image.hpp"
#include <algorithm>
#include <QApplication>
#include <QtMath>
#include <QResizeEvent>

/// Constructor: Create a zoomable image label with proper alignment and input handling.
ZoomableImageLabel::ZoomableImageLabel(QWidget* parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setScaledContents(false);
    setMouseTracking(true);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

/// Set the image to display and reset zoom to 1x, pan offset to origin.
void ZoomableImageLabel::setImage(const QPixmap& pix)
{
    original_ = pix;
    zoom_     = 1.0;
    panX_     = 0;
    panY_     = 0;
    panning_  = false;
    updateDisplay();
}

/// Reset zoom level to 1.0 (fit-to-window) and clear pan offset.
void ZoomableImageLabel::resetZoom()
{
    zoom_ = 1.0;
    panX_ = panY_ = 0;
    updateDisplay();
}

/// Set the container (parent widget) size for layout calculations.
/// Triggers display update if an image is loaded.
void ZoomableImageLabel::setContainerSize(int width, int height)
{
    containerW_ = width;
    containerH_ = height;
    // Trigger refresh with new container size
    if (!original_.isNull()) {
        updateDisplay();
    }
}

QSize ZoomableImageLabel::sizeHint() const
{
    return QSize(containerW_, containerH_);
}

void ZoomableImageLabel::resizeEvent(QResizeEvent* event)
{
    QLabel::resizeEvent(event);
    if (!original_.isNull()) {
        updateDisplay();
    }
}

void ZoomableImageLabel::wheelEvent(QWheelEvent* event)
{
    if (original_.isNull()) { event->ignore(); return; }

    double factor = (event->angleDelta().y() > 0) ? STEP : (1.0 / STEP);
    zoom_ *= factor;
    clampZoom();
    updateDisplay();
    event->accept();   // consume — prevents QScrollArea from scrolling
}

void ZoomableImageLabel::mousePressEvent(QMouseEvent* event)
{
    bool wantPan = (event->button() == Qt::MiddleButton) ||
                   (event->button() == Qt::LeftButton &&
                    QApplication::keyboardModifiers() & Qt::ShiftModifier);
    if (wantPan) {
        panning_  = true;
        panStart_ = event->pos();
        setCursor(Qt::ClosedHandCursor);
        event->accept();
    } else {
        QLabel::mousePressEvent(event);
    }
}

void ZoomableImageLabel::mouseMoveEvent(QMouseEvent* event)
{
    if (panning_) {
        QPoint d  = event->pos() - panStart_;
        panX_    += d.x();
        panY_    += d.y();
        panStart_ = event->pos();
        updateDisplay();
        event->accept();
    } else {
        QLabel::mouseMoveEvent(event);
    }
}

void ZoomableImageLabel::mouseReleaseEvent(QMouseEvent* event)
{
    if (panning_) {
        panning_ = false;
        setCursor(Qt::ArrowCursor);
        event->accept();
    } else {
        QLabel::mouseReleaseEvent(event);
    }
}

void ZoomableImageLabel::updateDisplay()
{
    if (original_.isNull()) return;

    int label_w = std::max(1, width());
    int label_h = std::max(1, height());

    const double fitScaleW = static_cast<double>(label_w) / std::max(1, original_.width());
    const double fitScaleH = static_cast<double>(label_h) / std::max(1, original_.height());
    const double fitScale  = std::min(fitScaleW, fitScaleH);
    const double renderScale = fitScale * zoom_;

    int scaled_w = std::max(1, static_cast<int>(std::round(original_.width() * renderScale)));
    int scaled_h = std::max(1, static_cast<int>(std::round(original_.height() * renderScale)));
    QPixmap scaled = original_.scaled(scaled_w, scaled_h, Qt::IgnoreAspectRatio,
                                      Qt::SmoothTransformation);

    if (scaled_w <= label_w && scaled_h <= label_h) {
        panX_ = panY_ = 0;
        QLabel::setPixmap(scaled);
        return;
    }

    int max_pan_x = scaled_w - label_w;
    int max_pan_y = scaled_h - label_h;
    
    if (max_pan_x < 0) max_pan_x = 0;
    if (max_pan_y < 0) max_pan_y = 0;
    
    if (panX_ < 0)         panX_ = 0;
    if (panX_ > max_pan_x) panX_ = max_pan_x;
    if (panY_ < 0)         panY_ = 0;
    if (panY_ > max_pan_y) panY_ = max_pan_y;

    QRect crop_rect(panX_, panY_, label_w, label_h);
    QPixmap visible = scaled.copy(crop_rect);
    
    QLabel::setPixmap(visible);
}

void ZoomableImageLabel::clampZoom()
{
    if (zoom_ < MIN_Z) zoom_ = MIN_Z;
    if (zoom_ > MAX_Z) zoom_ = MAX_Z;
}
