#pragma once
/**
 * @file zoomable_image.hpp
 * @brief ZoomableImageLabel — mouse-wheel zoom + middle-click pan.
 *
 * SRP: Only handles display zoom/pan; no image processing.
 * Usage: setImage(pixmap) then the user can scroll to zoom and
 *        middle-click-drag (or Shift+Left-drag) to pan.
 *
 * Zoom: mouse wheel, centered on cursor position.
 * Pan:  middle-mouse-button drag  OR  Shift + left-drag.
 */

#include <QLabel>
#include <QPixmap>
#include <QWheelEvent>
#include <QMouseEvent>

class ZoomableImageLabel : public QLabel
{
    Q_OBJECT

public:
    explicit ZoomableImageLabel(QWidget* parent = nullptr);

    void setImage(const QPixmap& pixmap);
    void resetZoom();
    void setContainerSize(int width, int height);

    double zoomLevel() const { return zoom_; }
    QSize sizeHint() const override;

protected:
    void wheelEvent(QWheelEvent*    event) override;
    void mousePressEvent(QMouseEvent*   event) override;
    void mouseMoveEvent(QMouseEvent*    event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void resizeEvent(QResizeEvent*      event) override;

private:
    void updateDisplay();
    void clampZoom();

    QPixmap original_;
    double  zoom_      = 1.0;
    bool    panning_   = false;
    QPoint  panStart_;
    int     panX_      = 0;
    int     panY_      = 0;
    int     containerW_ = 400;
    int     containerH_ = 300;

    static constexpr double STEP    = 1.15;
    static constexpr double MIN_Z   = 0.05;
    static constexpr double MAX_Z   = 8.0;
};
