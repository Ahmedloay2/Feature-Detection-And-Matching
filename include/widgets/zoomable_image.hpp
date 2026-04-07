/**
 * @file zoomable_image.hpp
 * @brief Declares a custom QLabel with mouse-wheel zoom and middle-click pan support.
 */

#pragma once
#include <QLabel>
#include <QPixmap>
#include <QWheelEvent>
#include <QMouseEvent>

/// @brief Interactive image display widget with zoom and pan capabilities.
///
/// Extends QLabel to provide smooth image viewing with mouse wheel zoom (1-8x),
/// Shift+drag and middle-click pan. Automatic scaling maintains aspect ratio.
class ZoomableImageLabel : public QLabel
{
    Q_OBJECT

public:
    /// @brief Construct a zoomable image label.
    explicit ZoomableImageLabel(QWidget* parent = nullptr);

    /// @brief Set the image to display (resets zoom to 1x).
    void setImage(const QPixmap& pixmap);
    
    /// @brief Reset zoom to 1x and clear pan offset.
    void resetZoom();
    
    /// @brief Set the container size for layout calculations.
    void setContainerSize(int width, int height);

    /// @brief Get current zoom level (1.0 = fit, 8.0 = max).
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
    static constexpr double MIN_Z   = 1.0;
    static constexpr double MAX_Z   = 8.0;
};
