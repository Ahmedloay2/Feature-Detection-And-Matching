/**
 * @file interactive_label.h
 * @brief Declares an interactive label widget for drawing and managing ROI rectangles.
 */

#pragma once
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QRect>
#include <vector>

/// @brief Interactive label widget for drawing rectangular regions of interest.
///
/// This widget extends QLabel to provide mouse-based ROI rectangle drawing.
/// Users click and drag to define rectangular regions; completed ROIs are stored and
/// rendered with visual feedback. ROIs are automatically scaled from widget coordinates
/// to underlying pixmap pixel coordinates.
class InteractiveLabel : public QLabel
{
    Q_OBJECT

public:
    /// @brief Construct an interactive label widget.
    /// @param parent Parent widget (typically the matching image panel)
    explicit InteractiveLabel(QWidget *parent = nullptr);

    /// @brief Get all confirmed ROI rectangles, scaled to pixmap pixel coordinates.
    /// @return Vector of QRect in pixmap space; empty if no ROIs or pixmap not set
    std::vector<QRect> getSelectedROIs() const;

    /// @brief Clear all ROI rectangles and reset to empty state.
    void clearROI();

    /// @brief Remove the most recently added ROI (undo operation).
    void removeLastROI();

    /// @brief When set to true, the very next mouse-press will clear all existing
    ///        ROIs before starting the new one.  Automatically resets to false after
    ///        the clear fires, so subsequent draws accumulate normally again.
    void setResetOnNextDraw(bool reset) { m_resetOnNextDraw = reset; }

signals:
    void roiSelected(); ///< Emitted when a new ROI is confirmed.

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    bool isDrawing = false;
    bool m_resetOnNextDraw = false; ///< If true, clear ROIs on next mouse-press
    QPoint startPoint;
    QPoint endPoint;
    QRect currentROI;
    std::vector<QRect> rois; ///< Confirmed ROIs in widget coordinates.
};