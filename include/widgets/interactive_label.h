#pragma once
/**
 * @file interactive_label.h
 * @brief QLabel subclass for drawing, managing, and removing ROI rectangles.
 *
 * SRP: Only handles ROI drawing + bookkeeping; no image-processing logic.
 * API:
 *   getSelectedROIs()  — returns all mapped ROI rects in pixmap coordinates
 *   clearROI()         — remove all ROIs
 *   removeLastROI()    — remove the most recently added ROI
 */

#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QRect>
#include <vector>

class InteractiveLabel : public QLabel
{
    Q_OBJECT

public:
    explicit InteractiveLabel(QWidget* parent = nullptr);

    /// All confirmed ROIs, remapped to pixmap pixel space.
    std::vector<QRect> getSelectedROIs() const;

    /// Remove every ROI.
    void clearROI();

    /// Remove the most-recently-added ROI (undo last).
    void removeLastROI();

signals:
    void roiSelected();   ///< Emitted when a new ROI is confirmed.

protected:
    void mousePressEvent(QMouseEvent*   event) override;
    void mouseMoveEvent(QMouseEvent*    event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void paintEvent(QPaintEvent*        event) override;

private:
    bool               isDrawing   = false;
    QPoint             startPoint;
    QPoint             endPoint;
    QRect              currentROI;
    std::vector<QRect> rois;   ///< Confirmed ROIs in widget coordinates.
};
