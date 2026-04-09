/**
 * @file interactive_label.cpp
 * @brief Implements ROI rectangle drawing, live feedback, and coordinate transformations.
 */

#include "ui/interactive_label.h"

/// Constructor: Create an interactive label and enable mouse tracking.
InteractiveLabel::InteractiveLabel(QWidget *parent)
    : QLabel(parent)
{
    setMouseTracking(true);
}

/// Get all confirmed ROI rectangles, rescaled to pixmap pixel space.
/// Computes aspect ratio from widget and pixmap dimensions, then scales stored ROIs.
std::vector<QRect> InteractiveLabel::getSelectedROIs() const
{
    std::vector<QRect> mapped;
    if (pixmap().isNull() || rois.empty())
        return mapped;

    float rx = static_cast<float>(pixmap().width()) / this->width();
    float ry = static_cast<float>(pixmap().height()) / this->height();

    for (const auto &r : rois)
    {
        QRect m(static_cast<int>(r.x() * rx),
                static_cast<int>(r.y() * ry),
                static_cast<int>(r.width() * rx),
                static_cast<int>(r.height() * ry));
        mapped.push_back(m.intersected(pixmap().rect()));
    }
    return mapped;
}

/// Remove all ROI rectangles and reset to empty state.
void InteractiveLabel::clearROI()
{
    currentROI = {};
    rois.clear();
    update();
}

/// Remove the most recently added ROI (undo operation).
void InteractiveLabel::removeLastROI()
{
    if (!rois.empty())
    {
        rois.pop_back();
        update();
    }
}

void InteractiveLabel::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && !pixmap().isNull())
    {
        // If a match was just run, the first new draw should wipe the old
        // ROIs so the user starts fresh.  Subsequent draws accumulate normally.
        if (m_resetOnNextDraw)
        {
            rois.clear();
            m_resetOnNextDraw = false;
        }
        isDrawing = true;
        startPoint = event->pos();
        endPoint = startPoint;
        currentROI = QRect(startPoint, endPoint);
        update();
    }
    QLabel::mousePressEvent(event);
}

void InteractiveLabel::mouseMoveEvent(QMouseEvent *event)
{
    if (isDrawing && (event->buttons() & Qt::LeftButton))
    {
        endPoint = event->pos();
        currentROI = QRect(startPoint, endPoint).normalized();
        update();
    }
    QLabel::mouseMoveEvent(event);
}

void InteractiveLabel::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && isDrawing)
    {
        isDrawing = false;
        endPoint = event->pos();
        currentROI = QRect(startPoint, endPoint).normalized();
        if (currentROI.width() > 5 && currentROI.height() > 5)
        {
            rois.push_back(currentROI);
            emit roiSelected();
        }
        currentROI = {};
        update();
    }
    QLabel::mouseReleaseEvent(event);
}

void InteractiveLabel::paintEvent(QPaintEvent *event)
{
    QLabel::paintEvent(event);
    QPainter p(this);

    // Confirmed ROIs — green
    for (const auto &r : rois)
    {
        p.setPen(QPen(Qt::green, 2));
        p.setBrush(QBrush(QColor(0, 255, 0, 40)));
        p.drawRect(r);
    }

    // In-progress ROI — red dashed
    if (!currentROI.isEmpty())
    {
        p.setPen(QPen(Qt::red, 2, Qt::DashLine));
        p.setBrush(QBrush(QColor(255, 0, 0, 30)));
        p.drawRect(currentROI);
    }
}