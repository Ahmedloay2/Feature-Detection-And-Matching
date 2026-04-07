#include "widgets/interactive_label.h"

InteractiveLabel::InteractiveLabel(QWidget* parent)
    : QLabel(parent)
{
    setMouseTracking(true);
}

// ─── Public API ──────────────────────────────────────────────────────────────

std::vector<QRect> InteractiveLabel::getSelectedROIs() const
{
    std::vector<QRect> mapped;
    if (pixmap().isNull() || rois.empty()) return mapped;

    float rx = static_cast<float>(pixmap().width())  / this->width();
    float ry = static_cast<float>(pixmap().height()) / this->height();

    for (const auto& r : rois) {
        QRect m(static_cast<int>(r.x()      * rx),
                static_cast<int>(r.y()      * ry),
                static_cast<int>(r.width()  * rx),
                static_cast<int>(r.height() * ry));
        mapped.push_back(m.intersected(pixmap().rect()));
    }
    return mapped;
}

void InteractiveLabel::clearROI()
{
    currentROI = {};
    rois.clear();
    update();
}

void InteractiveLabel::removeLastROI()
{
    if (!rois.empty()) {
        rois.pop_back();
        update();
    }
}

// ─── Mouse events ─────────────────────────────────────────────────────────────

void InteractiveLabel::mousePressEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && !pixmap().isNull()) {
        isDrawing  = true;
        startPoint = event->pos();
        endPoint   = startPoint;
        currentROI = QRect(startPoint, endPoint);
        update();
    }
    QLabel::mousePressEvent(event);
}

void InteractiveLabel::mouseMoveEvent(QMouseEvent* event)
{
    if (isDrawing && (event->buttons() & Qt::LeftButton)) {
        endPoint   = event->pos();
        currentROI = QRect(startPoint, endPoint).normalized();
        update();
    }
    QLabel::mouseMoveEvent(event);
}

void InteractiveLabel::mouseReleaseEvent(QMouseEvent* event)
{
    if (event->button() == Qt::LeftButton && isDrawing) {
        isDrawing  = false;
        endPoint   = event->pos();
        currentROI = QRect(startPoint, endPoint).normalized();
        if (currentROI.width() > 5 && currentROI.height() > 5) {
            rois.push_back(currentROI);
            emit roiSelected();
        }
        currentROI = {};
        update();
    }
    QLabel::mouseReleaseEvent(event);
}

// ─── Paint ───────────────────────────────────────────────────────────────────

void InteractiveLabel::paintEvent(QPaintEvent* event)
{
    QLabel::paintEvent(event);
    QPainter p(this);

    // Confirmed ROIs — green
    for (const auto& r : rois) {
        p.setPen(QPen(Qt::green, 2));
        p.setBrush(QBrush(QColor(0, 255, 0, 40)));
        p.drawRect(r);
    }

    // In-progress ROI — red dashed
    if (!currentROI.isEmpty()) {
        p.setPen(QPen(Qt::red, 2, Qt::DashLine));
        p.setBrush(QBrush(QColor(255, 0, 0, 30)));
        p.drawRect(currentROI);
    }
}
