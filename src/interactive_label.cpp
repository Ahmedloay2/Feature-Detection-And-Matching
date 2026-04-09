#include "interactive_label.h"

InteractiveLabel::InteractiveLabel(QWidget *parent)
    : QLabel(parent)
{
    setMouseTracking(true);
}

// ── Public API ────────────────────────────────────────────────────────────────

std::vector<QRect> InteractiveLabel::getSelectedROIs() const
{
    std::vector<QRect> mapped;
    if (pixmap().isNull() || rois.empty())
        return mapped;

    float rx = (float)pixmap().width() / (float)this->width();
    float ry = (float)pixmap().height() / (float)this->height();

    for (const auto &r : rois)
    {
        QRect m(qRound(r.x() * rx), qRound(r.y() * ry),
                qRound(r.width() * rx), qRound(r.height() * ry));
        mapped.push_back(m.intersected(pixmap().rect()));
    }
    return mapped;
}

void InteractiveLabel::clearROI()
{
    undoStack.clear();
    redoStack.clear();
    rois.clear();
    currentROI = QRect();
    update();
    emit roiHistoryChanged();
}

void InteractiveLabel::undoRoi()
{
    if (undoStack.empty())
        return;
    redoStack.push_back(rois); // current → redo
    rois = undoStack.back();   // restore previous
    undoStack.pop_back();
    update();
    emit roiSelected();
    emit roiHistoryChanged();
}

void InteractiveLabel::redoRoi()
{
    if (redoStack.empty())
        return;
    undoStack.push_back(rois); // current → undo
    rois = redoStack.back();
    redoStack.pop_back();
    update();
    emit roiSelected();
    emit roiHistoryChanged();
}

// ── Mouse events ──────────────────────────────────────────────────────────────

void InteractiveLabel::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && !pixmap().isNull())
    {
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
            undoStack.push_back(rois); // save state before change
            redoStack.clear();         // new action clears redo history
            rois.push_back(currentROI);
            emit roiSelected();
            emit roiHistoryChanged();
        }
        currentROI = QRect();
        update();
    }
    QLabel::mouseReleaseEvent(event);
}

// ── Paint ─────────────────────────────────────────────────────────────────────

void InteractiveLabel::paintEvent(QPaintEvent *event)
{
    QLabel::paintEvent(event);
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);

    for (const auto &r : rois)
    {
        p.setPen(QPen(QColor(0, 220, 100), 2));
        p.setBrush(QBrush(QColor(0, 220, 100, 40)));
        p.drawRect(r);
    }
    if (!currentROI.isEmpty())
    {
        p.setPen(QPen(Qt::red, 2, Qt::DashLine));
        p.setBrush(QBrush(QColor(255, 60, 60, 40)));
        p.drawRect(currentROI);
    }
}