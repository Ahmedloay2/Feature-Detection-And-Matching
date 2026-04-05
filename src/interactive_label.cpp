#include "interactive_label.h"

InteractiveLabel::InteractiveLabel(QWidget *parent)
    : QLabel(parent), isDrawing(false)
{
    setMouseTracking(true);
}

std::vector<QRect> InteractiveLabel::getSelectedROIs() const
{
    std::vector<QRect> mappedROIs;
    if (pixmap().isNull() || rois.empty())
        return mappedROIs;

    // Qt scale factor mapping logic translating from squashed display back to real Pixmap dims:
    float ratioX = (float)pixmap().width() / this->width();
    float ratioY = (float)pixmap().height() / this->height();

    for (const auto &r : rois)
    {
        int mappedX = r.x() * ratioX;
        int mappedY = r.y() * ratioY;
        int mappedW = r.width() * ratioX;
        int mappedH = r.height() * ratioY;

        QRect mappedROI(mappedX, mappedY, mappedW, mappedH);

        // Clamp to pixmap bounds safely
        mappedROIs.push_back(mappedROI.intersected(pixmap().rect()));
    }
    return mappedROIs;
}

void InteractiveLabel::clearROI()
{
    currentROI = QRect();
    rois.clear();
    update();
}

void InteractiveLabel::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && !this->pixmap().isNull())
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
            rois.push_back(currentROI);
            emit roiSelected();
        }
        currentROI = QRect();
        update();
    }
    QLabel::mouseReleaseEvent(event);
}

void InteractiveLabel::paintEvent(QPaintEvent *event)
{
    QLabel::paintEvent(event);

    QPainter painter(this);
    for (const auto &r : rois)
    {
        painter.setPen(QPen(Qt::green, 2, Qt::SolidLine));
        painter.setBrush(QBrush(QColor(0, 255, 0, 50)));
        painter.drawRect(r);
    }

    if (!currentROI.isEmpty())
    {
        painter.setPen(QPen(Qt::red, 2, Qt::SolidLine));
        painter.setBrush(QBrush(QColor(255, 0, 0, 50)));
        painter.drawRect(currentROI);
    }
}
