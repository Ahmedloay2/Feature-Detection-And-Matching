#pragma once

#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QRect>
#include <vector>

class InteractiveLabel : public QLabel
{
    Q_OBJECT

public:
    explicit InteractiveLabel(QWidget *parent = nullptr);

    std::vector<QRect> getSelectedROIs() const;
    void clearROI();

signals:
    void roiSelected(); // Signal when done drawing one

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    bool isDrawing;
    QPoint startPoint;
    QPoint endPoint;
    QRect currentROI;
    std::vector<QRect> rois;
};
