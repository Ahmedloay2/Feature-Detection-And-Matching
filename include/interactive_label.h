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
    void clearROI(); // clears all — also resets history
    void undoRoi();
    void redoRoi();
    bool canUndo() const { return !undoStack.empty(); }
    bool canRedo() const { return !redoStack.empty(); }

signals:
    void roiSelected();       // fired when a new box is completed
    void roiHistoryChanged(); // fired after any undo/redo/clear/add

protected:
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;
    void paintEvent(QPaintEvent *event) override;

private:
    bool isDrawing = false;
    QPoint startPoint, endPoint;
    QRect currentROI;

    std::vector<QRect> rois;
    std::vector<std::vector<QRect>> undoStack;
    std::vector<std::vector<QRect>> redoStack;
};