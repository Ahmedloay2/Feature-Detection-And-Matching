#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    setupTab1();
    setupTab2();
    setupTab3();
    setupConnections();
    setStatus("Ready — load Image 1 to begin.");
}

MainWindow::~MainWindow()
{
    if (watcherCorners.isRunning()) watcherCorners.cancel();
    if (watcherSift.isRunning()) watcherSift.cancel();
    if (watcherMatch.isRunning()) watcherMatch.cancel();

    watcherCorners.waitForFinished();
    watcherSift.waitForFinished();
    watcherMatch.waitForFinished();

    delete ui;
}
