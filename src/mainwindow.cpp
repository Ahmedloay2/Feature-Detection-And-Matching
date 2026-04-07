/**
 * @file mainwindow.cpp
 * @brief Implements MainWindow initialization, tab setup, and async task management.
 */

#include "mainwindow.h"
#include "ui_mainwindow.h"

/// Construct the main window and initialize all UI components.
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    // Load UI definition from .ui file
    ui->setupUi(this);
    
    // Initialize each tab's widgets and parameters
    setupTab1();      // Corner detection
    setupTab2();      // SIFT extraction
    setupTab3();      // Feature matching
    
    // Connect all signals to slots
    setupConnections();
    
    // Display welcome message
    setStatus("Ready — load Image 1 to begin.");
}

/// Destructor: Clean up resources and cancel any running background tasks.
MainWindow::~MainWindow()
{
    // Cancel running async operations
    if (watcherCorners.isRunning()) watcherCorners.cancel();
    if (watcherSift.isRunning()) watcherSift.cancel();
    if (watcherMatch.isRunning()) watcherMatch.cancel();

    // Block until all background threads have finished
    watcherCorners.waitForFinished();
    watcherSift.waitForFinished();
    watcherMatch.waitForFinished();

    delete ui;
}
