/**
 * @file main.cpp
 * @brief Main Entry Point: SIFT Extractor Application
 */

#include <QApplication>

#include "mainwindow.h"

// =============================================================================
//  Application Entry
// =============================================================================
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    MainWindow window;
    window.setWindowTitle("Feature Detection And Matching");
    window.show();

    return app.exec();
}

