/**
 * @file main.cpp
 * @brief Application entry point: initializes Qt framework and launches the main window.
 */

#include <QApplication>
#include "ui/mainwindow.h"

/// @brief Application entry point.
///
/// Creates a QApplication instance to manage the application lifecycle,
/// constructs and displays the main window, and enters the event loop.
int main(int argc, char *argv[])
{
    // Create the QApplication instance (required for all Qt GUIs)
    QApplication app(argc, argv);

    // Instantiate the main window
    MainWindow window;
    window.setWindowTitle("Feature Detection And Matching");

    // Display the window and enter the event loop
    window.show();
    return app.exec();
}
