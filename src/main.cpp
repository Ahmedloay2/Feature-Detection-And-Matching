#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    
    // Sleek Dark Theme App-Wide Stylesheet mapping distinct accents
    a.setStyleSheet(R"(
        QMainWindow {
            background-color: #1e1e1e;
        }
        QWidget {
            color: #ffffff;
            font-family: Inter, Roboto, Arial;
            font-size: 14px;
        }
        QPushButton {
            background-color: #0D47A1;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 10px 20px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #1565C0;
        }
        QPushButton:disabled {
            background-color: #37474F;
            color: #78909C;
        }
        QScrollArea {
            border: 1px solid #333333;
            background-color: #2b2b2b;
        }
        QLabel {
            color: #e0e0e0;
        }
    )");

    MainWindow w;
    w.show();
    return a.exec();
}
