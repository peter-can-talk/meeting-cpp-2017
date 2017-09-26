#include <QApplication>
#include <QQmlApplicationEngine>

#include "backend.h"

int main(int argc, char** argv) {
  QApplication app(argc, argv);
  QQmlApplicationEngine engine;

  qmlRegisterType<BackEnd>("demo.backend", 1, 0, "BackEnd");

  engine.load(QUrl("qrc:///main.qml"));

  return app.exec();
}
