#include "backend.h"

#include <QDebug>
#include <QObject>
#include <QString>
#include <QTcpSocket>

const int kPort = 6666;

BackEnd::BackEnd(QObject* parent)
: QObject(parent), socket(new QTcpSocket(this)) {
  socket->connectToHost("localhost", kPort);

  QObject::connect(socket, &QTcpSocket::connected, [] {
    qDebug() << "Connected to localhost:" << kPort;
  });

  QObject::connect(socket, &QTcpSocket::readyRead, [this] {
    emit imageReady(socket->readAll());
  });
}

void BackEnd::generateImage(int digit, double a, double b) {
  if (socket->waitForConnected(3000)) {
    const auto string = QString("%1 %2 %3").arg(digit).arg(a).arg(b);
    socket->write(string.toStdString().c_str());
  } else {
    qDebug() << "Error connecting to server!";
  }
}
