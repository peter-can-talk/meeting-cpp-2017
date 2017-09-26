#include "backend.h"

#include <QDataStream>
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
    emit prediction(socket->readAll().toInt());
  });
}

void BackEnd::predict(QString imageFilename) {
  if (socket->waitForConnected(3000)) {
    socket->write(imageFilename.toStdString().c_str());
  } else {
    qDebug() << "Error connecting to server!";
  }
}
