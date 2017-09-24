#include "backend.h"

#include <QDebug>
#include <QString>

BackEnd::BackEnd(QObject* parent) : QObject(parent) {}

int BackEnd::predict(QString imageFilename) {
  qDebug() << "Sending " << imageFilename;
  return 5;
}
