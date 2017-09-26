#ifndef BACKEND_H
#define BACKEND_H

#include <QObject>
#include <QString>
#include <QTcpSocket>

class BackEnd : public QObject {
  Q_OBJECT

 public:
  explicit BackEnd(QObject* parent = nullptr);

  Q_INVOKABLE void predict(QString imageFilename);

 signals:

  void prediction(int prediction);

 private:
  QTcpSocket* socket;
};

#endif  // BACKEND_H
