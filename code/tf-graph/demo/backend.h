#ifndef BACKEND_H
#define BACKEND_H

#include <QObject>
#include <QString>
#include <QTcpSocket>

class BackEnd : public QObject {
  Q_OBJECT

 public:
  explicit BackEnd(QObject* parent = nullptr);

  Q_INVOKABLE void generateImage(int digit, double a, double b);

 signals:

  void imageReady(QString imagePath);

 private:
  QTcpSocket* socket;
};

#endif  // BACKEND_H
