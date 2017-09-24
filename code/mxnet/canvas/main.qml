import QtQuick 2.6
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls.Styles 1.4
import canvas.backend 1.0

ApplicationWindow {
  id: root
  width: 500
  height: 500
  visible: true
  title: "Canvas"

  property string imageFilename: "out.png"

  BackEnd { id: backend }

  Canvas {
    id: canvas
    anchors.fill: parent
    property int posX;
    property int posY;
    property bool pressed;

    signal clear

    onPaint: {
      var ctx = getContext("2d");
      if (pressed) {
        ctx.fillStyle = "white";
        ctx.ellipse(posX, posY, 25, 25);
        ctx.fill();
      } else {
        ctx.reset();
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, width, height);
      }
    }

    onClear: {
      pressed = false;
      requestPaint();
      canvas.save(root.imageFilename);
    }

    MouseArea {
      anchors.fill: parent
      onPressed: {
        parent.pressed = true;
      }
      onPositionChanged: {
        parent.posX = mouseX;
        parent.posY = mouseY;
        parent.requestPaint();
      }
    }
  }

  RowLayout {
    anchors.bottom: parent.bottom
    anchors.horizontalCenter: parent.horizontalCenter
    Button {
      flat: true
      onClicked: {
        var prediction = backend.predict(root.imageFilename);
        predicted.text = prediction;
      }
      contentItem: Text {
        text: "Predict"
        color: "white"
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
      }
    }
    Label {
      id: predicted
      text: "9"
      font.pixelSize: 72
      color: "white"
    }
    Button {
        flat: true
        onClicked: canvas.clear();
        contentItem: Text {
          text: "Reset"
          color: "white"
          horizontalAlignment: Text.AlignHCenter
          verticalAlignment: Text.AlignVCenter
        }
    }
  }
}
