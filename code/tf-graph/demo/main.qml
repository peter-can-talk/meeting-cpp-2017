import QtQuick 2.6
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.3
import QtQuick.Controls.Styles 1.4
import demo.backend 1.0

ApplicationWindow {
  id: root
  width: 400
  height: 450
  visible: true
  title: "Demo"
  color: "black"

  Image {
    id: image
    width: 400
    height: 280
    fillMode: Image.PreserveAspectFit
    source: "file:/tmp/gan-out.png"
    cache: false
  }

  BackEnd {
    id: backend
    function generate() {
      backend.generateImage(digit.value, a.value, b.value);
    }
    onImageReady: image.source = "file:" + imagePath
  }

  ColumnLayout {
    anchors.top: image.bottom
    anchors.bottom: parent.bottom
    anchors.bottomMargin: 10
    anchors.horizontalCenter: parent.horizontalCenter
    Slider {
      id: digit
      from: 0
      to: 9
      value: 0
      stepSize: 1
      snapMode: Slider.SnapAlways
      onMoved: backend.generate()

      Text {
        anchors.right: parent.left
        anchors.rightMargin: 10
        anchors.verticalCenter: parent.verticalCenter
        text: Math.ceil(digit.value)
        color: "white"
        font.pixelSize: 20
      }
    }

    Slider {
      id: a
      from: -3
      to: +3
      value: 0
      stepSize: 0.1
      snapMode: Slider.SnapAlways
      onMoved: backend.generate()

      Text {
        anchors.right: parent.left
        anchors.rightMargin: 10
        anchors.verticalCenter: parent.verticalCenter
        text: a.value.toPrecision(1)
        color: "white"
        font.pixelSize: 20
      }
    }

    Slider {
      id: b
      from: -3
      to: +3
      value: 0
      stepSize: 0.1
      snapMode: Slider.SnapAlways
      onMoved: backend.generate()

      Text {
        anchors.right: parent.left
        anchors.rightMargin: 10
        anchors.verticalCenter: parent.verticalCenter
        text: b.value.toPrecision(1)
        color: "white"
        font.pixelSize: 20
      }
    }
  }
}
