import time
import sys

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from cv2 import *
import cv2


class VideoBox(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.rec_status = False

        # 组件展示
        self.pictureLabel = QLabel()
        init_image = QPixmap("../assets/images/no_video.jpeg").scaled(self.width(), self.height())
        self.pictureLabel.setPixmap(init_image)

        self.playButton = QPushButton()
        self.recordButton = QPushButton()
        self.recordButton.setText("开始录制")
        self.recordButton.clicked.connect(self.record)
        self.warning = QLabel()
        self.warning.setText("..........."*10)

        control_box = QHBoxLayout()
        control_box.setContentsMargins(0, 0, 0, 0)
        control_box.addWidget(self.recordButton)

        message_box = QHBoxLayout()
        message_box.setContentsMargins(0, 0, 0, 0)
        message_box.addWidget(self.warning)

        layout1 = QVBoxLayout()
        layout1.addWidget(self.pictureLabel)
        layout1.addLayout(control_box)
        layout = QHBoxLayout()
        layout.addLayout(layout1)
        layout.addLayout(message_box)

        self.setLayout(layout)
            # self.videoWriter = VideoWriter('*.mp4', VideoWriter_fourcc('M', 'J', 'P', 'G'), self.fps, size)

    def show_image(self, frame):
        frame = resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)))
        height, width = frame.shape[:2]
        if frame.ndim == 3:
            rgb = cvtColor(frame, COLOR_BGR2RGB)
        elif frame.ndim == 2:
            rgb = cvtColor(frame, COLOR_GRAY2BGR)

        temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
        temp_pixmap = QPixmap.fromImage(temp_image)
        self.pictureLabel.setPixmap(temp_pixmap)
        return

    def record(self):
        if self.rec_status:
            print("stop record")
            # self.out.release()
            self.recordButton.setText("开始录制")
            self.warning.setText("Record")
        else:
            print("record")
            self.recordButton.setText("停止录制")
            self.warning.setText("Stop Record")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # self.out = cv2.VideoWriter('output.avi', fourcc, 20.0, (self.x, self.y))
        self.rec_status = ~self.rec_status


if __name__ == "__main__":
    cap = cv2.VideoCapture('testdata/test.mov')
    r, frame = cap.read()
    app = QApplication(sys.argv)
    box = VideoBox()
    box.show()
    while True:
        r, frame = cap.read()
        cv2.imshow('', frame)
        box.show_image(frame)
        box.show()
        if not r:
            break
    sys.exit(app.exec_())
    # while True:
    #     r, frame = cap.read()
    #     imshow('1', frame)
    #     cv2.waitKey(100)

