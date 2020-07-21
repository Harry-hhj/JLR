import sys
from Demo import Ui_MainWindow, identifier
from PyQt5 import QtWidgets
from PyQt5.QtGui import QImage, QPixmap
import cv2, time
from datetime import datetime

"""""""""""""""""""""""""""""""""""""""""""""""""""
|identifier:3a879f86-dfde-45b0-90c4-73e14fd77fe8  |
"""""""""""""""""""""""""""""""""""""""""""""""""""

assert(identifier == '3a879f86-dfde-45b0-90c4-73e14fd77fe8')


class mywindow(QtWidgets.QMainWindow, Ui_MainWindow):  # 这个地方要注意Ui_MainWindow
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        frame = cv2.imread("testdata/victory.jpeg")
        for i in ["main_demo", "sub_demo1", "sub_demo2"]:
            self.set_image(frame, i)
        del frame
        self.feedback_message_box = []
        self.alarm_location_message_box = []
        self.set_text("feedback", "intializing...")
        self.record_state = False  # 0:开始 1:停止
        self.btn1.setText("开始录制")
        self.btn2.setText("无用")
        self.btn3.setText("无用")
        self.btn4.setText("无用")

    # 。。。加自己的函数等
    def btn1_on_clicked(self):
        self.record_state = not self.record_state
        if not self.record_state:
            self.btn1.setText("开始录制")
            save_address = ""
            self.set_text("feedback", "录制已保存于{0}".format(save_address))
        else:
            self.btn1.setText("停止录制")
            self.set_text("feedback", "录制已开始于{0}".format(datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')))

    def btn2_on_clicked(self):
        print("btn2")

    def btn3_on_clicked(self):
        print("btn3")

    def btn4_on_clicked(self):
        print("btn4")

    def set_image(self, frame, position=""):
        """

        :param frame: cv form
        :param position:
        :return:
        """
        assert (position in ["main_demo", "sub_demo1", "sub_demo2"])
        if position == "main_demo":
            width = self.main_demo.width()
            height = self.main_demo.height()
        elif position == "sub_demo1":
            width = self.sub_demo1.width()
            height = self.sub_demo1.height()
        elif position == "sub_demo2":
            width = self.sub_demo2.width()
            height = self.sub_demo2.height()
        frame = cv2.resize(frame, (int(width), int(height)))
        if frame.ndim == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            return False

        temp_image = QImage(rgb.flatten(), width, height, QImage.Format_RGB888)
        #temp_image = QImage(rgb.flatten(), width*2, height*2, QImage.Format_RGB888)
        temp_pixmap = QPixmap.fromImage(temp_image)
        if position == "main_demo":
            self.main_demo.setPixmap(temp_pixmap)
            self.main_demo.setScaledContents(True)
        elif position == "sub_demo1":
            self.sub_demo1.setPixmap(temp_pixmap)
            self.sub_demo1.setScaledContents(True)
        elif position == "sub_demo2":
            self.sub_demo2.setPixmap(temp_pixmap)
            self.sub_demo2.setScaledContents(True)
        #self.pictureLabel.setPixmap(temp_pixmap)
        return True

    def set_text(self, position: str, message=""):
        """
        if you want a blank line please use set_text(position[, ""]) explicitly, do not hide it in message in the
        form of "\n\n"
        :param position: must be one of the followings: "feedback", "message_box", "alarm_location", "small_space"
        :param message: Either a single string or a compound string splited by '\n' is accepted.
        :return:
        True if operation succeeded.
        """
        if '\n' not in message:
            if len(message) > 50:
                print("The message may not be fully displayed.")
            if position not in ["feedback", "message_box", "alarm_location", "small_space"]:
                return False
            if position == "feedback":
                if len(self.feedback_message_box) >= 12:
                    self.feedback_message_box.pop(0)
                self.feedback_message_box.append(message)
                message = "\n".join(self.feedback_message_box)
                self.feedback.setText(message)
            elif position == "message_box":
                self.message_box.setText(message)
            elif position == "alarm_location":
                if len(self.alarm_location_message_box) >= 20:
                    self.alarm_location_message_box.pop(0)
                self.alarm_location_message_box.append(message)
                message = "\n".join(self.alarm_location_message_box)
                self.alarm_location.setText(message)
            elif position == "small_space":
                self.small_space.setText(message)
            return True
        else:
            message = message.split("\n")
            message.remove("")
            print(message)
            flag = True
            for s in message:
                flag = flag and self.set_text(position, s)
            return flag


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    myshow.set_text("alarm_location", "aaa")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "bbb")
    myshow.set_text("alarm_location", "ccc\n"*5)
    myshow.show()  # 显示
    sys.exit(app.exec_())
