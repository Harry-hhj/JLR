# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Demo_v1.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
|  JiaoLong Radar Dashboard                                         |
|  Created by hhj on 7.21                                           |
|  identifier:3a879f86-dfde-45b0-90c4-73e14fd77fe8                  |
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from PyQt5 import QtCore, QtGui, QtWidgets

identifier = '3a879f86-dfde-45b0-90c4-73e14fd77fe8'


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1337, 899)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.left = QtWidgets.QGridLayout()
        self.left.setContentsMargins(5, 5, 5, 5)
        self.left.setSpacing(10)
        self.left.setObjectName("left")
        self.btn4 = QtWidgets.QPushButton(self.centralwidget)
        self.btn4.setMinimumSize(QtCore.QSize(115, 35))
        self.btn4.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.btn4.setObjectName("btn4")
        self.left.addWidget(self.btn4, 1, 3, 1, 1)
        self.small_space = QtWidgets.QLabel(self.centralwidget)
        self.small_space.setMinimumSize(QtCore.QSize(470, 35))
        self.small_space.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.small_space.setText("")
        self.small_space.setObjectName("small_space")
        self.left.addWidget(self.small_space, 1, 4, 1, 2)
        self.message_box = QtWidgets.QLabel(self.centralwidget)
        self.message_box.setMinimumSize(QtCore.QSize(475, 225))
        self.message_box.setMaximumSize(QtCore.QSize(16777215, 225))
        self.message_box.setObjectName("message_box")
        self.left.addWidget(self.message_box, 2, 5, 1, 1)
        self.btn3 = QtWidgets.QPushButton(self.centralwidget)
        self.btn3.setMinimumSize(QtCore.QSize(115, 35))
        self.btn3.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.btn3.setObjectName("btn3")
        self.left.addWidget(self.btn3, 1, 2, 1, 1)
        self.btn2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn2.setMinimumSize(QtCore.QSize(115, 35))
        self.btn2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.btn2.setObjectName("btn2")
        self.left.addWidget(self.btn2, 1, 1, 1, 1)
        self.btn1 = QtWidgets.QPushButton(self.centralwidget)
        self.btn1.setMinimumSize(QtCore.QSize(115, 35))
        self.btn1.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.btn1.setObjectName("btn1")
        self.left.addWidget(self.btn1, 1, 0, 1, 1)
        self.feedback = QtWidgets.QLabel(self.centralwidget)
        self.feedback.setMinimumSize(QtCore.QSize(475, 225))
        self.feedback.setMaximumSize(QtCore.QSize(16777215, 225))
        self.feedback.setObjectName("feedback")
        self.left.addWidget(self.feedback, 2, 0, 1, 5)
        self.main_demo = QtWidgets.QLabel(self.centralwidget)
        self.main_demo.setMinimumSize(QtCore.QSize(960, 540))
        self.main_demo.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.main_demo.setObjectName("main_demo")
        self.left.addWidget(self.main_demo, 0, 0, 1, 6)
        self.left.setColumnStretch(0, 1)
        self.left.setColumnStretch(1, 1)
        self.left.setColumnStretch(2, 1)
        self.left.setColumnStretch(3, 1)
        self.left.setColumnStretch(4, 1)
        self.left.setColumnStretch(5, 5)
        self.left.setRowStretch(0, 108)
        self.left.setRowStretch(1, 7)
        self.left.setRowStretch(2, 45)
        self.horizontalLayout.addLayout(self.left)
        self.right = QtWidgets.QVBoxLayout()
        self.right.setContentsMargins(5, 5, 5, 5)
        self.right.setSpacing(10)
        self.right.setObjectName("right")
        self.sub_demo1 = QtWidgets.QLabel(self.centralwidget)
        self.sub_demo1.setMinimumSize(QtCore.QSize(320, 180))
        self.sub_demo1.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.sub_demo1.setObjectName("sub_demo1")
        self.right.addWidget(self.sub_demo1)
        self.sub_demo2 = QtWidgets.QLabel(self.centralwidget)
        self.sub_demo2.setMinimumSize(QtCore.QSize(320, 180))
        self.sub_demo2.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.sub_demo2.setObjectName("sub_demo2")
        self.right.addWidget(self.sub_demo2)
        self.alarm_location = QtWidgets.QLabel(self.centralwidget)
        self.alarm_location.setEnabled(True)
        self.alarm_location.setMinimumSize(QtCore.QSize(320, 440))
        self.alarm_location.setObjectName("alarm_location")
        self.right.addWidget(self.alarm_location)
        self.right.setStretch(0, 9)
        self.right.setStretch(1, 9)
        self.right.setStretch(2, 22)
        self.horizontalLayout.addLayout(self.right)
        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout_2.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1337, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        self.btn1.clicked.connect(MainWindow.btn1_on_clicked)
        self.btn2.clicked.connect(MainWindow.btn2_on_clicked)
        self.btn3.clicked.connect(MainWindow.btn3_on_clicked)
        self.btn4.clicked.connect(MainWindow.btn4_on_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn4.setText(_translate("MainWindow", "btn4"))
        self.message_box.setText(_translate("MainWindow", "message_box"))
        self.btn3.setText(_translate("MainWindow", "btn3"))
        self.btn2.setText(_translate("MainWindow", "btn2"))
        self.btn1.setText(_translate("MainWindow", "btn1"))
        self.feedback.setText(_translate("MainWindow", "feedback"))
        self.main_demo.setText(_translate("MainWindow", "main_demo"))
        self.sub_demo1.setText(_translate("MainWindow", "sub_demo1"))
        self.sub_demo2.setText(_translate("MainWindow", "sub_demo2"))
        self.alarm_location.setText(_translate("MainWindow", "alarm_location"))