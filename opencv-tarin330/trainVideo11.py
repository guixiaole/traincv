# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'trainVideo-11.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        # MainWindow.resize(900, 700)
        # MainWindow.resize(2100, 1300)
        MainWindow.resize(1920, 1080)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(110, 900, 78, 23))
        # self.pushButton.setGeometry(QtCore.QRect(110, 1210, 78, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        # self.pushButton_2.setGeometry(QtCore.QRect(210, 1210, 78, 23))
        self.pushButton_2.setGeometry(QtCore.QRect(210, 900, 78, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(310, 900, 78, 23))
        # self.pushButton_3.setGeometry(QtCore.QRect(310, 1210, 78, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(410, 900, 78, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.widget = QtWidgets.QLabel(self.centralwidget)
        # self.widget.setGeometry(QtCore.QRect(10, 10, 1920, 1080))
        self.widget.setGeometry(QtCore.QRect(10, 10, 1300, 800))
        self.widget.setText("")
        self.widget.setObjectName("widget")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(1850, 800, 113, 20))
        # self.lineEdit.setGeometry(QtCore.QRect(1930, 1100, 113, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(1600, 700, 61, 20))
        # self.label_2.setGeometry(QtCore.QRect(1930, 1100, 61, 20))
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.pushButton_2.clicked.connect(MainWindow.startVideo)  # 打开视频
        self.pushButton.clicked.connect(MainWindow.videoprocessiong)  # 打开视频
        self.pushButton_3.clicked.connect(MainWindow.videospeedup)  # 视频加速
        self.pushButton_4.clicked.connect(MainWindow.videospeeddown)  # 视频加速
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "打开视频"))
        self.pushButton_2.setText(_translate("MainWindow", "开始"))
        self.pushButton_3.setText(_translate("MainWindow", "加速"))
        self.pushButton_4.setText(_translate("MainWindow", "减速"))
        # self.label.setText(_translate("MainWindow", "速度显示"))
        self.label_2.setText(_translate("MainWindow", "输入速度："))
