import os
import sys

from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.uic import loadUi

import cv2


class my_software(QMainWindow):
    def __init__(self, parent=None):
        super(my_software, self).__init__(parent)
        loadUi('test.ui', self)

        self.frame = []
        # self.detectFlag = False
        self.cap = []
        self.timer_camera = QTimer()

        video = "E:/衡阳到长沙/衡阳-岳阳.mp4"
        self.cap = cv2.VideoCapture(video)
        self.Initial_GUI()

    def Initial_GUI(self):
        self.setWindowTitle("VideoDisplay")

        self.lb_Screen.setFixedSize(self.lb_Screen.width(), self.lb_Screen.height())

        self.btn_Start.clicked.connect(self.Btn_Start)
        self.btn_Stop.clicked.connect(self.Btn_Stop)

    # 窗口关闭时间
    def Btn_Close(self, event):
        event.accept()
        self.cap.release()

    def Btn_Start(self):
        # 定时器开启，每隔一段时间，读取一帧
        self.timer_camera.start(100)
        self.timer_camera.timeout.connect(self.OpenFrame)

    def Btn_Stop(self):
        # self.cap.release()
        self.timer_camera.stop()

    def OpenFrame(self):
        ret, frame = self.cap.read()

        if ret:
            # Process
            # cv2.putText(frame,"good",(50,100),2,cv2.FONT_HERSHEY_COMPLEX,(0,0,255),3)

            self.Display_Image(frame)
        else:
            self.cap.release()
            self.timer_camera.stop()

    def Display_Image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_RGB888)
        elif len(image.shape) == 1:
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_Indexed8)
        else:
            Q_img = QImage(image.data,
                           image.shape[1],
                           image.shape[0],
                           QImage.Format_RGB888)

        self.lb_Screen.setPixmap(QtGui.QPixmap(Q_img))
        self.lb_Screen.setScaledContents(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    my = my_software()
    my.show()

    sys.exit(app.exec_())
