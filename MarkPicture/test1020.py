import datetime

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from time import time
import sys
import cv2
import trainVideo11  # GUI Module created by pydesigner


class VideoCapture(QMainWindow, trainVideo11.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)  # This is defined in design.py file automatically
        self.pushButton.clicked.connect(self.loadVideoFile)
        self.pushButton_2.clicked.connect(self.start)
        self.pushButton_3.clicked.connect(self.closeApplication)
        self.durations = []
        self.timenow = float(datetime.datetime.now().strftime('%S.%f'))

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = frame.shape[1]
        y = frame.shape[0]
        img = QImage(frame, x, y, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.widget.setPixmap(pix)
        # a = time() # Used to record loop time
        a = float(datetime.datetime.now().strftime('%S.%f'))-self.timenow
        self.timenow = float(datetime.datetime.now().strftime('%S.%f'))

        self.durations.append(a)  # Used to record loop time

    def start(self):
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        print("Rate = ", self.vid_rate)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(self.vid_rate)

    def loadVideoFile(self):
        self.videoFileName = "E:/衡阳到长沙/衡阳-岳阳.mp4"
        self.cap = cv2.VideoCapture(str(self.videoFileName))
        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
        self.vid_rate = 20  # reset this integer from 15 through 34
        self.nextFrameSlot()

    def closeApplication(self):
        for i in self.durations:
            print(i)
        self.cap.release()
        sys.exit(0)


def main():
    app = QApplication(sys.argv)
    form = VideoCapture()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
