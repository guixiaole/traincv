import datetime
import sys
import time
import cv2
from threading import Thread, Lock
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import *
# sys.path.append("./opencv-train330/get_data")
import get_data as getdata
from trainVideo11 import Ui_MainWindow
from get_frame_singal import get_frame_singal
from new_singallump_replace import get_corrdinate, predict_frame, replace_frame_smooth, half_yoloandhandwork
from opencvyolo_0502 import frame_to_trans
# from singallump_extract import findhigh_low,findleft_right,findcenter,replace_frame_smooth,replace_image
import singallump_extract as singallump


class video_Box(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.th = Thread1()

        # self.lineEdit.editingFinished.connect(self.emittextspeed)
        # self.th.labeltext.connect(self.setlabeltext)  # 显示时速
        self.th.changePixmap.connect(self.setImage)
        self.th.start()

    # 获取时速，显示在pyqt上的函数
    def setlabeltext(self, string):
        self.label.setText(str(string))

    # 获取到时速表中得到的信息
    def emittextspeed(self):
        # self.th.linetext.emit(str(self.lineEdit.text()))
        str1 = self.lineEdit.text()
        self.th.sethousespeed(str1)

    # 开始播放视频
    def startVideo(self):
        self.th.start()

    # 打开视频
    def videoprocessiong(self):
        global videoName
        videoName, videoType = QFileDialog.getOpenFileName(self,
                                                           "打开视频",
                                                           "",
                                                           " *.mp4;;*.avi;;All Files (*)"
                                                           )
        # self.th.timesingal.signal[str].connect(self.showvideo())
        print(videoName)
        self.th.changePixmap.connect(self.setImage)

    # 将图片放置界面上的槽函数
    def setImage(self, image):
        self.widget.setPixmap(QPixmap.fromImage(image))
        self.widget.setScaledContents(True)

    # 测试加速使用
    def videospeedup(self):
        self.th.stop()

    # 测试减速使用
    def videospeeddown(self):
        self.th.down()


class Thread1(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)
    labeltext = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)

        self.housespeed = -1  # 最终的速度
        self.speed = 15
        self.housespeed_now = 0  # 现在的速度

    # linetext = pyqtSignal(str)
    # print(str(linetext.__getitem__(str)))

    def run(self):
        # MP4的格式播放每帧需要耗时大概在10毫秒左右
        # avi格式则需要5到6毫秒左右
        #
        frame_distance, house_speed = getdata.getxlsxdata()
        self.housespeed_now = house_speed[0]  # 最开始的速度就是excel表格给的速度
        cap = cv2.VideoCapture("E:/衡阳到长沙/衡阳-岳阳.mp4")
        # 跳转到指定的帧
        order_frame = 1  # 跳到指定的某一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
        # print(videoName)
        distance_count = 0.0
        distance_error = 0
        while cap.isOpened():
            ret, frame = cap.read()
            print("当前帧：", order_frame)
            if ret:
                if order_frame >= 0:
                    # sps=self.getspeed()
                    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                    print("Start : %s" % time_now)
                    h = int(order_frame / 50)
                    '''
                    distance_count = float(int(self.housespeed) * 1000 / 3600
                                           / frame_now) + distance_count
                    '''
                    # 该段反应的是距离。假设超帧了，则按照最后一秒的速度计算
                    if h >= len(frame_distance):
                        h = len(frame_distance) - 1
                    distance_count = distance_count + frame_distance[h]
                    if self.housespeed != -1:  # 设定了速度
                        # print(house_speed)
                        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count+5)#这样跳帧是不行的。
                        # frame_count+=4
                        self.housespeed_now = getdata.acceleration(float(self.housespeed), float(self.housespeed_now),
                                                                   50)
                        # 计算出了当前的速度。做个显示作用。
                        # 根据s = v0t+at^2/2
                        distance_everyframe = getdata.distance_frame(self.housespeed_now)  # 在这一帧走过的距离。
                        # 计算需要跳出的多少帧。另外做一个误差补偿。
                        jump_frame, last_distance = getdata.jump_frame_for_distance(distance_everyframe,
                                                                                    frame_distance[h])
                        # 这边只考虑了加速。但是没有考虑减速。
                        distance_error += last_distance
                        print("当前速度:", self.housespeed_now)
                    '''
                    是这样的要在这个中间插上一个更改信号灯的功能
                    首先要获取信号灯出现的帧数，其次的话就是获得信号灯在这一帧的图片
                    
                    '''
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                    covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                                                    rgbImage.shape[0], QImage.Format_RGB888)
                    p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)

                    h = round(distance_count / 1000, 4)
                    # self.labeltext.emit(str(h))
                order_frame = order_frame + 1

            else:
                break

            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            print("END : %s" % time_now)

    def stop(self):
        self.setspeed(10)

    def setspeed(self, speed):
        self.speed = speed

    def getspeed(self):
        return self.speed

    def down(self):
        self.setspeed(100)

    def gethousespeed(self):
        return self.housespeed

    def sethousespeed(self, str1):
        self.housespeed = str1


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = video_Box()
    window.show()
    sys.exit(app.exec_())
