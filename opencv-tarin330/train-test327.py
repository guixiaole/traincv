import datetime
import sys
import time
import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import *
# sys.path.append("./opencv-train330/get_data")
import get_data as getdata
from trainVideo import Ui_MainWindow
from get_frame_singal import get_frame_singal
from opencvyolo_0502 import frame_to_trans
# from singallump_extract import findhigh_low,findleft_right,findcenter,replace_frame_smooth,replace_image
import singallump_extract as singallump


class video_Box(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.th = Thread()
        self.lineEdit.editingFinished.connect(self.emittextspeed)
        self.th.labeltext.connect(self.setlabeltext)  # 显示时速

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

    # 测试加速使用
    def videospeedup(self):
        self.th.stop()

    # 测试减速使用
    def videospeeddown(self):
        self.th.down()


class Thread(QThread):
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
        cap = cv2.VideoCapture(videoName)
        # 跳转到指定的帧
        order_frame = 73721  # 跳到指定的某一帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
        print(videoName)
        distance_count = 0.0
        frame_count = order_frame  # 帧的数量
        time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print("Start : %s" % time_now)
        frame_now = 50  # 初始值是50帧每秒
        frame_singal = get_frame_singal()  # 指的是获取信号灯的那一帧
        replace_box_smooth = singallump.replace_frame_smooth()
        while (cap.isOpened() == True):
            ret, frame = cap.read()
            print("当前帧：", frame_count)
            if ret:
                if frame_count >= 0:
                    # sps=self.getspeed()
                    h = int(frame_count / 50)
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
                        self.housespeed_now = getdata.acceleration(float(self.housespeed), float(self.housespeed_now),
                                                                   frame_now)
                        self.speed, frame_now = getdata.speed_up(self.housespeed_now, house_speed[h])
                        print("当前速度:", self.housespeed_now)
                    '''
                    是这样的要在这个中间插上一个更改信号灯的功能
                    首先要获取信号灯出现的帧数，其次的话就是获得信号灯在这一帧的图片
                    
                    '''
                    print(frame.shape)
                    if len(frame_singal) > 0 and frame_count == frame_singal[0][0]:
                        (singalump_frame, boxes, conf) = frame_singal.pop(0)  # 这表明获取到了这一帧的照片了
                        detect_box = replace_box_smooth.pop(0)
                        detect_box.pop(0)  # 将帧给吐出来。剩下的就是box了
                        frame = singallump.replace_image(frame, detect_box)

                        # frame=singallump.findcenter(frame,boxes)

                    cv2.waitKey(int(self.speed))
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                    covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                                                    rgbImage.shape[0], QImage.Format_RGB888)
                    p = covertToQtFormat.scaled(1770, 1100, Qt.KeepAspectRatio)
                    self.changePixmap.emit(p)

                    h = round(distance_count / 1000, 4)
                    self.labeltext.emit(str(h))
                frame_count = frame_count + 1

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
