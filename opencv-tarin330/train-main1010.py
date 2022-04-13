import datetime
import sys
import time
import cv2
from time import sleep, ctime

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
from new_singallump_replace import get_corrdinate, predict_frame, replace_frame_smooth, half_yoloandhandwork, \
    xml_handwork_corrodiance, two_path_yolo_handwork_together
from opencvyolo_0502 import frame_to_trans
# from singallump_extract import findhigh_low,findleft_right,findcenter,replace_frame_smooth,replace_image
import singallump_extract as singallump


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


class video_Box(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.housespeed = -1  # 最终的速度
        self.speed = 10  # 延时的时间
        self.frame_count = 1
        self.frame_dict2 = {}
        self.frame_dict3 = {}
        self.choose_dict = [self.frame_dict2, self.frame_dict3]
        self.distance_error = 0
        self.frame_distance, self.house_speed = getdata.getxlsxdata()
        self.housespeed_now = self.house_speed[0]  # 最开始的速度就是excel表格给的速度
        self.timenow = float(datetime.datetime.now().strftime('%S.%f'))
        self.durations = []
        #
        # self.lineEdit.editingFinished.connect(self.emittextspeed)
        # self.th.labeltext.connect(self.setlabeltext)  # 显示时速

    # 获取时速，显示在pyqt上的函数
    def setlabeltext(self, string):
        self.label.setText(str(string))

    # 获取到时速表中得到的信息
    def emittextspeed(self):
        # self.th.linetext.emit(str(self.lineEdit.text()))
        str1 = self.lineEdit.text()
        # 进行一个判断，然后将其修改
        if is_number(str1):
            self.housespeed = float(str1)

    # 开始播放视频
    def startVideo(self):
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.playvideo)
        self.timer.start(15)

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
        t1 = Thread(target=self.run2)
        t2 = Thread(target=self.run3)
        t1.start()
        t2.start()
        # self.th.changePixmap.connect(self.setImage)

    # 将图片放置界面上的槽函数
    def setImage(self, image):
        self.widget.setPixmap(QPixmap.fromImage(image))

    # 测试加速使用
    def videospeedup(self):
        self.housespeed += 10

    # 测试减速使用
    def videospeeddown(self):
        self.housespeed -= 10

    def playvideo(self, ):
        """
        播放视频，单独列出来。
        :return:
        """
        if self.frame_count % 1000 == 0:
            with open("hashmap.txt", "a") as file:
                for i in range(len(self.durations)):
                    file.write(str(self.durations[i][1]) + ":" + str(self.durations[i][0]) + ";")
                    file.write("\n")
            self.durations.clear()
        print("framedict2", len(self.frame_dict2))
        print("framedict3", len(self.frame_dict3))
        a = float(datetime.datetime.now().strftime('%S.%f')) - self.timenow
        self.timenow = float(datetime.datetime.now().strftime('%S.%f'))
        if a >= 0.03:
            self.durations.append([a, self.frame_count])  # Used to record loop time
        jump_frame = 1
        if self.frame_count > 1500:
            h = int((self.frame_count - 1500) / 50)
        else:
            h = 0
        if h >= len(self.frame_distance):
            h = len(self.frame_distance) - 1
        print("当前帧：" + str(self.frame_count))
        if self.housespeed != -1:
            if int(self.housespeed) != 0:  # 设定了速度
                self.housespeed_now = getdata.acceleration(float(self.housespeed), float(self.housespeed_now),
                                                           50)
                # 计算出了当前的速度。做个显示作用。
                # 根据s = v0t+at^2/2
                distance_everyframe = getdata.distance_frame(self.housespeed_now)  # 在这一帧走过的距离。
                # 计算需要跳出的多少帧。另外做一个误差补偿。
                jump_frame, last_distance = getdata.jump_frame_for_distance(distance_everyframe,
                                                                            self.frame_distance[h])
                self.distance_error += last_distance
                # 这边只考虑了加速。但是没有考虑减速。
                print("当前帧数的速度。", self.housespeed_now)
                print("本来当前帧数的速度。", self.house_speed[h])
                print("当前帧距离", self.frame_distance[h])
                print("jump_frame", jump_frame)
                print("这一帧走过的距离", distance_everyframe)
                print("距离误差", last_distance)
                print("总误差", self.distance_error)

        if ((self.frame_count - 1) // 100) % 2 == 0 or self.frame_count == 1:
            choose_flag = 0
        else:
            choose_flag = 1
        if str(self.frame_count) in self.choose_dict[choose_flag]:
            # frame = self.choose_dict[choose_flag][str(self.frame_count)]
            jump_flag = jump_frame
            while jump_frame > 1:
                if (self.frame_count - 1) % 100 == 0:
                    if ((self.frame_count - 1) // 100) % 2 == 0 or self.frame_count == 1:
                        choose_flag = 0
                    else:
                        choose_flag = 1
                if len(self.choose_dict[choose_flag]) <= 0:
                    continue
                else:
                    del self.choose_dict[choose_flag][str(self.frame_count)]
                    self.frame_count += 1
                    jump_frame -= 1
            if ((self.frame_count - 1) // 100) % 2 == 0 or self.frame_count == 1:
                choose_flag = 0
            else:
                choose_flag = 1
            if self.distance_error > self.frame_distance[h]:
                if len(self.choose_dict[choose_flag]) > 0:
                    self.distance_error -= self.frame_distance[h]
                    del self.choose_dict[choose_flag][str(self.frame_count)]
                    self.frame_count += 1
                    # 减去误差项的影响。
            if ((self.frame_count - 1) // 100) % 2 == 0 or self.frame_count == 1:
                choose_flag = 0
            else:
                choose_flag = 1
            if len(self.choose_dict[choose_flag]) > 0:
                # if len(choose_dict[choose_flag]) > 0:
                frame = self.choose_dict[choose_flag][str(self.frame_count)]
                if jump_flag != 0:  # 这边减少误差。
                    del self.choose_dict[choose_flag][str(self.frame_count)]
                    self.frame_count = self.frame_count + 1

                # self.changePixmap.emit(frame)

                # 7ms
                # print(type(covertToQtFormat))
                # pix = frame.scaled(1080, 860, Qt.KeepAspectRatio)
                time_now = float(datetime.datetime.now().strftime('%S.%f'))
                frame = QtGui.QImage(frame.data, frame.shape[1],
                                     frame.shape[0], QImage.Format_RGB888)
                self.widget.setPixmap(QPixmap.fromImage(frame))
                b = float(datetime.datetime.now().strftime('%S.%f')) - self.timenow
                c = float(datetime.datetime.now().strftime('%S.%f')) - time_now
                print('b:', b)
                print('c:', c)

    def run2(self):

        cap = cv2.VideoCapture(videoName)
        # 跳转到指定的帧

        step = 0
        order_frame = 1  # 跳到指定的某一帧
        # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)

        while cap.isOpened():
            if len(self.frame_dict2) <= 501:

                # time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                # with open("hashmap.txt", "a") as file:
                #     file.write(str(time_now) + ";")
                #     file.write(str(order_frame) + ";")
                #     file.write("\n")

                if (order_frame - 1) % 100 == 0 and order_frame != 1:
                    step += 2

                    cap.set(cv2.CAP_PROP_POS_FRAMES, step * 100 + 1)
                    order_frame = step * 100 + 1

                # print("当前帧：", order_frame)
                # if len(self.frame_list) < 100:

                # print('run2 order_frame:' + str(order_frame) + 'stop')
                ret, frame = cap.read()
                if ret:
                    # self.frame_list.append((order_frame, frame))
                    self.frame_dict2[str(order_frame)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # # 2ms
                    # # rgbImage = frame
                    # covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                    #                                 rgbImage.shape[0], QImage.Format_RGB888)
                    #
                    # # 7ms
                    # # print(type(covertToQtFormat))
                    # p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                    # print(type(p))
                    # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # 2ms
                    # rgbImage = frame
                    # covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                    #                                 rgbImage.shape[0], QImage.Format_RGB888)

                    # = rgbImage
                    # self.frame_list.append((self.frame_count, frame))
                    # order_frame += 1
                    order_frame += 1
            else:
                # sleep(0.0001)
                continue

    def run3(self):

        cap = cv2.VideoCapture(videoName)
        # 跳转到指定的帧
        step = 1
        order_frame = 101  # 跳到指定的某一帧
        # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
        while cap.isOpened():

            if len(self.frame_dict3) <= 501:
                time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                # print("frame_dict3 : %s" % time_now)
                # with open("hashmap1.txt", "a") as file:
                #     file.write(str(time_now) + ";")
                #     file.write(str(order_frame) + ";")
                #     file.write("\n")
                if (order_frame - 1) % 100 == 0 and order_frame != 101:
                    step += 2
                    # print('run3 step:', step,'stop')
                    cap.set(cv2.CAP_PROP_POS_FRAMES, step * 100 + 1)
                    order_frame = step * 100 + 1

                # print("当前帧：", order_frame)
                # if len(self.frame_list) < 100:
                # print('run3 order_frame:' + str(order_frame) + 'stop')
                ret, frame = cap.read()
                if ret:
                    # self.frame_list.append((order_frame, frame))
                    self.frame_dict3[str(order_frame)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # print(type(rgbImage))
                    # rgbImage = frame
                    # print("2 : %s" % time_now)
                    # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                    # 2ms
                    # rgbImage = frame
                    # covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                    #                                 rgbImage.shape[0], QImage.Format_RGB888)
                    # # print("3 : %s" % time_now)
                    # # # 7ms
                    # # # print(type(covertToQtFormat))
                    # p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                    # # print(type(p))
                    # self.frame_dict3[str(order_frame)] = rgbImage
                    # self.frame_list.append((self.frame_count, frame))
                    # order_frame += 1
                    order_frame += 1
            else:
                # sleep(0.0001)
                continue


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = video_Box()
    window.show()
    sys.exit(app.exec_())
