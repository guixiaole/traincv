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


class video_Box(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.th = Thread1()

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


class Thread1(QThread):
    changePixmap = pyqtSignal(QtGui.QImage)
    labeltext = pyqtSignal(str)

    def __init__(self):
        QThread.__init__(self)
        self.housespeed = -1  # 最终的速度
        self.speed = 10  # 延时的时间
        self.housespeed_now = 0  # 现在的速度
        self.frame_list = []
        self.frame_count = 1
        self.frame_dict2 = {}
        self.frame_dict3 = {}
        # self.cap = cv2.VideoCapture('E:\衡阳到长沙\衡阳-岳阳.mp4')

    # linetext = pyqtSignal(str)
    # print(str(linetext.__getitem__(str)))
    def run2(self):

        cap = cv2.VideoCapture(videoName)
        # 跳转到指定的帧
        step = 0
        order_frame = 1  # 跳到指定的某一帧
        # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)

        while cap.isOpened():
            if len(self.frame_dict2) <= 101:
                # time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                # with open("hashmap.txt", "a") as file:
                #     file.write(str(time_now) + ";")
                #     file.write(str(order_frame) + ";")
                #     file.write("\n")
                # print("frame_dict2 : %s" % time_now)
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
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # print(type(rgbImage))
                    # rgbImage = frame
                    # print("2 : %s" % time_now)
                    # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                    # 2ms
                    # rgbImage = frame
                    covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                                                    rgbImage.shape[0], QImage.Format_RGB888)
                    # print("3 : %s" % time_now)
                    # 7ms
                    # print(type(covertToQtFormat))
                    p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                    # print(type(p))
                    self.frame_dict3[str(order_frame)] = p
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
            if len(self.frame_dict3) <= 101:
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
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # # print(type(rgbImage))
                    # # rgbImage = frame
                    # # print("2 : %s" % time_now)
                    # # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                    # # 2ms
                    # rgbImage = frame
                    covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                                                    rgbImage.shape[0], QImage.Format_RGB888)
                    # # print("3 : %s" % time_now)
                    # # 7ms
                    # # print(type(covertToQtFormat))
                    p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                    # print(type(p))
                    self.frame_dict3[str(order_frame)] = p
                    # self.frame_list.append((self.frame_count, frame))
                    # order_frame += 1
                    order_frame += 1
            else:
                # sleep(0.0001)
                continue

    def run(self):
        # MP4的格式播放每帧需要耗时大概在10毫秒左右
        # avi格式则需要5到6毫秒左右

        # 统一按照100帧/s来计算。
        # 这样的话基本上每帧的基础处理时间是10ms
        # 所以初始值的wakity是10ms
        # 在这个基础上进行修改速度。（不做跳帧处理）
        t1 = Thread(target=self.run2)
        t1.start()
        t2 = Thread(target=self.run3)
        t2.start()
        # t1.join()
        # t2.join()
        # sleep(1)
        frame_distance, house_speed = getdata.getxlsxdata()
        self.housespeed_now = house_speed[0]  # 最开始的速度就是excel表格给的速度

        # print(frame_distance)
        distance_count = 0.0
        frame_count = self.frame_count  # 帧的数量
        time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print("Start : %s" % time_now)
        frame_now = 50  # 初始值是50帧每秒
        # replace_box_smooth=predict_frame()
        distance_error = 0  # 距离误差。
        time_final = float(datetime.datetime.now().strftime('%S.%f'))
        time_judge_minus = 0  # 动态的改变timenowjudge
        time_judge_normal = 0.019  # 动态的调整每一帧的间隔。
        all_time = 0
        falg_time_minus = 1
        choose_dict = [self.frame_dict2, self.frame_dict3]
        while True:
            time_now_judge = abs(float(datetime.datetime.now().strftime('%S.%f')) - time_final)
            if time_judge_minus != 0 and falg_time_minus == 1:  # 这边作为
                time_judge_normal = time_judge_normal - time_judge_minus
                with open("timejudgenormale.txt", "a") as file:
                    file.write(str(self.frame_count) + ":time_now_judge:" +
                               str(time_judge_minus) + ":time_judge_normal:" + str(time_judge_normal) + ";")
                    file.write("\n")
                time_judge_minus = 0
                falg_time_minus = 0
            if time_now_judge >= time_judge_normal or self.frame_count == 1:
                time_start = float(datetime.datetime.now().strftime('%S.%f'))
                time_judge_normal = 0.019  # 动态的调整每一帧的间隔。
                falg_time_minus = 1
                if self.housespeed != 0:
                    print("timejudge_normae:", time_judge_normal)
                    print("time_now_judge", time_now_judge)
                # with open("timejudge.txt", "a") as file:
                #     file.write(str(self.frame_count) + ":time_now_judge:" + str(time_now_judge) + " all_time:" + str(
                #         all_time) + ";")
                #     file.write("\n")

                # sps=self.getspeed()
                # print("当前帧：", frame_count)
                # time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                # print("Start : %s" % time_now)
                if self.frame_count > 1500:
                    h = int((self.frame_count - 1500) / 50)
                else:
                    h = 0
                '''
                distance_count = float(int(self.housespeed) * 1000 / 3600
                                       / frame_now) + distance_count
                '''
                # 该段反应的是距离。假设超帧了，则按照最后一秒的速度计算
                if h >= len(frame_distance):
                    h = len(frame_distance) - 1

                distance_count = distance_count + frame_distance[h]
                jump_frame = 1
                if self.housespeed != -1:
                    if int(self.housespeed) != 0:  # 设定了速度
                        # print(house_speed)
                        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count+5)#这样跳帧是不行的。
                        # frame_count+=4
                        # house_speed_final = self.housespeed
                        time_housespeed = float(datetime.datetime.now().strftime('%S.%f'))
                        self.housespeed_now = getdata.acceleration(float(self.housespeed), float(self.housespeed_now),
                                                                   frame_now)
                        # 计算出了当前的速度。做个显示作用。
                        # 根据s = v0t+at^2/2
                        distance_everyframe = getdata.distance_frame(self.housespeed_now)  # 在这一帧走过的距离。
                        # 计算需要跳出的多少帧。另外做一个误差补偿。
                        jump_frame, last_distance = getdata.jump_frame_for_distance(distance_everyframe,
                                                                                    frame_distance[h])
                        # 这边只考虑了加速。但是没有考虑减速。
                        distance_error += last_distance
                        time_housespeed = abs(float(datetime.datetime.now().strftime('%S.%f')) - time_housespeed)
                        # with open("time_housespeed.txt", "a") as file:
                        #     file.write(str(self.frame_count) + ":time_housespeed:" + str(time_housespeed) + ";")
                        #     file.write("\n")
                        # # self.housespeed_now 指的应该是当前的速度。
                        print("当前帧数的速度。", self.housespeed_now)
                        print("本来当前帧数的速度。", house_speed[h])
                        # # 也就是在
                        # # 想要的速度，现在的速度，以及帧数。
                        # self.speed, frame_now, beisu = getdata.speed_up(self.housespeed_now, house_speed[h])
                        # 计算出了每帧多少距离。计算每帧的加速度是多少
                '''
                是这样的要在这个中间插上一个更改信号灯的功能4
                首先要获取信号灯出现的帧数，其次的话就是获得信号灯在这一帧的图片
                '''
                time_now10 = float(datetime.datetime.now().strftime('%S.%f'))
                if len(self.frame_dict2) > 3 and int(self.housespeed) != 0 and len(self.frame_dict3) > 3:

                    print('frameList2', len(self.frame_dict2))
                    print('frameList3', len(self.frame_dict3))

                    choose_flag = -1
                    # frame_count, frame = self.frame_list[0]
                    # if str(self.frame_count) in self.frame_dict2 or str(
                    #         self.frame_count) in self.frame_dict3:
                    #     if str(self.frame_count) in self.frame_dict2:
                    #         choose_flag = 0
                    #         frame = self.frame_dict2[str(self.frame_count)]
                    #     else:
                    #         frame = self.frame_dict3[str(self.frame_count)]
                    #         choose_flag = 1
                    if ((self.frame_count - 1) // 100) % 2 == 0 or self.frame_count == 1:
                        choose_flag = 0
                    else:
                        choose_flag = 1
                    if str(self.frame_count) in choose_dict[choose_flag]:
                        frame = choose_dict[choose_flag][str(self.frame_count)]
                        # print("当前的跳帧数量", jump_frame)
                        jump_flag = jump_frame
                        with open("judgechoosedict.txt", "a") as file:
                            file.write(str(self.frame_count) + ":judgechoosedict:" + str(
                                abs(float(datetime.datetime.now().strftime('%S.%f')) - time_now10)) + ";")
                            file.write("\n")
                        time_nowjump = float(datetime.datetime.now().strftime('%S.%f'))
                        while jump_frame > 1:
                            if (self.frame_count - 1) % 100 == 0:
                                if str(self.frame_count) in self.frame_dict2:
                                    choose_flag = 0
                                elif str(self.frame_count) in self.frame_dict3:
                                    choose_flag = 1
                                else:
                                    continue
                            if len(choose_dict[choose_flag]) <= 0 or str(self.frame_count) not in \
                                    choose_dict[choose_flag]:
                                # frame = frame_last

                                time_now1 = float(datetime.datetime.now().strftime('%S.%f'))
                                with open("jumpframecontinue.txt", "a") as file:
                                    file.write(str(self.frame_count) + ":jump frame continue:" + str(
                                        abs(time_now1 - time_now10)) + ";")
                                    file.write("\n")
                                print("跳帧的continue : %s" % time_now)
                                continue
                                # break
                            else:
                                # frame_last = choose_dict[choose_flag][str(self.frame_count)]
                                # time_now2 = datetime.datetime.now().strftime('%H:%M:%S.%f')
                                time_now2 = float(datetime.datetime.now().strftime('%S.%f'))
                                del choose_dict[choose_flag][str(self.frame_count)]
                                with open("jumpframe.txt", "a") as file:
                                    # file.write(str(self.frame_count) + ":jump frame spend time: " + str(
                                    #     abs(time_now2 - time_now10)) + ";")
                                    file.write(
                                        str(self.frame_count) + ":jumpframe all: " + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now10)) + ";"
                                        + "jumpframe:" + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now2)))
                                    file.write("\n")
                                # print("跳几帧花费的时间 : %s" % time_now)

                                self.frame_count += 1
                                jump_frame -= 1
                        with open("timejump.txt", "a") as file:
                            file.write(
                                str(self.frame_count) + ":timejump all: " + str(abs(
                                    float(datetime.datetime.now().strftime('%S.%f')) - time_now10)) + ";"
                                + "timejump:" + str(abs(
                                    float(datetime.datetime.now().strftime('%S.%f')) - time_nowjump)))
                            file.write("\n")

                        if distance_error > frame_distance[h]:
                            if len(choose_dict[choose_flag]) > 0 and str(self.frame_count) in \
                                    choose_dict[choose_flag]:
                                distance_error -= frame_distance[h]
                                # frame_last = choose_dict[choose_flag][str(self.frame_count)]
                                # print("误差 : %s" % time_now)
                                time_now3 = float(datetime.datetime.now().strftime('%S.%f'))

                                del choose_dict[choose_flag][str(self.frame_count)]
                                with open("distanceerror.txt", "a") as file:
                                    # file.write(str(self.frame_count) + ":distance error: " + str(
                                    #     abs(time_now3 - time_now10)) + ";")
                                    file.write(
                                        str(self.frame_count) + ":distance error all: " + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now10)) + ";"
                                        + "distance error:" + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now3)))
                                    file.write("\n")

                                self.frame_count += 1
                                # 减去误差项的影响。
                        if len(choose_dict[choose_flag]) > 0 and str(self.frame_count) in \
                                choose_dict[choose_flag]:
                            # if len(choose_dict[choose_flag]) > 0:
                            if jump_flag == 0:  # 这边减少误差。
                                # distance_error -= frame_distance[h]
                                # frame_count, frame = self.frame_list[0]
                                # frame_last = choose_dict[choose_flag][str(self.frame_count)]
                                # time_now9 = datetime.datetime.now().strftime('%H:%M:%S.%f')
                                # print("jump=0 : %s" % time_now)
                                time_now9 = float(datetime.datetime.now().strftime('%S.%f'))
                                frame = choose_dict[choose_flag][str(self.frame_count)]

                                with open("jump0.txt", "a") as file:
                                    file.write(
                                        str(self.frame_count) + ":jump=0all: " + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now10)) + ";"
                                        + "jump=0:" + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now9)) + "" +
                                        str(abs(float(datetime.datetime.now().strftime('%S.%f')) - time_nowjump))
                                    )
                                    file.write("\n")
                            else:
                                # frame_last = choose_dict[choose_flag][str(self.frame_count)]
                                time_now12 = float(datetime.datetime.now().strftime('%S.%f'))
                                frame = choose_dict[choose_flag][str(self.frame_count)]
                                del choose_dict[choose_flag][str(self.frame_count)]
                                self.frame_count = self.frame_count + 1

                                with open("normalreadrframe.txt", "a") as file:
                                    # file.write(str(self.frame_count) + ":normal read frame： " + str(
                                    #     abs(time_now12 - time_now10)) + ";")
                                    file.write(
                                        str(self.frame_count) + ":normalreadrframe all: " + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now10)) + ";"
                                        + "normalreadrframe:" + str(abs(
                                            float(datetime.datetime.now().strftime('%S.%f')) - time_now12)))
                                    file.write("\n")
                        print("当前帧：" + str(self.frame_count))
                        time_now7 = float(datetime.datetime.now().strftime('%S.%f'))
                        # print("4 : %s" % time_now)
                        self.changePixmap.emit(frame)
                        with open("emit.txt", "a") as file:
                            file.write(
                                str(self.frame_count) + ":emit: all: " + str(abs(
                                    float(datetime.datetime.now().strftime('%S.%f')) - time_now10)) + ";"
                                + ":emit:" + str(abs(
                                    float(datetime.datetime.now().strftime('%S.%f')) - time_now7)))
                            # file.write(str(self.frame_count) + ":emit:" + str(
                            #     abs(float(datetime.datetime.now().strftime('%S.%f')) - time_now7)) + ";")
                            file.write("\n")
                        time_final = float(datetime.datetime.now().strftime('%S.%f'))
                        all_time = abs(time_final - time_start)
                        # h = round(distance_count / 1000, 4)
                        # self.labeltext.emit(str(h))

                    else:
                        time_final = float(datetime.datetime.now().strftime('%S.%f'))
                        continue
                else:
                    time_final = float(datetime.datetime.now().strftime('%S.%f'))

                    continue
                time_final = float(datetime.datetime.now().strftime('%S.%f'))
                time_now8 = float(datetime.datetime.now().strftime('%S.%f'))
                time_judge_minus += abs(time_now8 - time_now10)
                time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                print("END : %s" % time_now)
            else:
                continue

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
