import configparser
import csv
from datetime import datetime
import xml.dom.minidom
import math
import cv2
import numpy as np
from typing import List
from numpy import *
from get_frame_singal import get_frame_singal
import os
import re
from opencvyolo_0502 import yolov3_detect, findnet, finln_out
# from opencvyolo_0502 import yolov3_detect,findnet
from singallump_extract import findcenter, findcenter_test
import time
import numpy as np
import math


def replace_image(image, box):
    """
    进行图片的替换
    :param image:源图片
    :param box: 坐标
    :return:
    """
    # 这里仅仅只是将源文件传进来，然后再进行替换
    image_trans_path = "D:/train_photo_all/trans/trans3.png"
    image_trans = cv2.imread(image_trans_path)
    shrink = cv2.resize(image_trans, (box[2], box[3]),
                        interpolation=cv2.INTER_AREA)
    temp = shrink
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)
    for i in range(box[2] - 1):
        for j in range(box[3] - 1):
            if temp[j][i][0] < 100 or temp[j][i][1] < 100 or temp[j][i][2] < 100:
                image[box[1] + j][box[0] + i] = shrink[j][i]
    # cv2.imshow('image1', image)
    return image


def detect_smooth(frame_detect):
    """
    目标检测的平滑过渡，主要功能是当有一些目标检测可能没有识别出来
    如果这一帧没有识别出来的话。那就给他加上最近的一帧的

    :return:
    """
    flag = 1
    while flag < len(frame_detect):
        temp_len = frame_detect[flag][0] - frame_detect[flag - 1][0]
        if temp_len > 1:
            temp_frame = []
            for i in range(len(frame_detect[flag - 1])):
                if i == 0:
                    temp_frame.append(frame_detect[flag - 1][i] + 1)
                else:
                    temp_frame.append(frame_detect[flag - 1][i])
            frame_detect.insert(flag, temp_frame)
        else:
            flag += 1
    # 做一个平滑过渡
    for j in range(1, len(frame_detect[0])):  # 4个坐标进行检测
        # 先检查是递减还是递增，递减flag=0，递增则是1
        temp = frame_detect[0][j] - frame_detect[len(frame_detect) // 2][j]
        if temp > 0:
            flag = 0
        elif temp < 0:
            flag = 1
        else:
            temp = frame_detect[0][j] - frame_detect[len(frame_detect) // 4 * 3][j]
            if temp > 0:
                flag = 0
            else:
                flag = 1
        # 确定递减还是递增
        for i in range(len(frame_detect) - 15):  # 最后20帧不进行平滑
            sum = 0
            if 3 <= i < len(frame_detect) - 3:  # 前窗口是有的。
                sum += frame_detect[i - 3][j] + frame_detect[i - 2][j] + frame_detect[i - 1][j]
                sum += frame_detect[i + 1][j] + frame_detect[i + 2][j] + frame_detect[i + 3][j] + frame_detect[i][j]
            else:
                if i == 1:
                    sum += frame_detect[0][j]
                    sum += frame_detect[i + 1][j] + frame_detect[i + 2][j] + frame_detect[i + 3][j] + frame_detect[i][j]
                elif i == 2:
                    sum += frame_detect[0][j] + frame_detect[1][j]
                    sum += frame_detect[i + 1][j] + frame_detect[i + 2][j] + frame_detect[i + 3][j] + frame_detect[i][j]
                elif i == 0:
                    sum += frame_detect[i + 1][j] + frame_detect[i + 2][j] + frame_detect[i + 3][j] + frame_detect[i][j]
                elif i == len(frame_detect) - 1:
                    sum += frame_detect[i - 1][j] + frame_detect[i - 2][j] + frame_detect[-3][j] + frame_detect[i][j]
                elif i == len(frame_detect) - 2:
                    sum += frame_detect[i - 1][j] + frame_detect[i - 2][j] + frame_detect[-3][j] + frame_detect[i][j] + \
                           frame_detect[i + 1][j]

                elif i == len(frame_detect) - 3:
                    sum += frame_detect[i - 1][j] + frame_detect[i - 2][j] + frame_detect[-3][j] + frame_detect[i][j] + \
                           frame_detect[i + 1][j] + frame_detect[i + 2][j]
                # 这边滑动窗口还是有问题啊。
            # 但是滑动之后，应该就不会特别的明显。
            # 然后就是求平均了
            avera = frame_detect[i][j]

            if 3 <= i < len(frame_detect) - 3:
                avera = sum // 7  # 求平均
            elif i == 1 or i == len(frame_detect) - 2:
                avera = sum // 5
            elif i == 0 or i == len(frame_detect) - 1:
                avera = sum // 4
            elif i == 2 or i == len(frame_detect) - 3:
                avera = sum // 6
            if i != 0:
                if flag == 1 and avera < frame_detect[i - 1][j]:  # 递增的情况下。
                    avera = frame_detect[i - 1][j]
                if flag == 0 and avera > frame_detect[i - 1][j]:
                    avera = frame_detect[i - 1][j]
            frame_detect[i][j] = avera

    return frame_detect


def predict_frame(object_frame):
    """
    假设对前两百帧进行一个预测
    :param object_frame:
    :return:
    """
    # object_frame=[[73721,887,612,13,15],[73771,881,608,12,17,],[73821,869,601,15,20],[73871,856,591,17,24],[73921,836,574,22,29],[73971,803,545,28,37],[74021,734,485,47,60],[74065,567,330,87,115],[74089,225,23,175,216]]
    frame_detct = [object_frame[0]]
    for i in range(1, len(object_frame)):
        x_list, y_list, h_list, w_list = [], [], [], []
        temp_len = object_frame[i][0] - object_frame[i - 1][0]
        x, y, h, w = object_frame[i][1] - object_frame[i - 1][1], object_frame[i][2] - object_frame[i - 1][2], \
                     object_frame[i][3] - object_frame[i - 1][3], object_frame[i][4] - object_frame[i - 1][4]
        x_len, y_len, h_len, w_len = x / temp_len, y / temp_len, h / temp_len, w / temp_len
        x_flag, y_flag, h_flag, w_flag = 0, 0, 0, 0
        flag = 1
        while flag < temp_len:
            x_list.append(object_frame[i - 1][1] + round(x_flag))
            y_list.append(object_frame[i - 1][2] + round(y_flag))
            h_list.append(object_frame[i - 1][3] + round(h_flag))
            w_list.append(object_frame[i - 1][4] + round(w_flag))
            x_flag += x_len
            y_flag += y_len
            w_flag += w_len
            h_flag += h_len
            '''
            if flag%x_len==0:
                if x_len>=0:
                    x_flag+=1
                else:
                    x_flag-=1
            if flag%y_len==0:
                if y_len>=0:
                    y_flag+=1
                else:
                    y_flag-=1
            if flag%h_len==0:
                if h_len>=0:
                    h_flag+=1
                else:
                    h_flag-=1
            if flag%w_len==0:
                if w_len>=0:
                    w_flag+=1
                else:
                    w_flag-=1
            '''
            flag += 1
        count = frame_detct[len(frame_detct) - 1][0]
        temp_len = len(x_list)
        for q in range(temp_len):
            frame_detct.append([count + 1, x_list.pop(0), y_list.pop(0), h_list.pop(0), w_list.pop(0)])
            count += 1
        frame_detct.append(object_frame[i])
    return frame_detct


def predict_frame_tobond(object_frame):
    """
    假设对前两百帧进行一个预测
    :param object_frame:
    :return:
    """
    # object_frame=[[73721,887,612,13,15],[73771,881,608,12,17,],[73821,869,601,15,20],[73871,856,591,17,24],[73921,836,574,22,29],[73971,803,545,28,37],[74021,734,485,47,60],[74065,567,330,87,115],[74089,225,23,175,216]]
    frame_detct = []
    frame_detct.append(object_frame[0])
    for i in range(1, len(object_frame)):
        x_list, y_list, h_list, w_list = [], [], [], []
        temp_len = object_frame[i][0] - object_frame[i - 1][0]
        x, y, h, w = object_frame[i][1] - object_frame[i - 1][1], object_frame[i][2] - object_frame[i - 1][2], \
                     object_frame[i][3] - object_frame[i - 1][3], object_frame[i][4] - object_frame[i - 1][4]
        x_len, y_len, h_len, w_len = x / temp_len, y / temp_len, h / temp_len, w / temp_len
        x_flag, y_flag, h_flag, w_flag = 0, 0, 0, 0
        flag = 1
        while flag < temp_len:
            x_list.append(object_frame[i - 1][1] + round(x_flag))
            y_list.append(object_frame[i - 1][2] + round(y_flag))
            h_list.append(object_frame[i - 1][3] + round(h_flag))
            w_list.append(object_frame[i - 1][4] + round(w_flag))
            x_flag += x_len
            y_flag += y_len
            w_flag += w_len
            h_flag += h_len
            '''
            if flag%x_len==0:
                if x_len>=0:
                    x_flag+=1
                else:
                    x_flag-=1
            if flag%y_len==0:
                if y_len>=0:
                    y_flag+=1
                else:
                    y_flag-=1
            if flag%h_len==0:
                if h_len>=0:
                    h_flag+=1
                else:
                    h_flag-=1
            if flag%w_len==0:
                if w_len>=0:
                    w_flag+=1
                else:
                    w_flag-=1
            '''
            flag += 1
        count = frame_detct[len(frame_detct) - 1][0]
        temp_len = len(x_list)
        for q in range(temp_len):
            frame_detct.append([count + 1, x_list.pop(0), y_list.pop(0), h_list.pop(0), w_list.pop(0)])
            count += 1
        frame_detct.append(object_frame[i])
    first_w = 13
    first_h = 15
    for i in range(0, len(frame_detct)):

        if frame_detct[i][3] == first_w:
            frame_detct[i][4] = first_h
        else:
            temp = frame_detct[i][3] - first_w
            first_h += temp
            first_w += temp
            frame_detct[i][4] = first_h
    return frame_detct


def half_yoloandhandwork(handwork, yolo_corrdinate):
    """
    一半手工一半yolo去实现
    handwork: 手工标注预测的那几帧
    yolo_corrdinate: yolo预测的所有桢
    :return:*
    """
    edge_px = 40  # 边缘像素点
    half_yolo_handwork = []
    #  yolo_corrdinate_smooth = yolo_frame_smooth(yolo_corrdinate, edge_px)
    yolo_corrdinate_smooth = yolo_corrdinate
    # 这里保存最终的结果。一预半是yolo，一半是手工标注的预测
    handwork_pro = predict_frame_tobond(handwork)  # 测的标记
    temp = 0
    for i in range(len(handwork_pro)):
        if handwork_pro[i][3] >= edge_px:  #
            temp = i
            break
        else:
            half_yolo_handwork.append(handwork_pro[i])
    if temp == 0:
        frame_count_edge = handwork_pro[len(handwork_pro) - 1][0]
    else:
        frame_count_edge = handwork_pro[temp][0]
    temp_yolo_count = 0
    if yolo_corrdinate_smooth[0][0] > frame_count_edge:
        while handwork[frame_count_edge][0] < yolo_corrdinate_smooth[0][0]:
            half_yolo_handwork.append(handwork_pro[frame_count_edge])
            frame_count_edge += 1
    else:
        for i in range(len(yolo_corrdinate_smooth)):
            if yolo_corrdinate_smooth[i][0] == frame_count_edge:
                temp_yolo_count = i
                break
    for j in range(temp_yolo_count, len(yolo_corrdinate_smooth)):
        half_yolo_handwork.append(yolo_corrdinate_smooth[j])
    #  print('temp=', temp, 'temp_yolo', temp_yolo_count)
    """
    for i in range(len(half_yolo_handwork)):
        print(half_yolo_handwork[i])
    """
    return half_yolo_handwork


def yolo_frame_smooth(yolo_detect, edge_px):
    """
    对yolo目标检测之后过滤的帧进行一个平滑处理
    :return:
    """
    # 首先基于宽度，将宽去做一个平滑处理
    # edge_px=40#边缘的像素点。
    temp_edge = 0
    for i in range(len(yolo_detect)):
        if yolo_detect[i][3] >= edge_px:
            temp_edge = i
            break
    # 从这个开始，基于宽度进行预测\
    # 当flag=0是递减的时候，当flag=1的时候，就是递增
    temp = yolo_detect[0][3] - yolo_detect[len(yolo_detect) // 2][3]
    if temp > 0:
        flag = 0
    elif temp < 0:
        flag = 1
    else:
        temp = yolo_detect[0][3] - yolo_detect[len(yolo_detect) // 4 * 3][3]
        if temp > 0:
            flag = 0
        else:
            flag = 1
    # 确定递减还是递增
    # 滑动窗口应该采取3个左右，不应该采取过多
    for i in range(temp_edge, len(yolo_detect) - 3):
        sum = 0
        if flag == 0:
            if yolo_detect[i - 1][3] - yolo_detect[i][3] < 0:
                # 出现不寻常的时候就开始滑动窗口
                # 默认前后都是正常的
                sum += yolo_detect[i - 1][3] + yolo_detect[i][3] + yolo_detect[i + 1][3]
            else:
                sum = yolo_detect[i][3] * 3
        else:
            if yolo_detect[i - 1][3] - yolo_detect[i][3] > 0:
                sum += yolo_detect[i - 1][3] + yolo_detect[i][3] + yolo_detect[i + 1][3]
            else:
                sum = yolo_detect[i][3] * 3
        avera = int(round(sum / 3))  # 使用四舍五入
        yolo_detect[i][3] = avera  # 对宽度进行修改之后，然后开始对高度修改
        # 按照长宽比来进行比较
    min_ratio = 1.26
    max_ratio = 1.32
    # 最大和最小的高宽比
    # 进行高度修改
    for i in range(temp_edge, len(yolo_detect) - 3):
        if yolo_detect[i][4] / yolo_detect[i][3] < min_ratio or yolo_detect[i][4] / yolo_detect[i][3] > max_ratio:
            # 假设这个不行的话，那就用宽度乘以1.3
            yolo_detect[i][4] = int(round(yolo_detect[i][3] * 1.3))

        # 使用滑动窗口
    return yolo_detect


def replace_frame_smooth():
    """
     使用窗口进行平滑处理
    :return:
    """
    frame_detect = []
    filepath = "corrdinate_judge.txt"
    # 已经将坐标存储在了txt文件中。将其从txt文件中获取之后，进行平滑处理
    for txt in open(filepath):
        all = txt.strip().split(";")
        frame = int(all[0])
        box = all[1].strip().split(",")
        box.pop(len(box) - 1)
        boxs = []
        for i in range(len(box)):
            boxs.append(int(box[i]))
        boxs.insert(0, frame)
        frame_detect.append(boxs)
    frame_detect = detect_smooth(frame_detect)
    return frame_detect


def get_corrdinate():
    """
    获取需要的坐标
    :return:
    """
    frame_detect = []
    filepath = "corrdinate_newvideo.txt"
    # 已经将坐标存储在了txt文件中。将其从txt文件中获取之后，进行平滑处理
    for txt in open(filepath):
        all = txt.strip().split(";")
        frame = int(all[0])
        box = all[1].strip().split(",")
        box.pop(len(box) - 1)
        boxs = []
        for i in range(len(box)):
            boxs.append(int(box[i]))
        boxs.insert(0, frame)
        frame_detect.append(boxs)
    #  frame_detect = detect_smooth(frame_detect)
    return frame_detect


def detect_takeoff():
    """
    主要的内容是将不是检测出来的信号灯做一个删减。
    :return:
    7/18    @gxl
    """
    object_frame = get_frame_singal()
    print(object_frame)
    '''
    with open("corrdinate_test.txt", "a") as file:
        file.write(str(frame_count) + ";")
        for i in range(len(frame)):
            file.write(str(frame[i]) + ",")
        file.write("\n")
    '''


def get_frame_countinus(frame_object):
    """
    获取所有的帧之后，进行一个帧的选择，将目前一个连续帧放进来。
    然后返回的是一个当前的若干帧以及总的帧。
    :return:
    7/19 @gxl
    """

    count = 0
    for i in range(len(frame_object) - 1):
        count += 1
        if frame_object[i + 1][0] - frame_object[i][0] > 5:
            break
    frame_now = []
    for j in range(count):
        frame_now.append(frame_object.pop(0))
    #  将这一段的获取到了之后，然后进行处理。首先手工标注的是其中的一部分。
    #  获取手工标注的信号灯。然后用前面写的算法，各个一半。然后返回的就是手工标注与算法识别的各个一半
    #  主程序那里需要改一下。


def yolo_detect(start, end):
    """
    主要是利用yolov3进行探测。在指定的帧内。
    :param start:  开始的帧
    :param end: 结束的帧
    :return:
    """

    video_path = 'D:/衡阳到长沙/衡阳-岳阳.mp4'
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    net = findnet()
    ln, out = finln_out(net)
    while cap.isOpened():

        # cap.set(cv2.CAP_PROP_POS_FRAMES,6000)
        # print(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        print("i=", i)
        if ret:
            if end >= i >= start:
                boxes, conf = yolov3_detect(frame, net, ln, out)
                print(boxes, conf)
                if len(conf) > 0:

                    with open("corrdinate_newvideo724.txt", "a") as file:
                        file.write(str(i) + ";")
                        for p in range(len(boxes)):
                            for j in range(len(boxes[p])):
                                file.write(str(boxes[p][j]) + ",")
                        file.write(";")
                        for h in range(len(conf)):
                            file.write(str(conf[h]) + ",")
                        file.write("\n")
            else:
                cap.release()
                break
            i += 1
        else:
            break


def yolo_detect_pre():
    """
    # 这是一个工具类，主要是把前面的大概提取的信号灯的模板，需要去用yolov3而去检测哪里需要检测的。
    :return:
    7/21 @gxl
    """
    frame_detect = get_corrdinate()
    cv_read_count = []
    start_count = frame_detect[0][0]
    for i in range(1, len(frame_detect)):
        if frame_detect[i][0] - frame_detect[i - 1][0] > 5:
            break_count = frame_detect[i - 1][0]
            cv_read_count.append((start_count, break_count))
            start_count = frame_detect[i][0]
    cv_read_count.append((start_count, frame_detect[len(frame_detect) - 1][0]))
    #  videos_src_path = 'D:/衡阳到长沙/衡阳-岳阳.mp4'
    #  cap = cv2.VideoCapture(videos_src_path)
    for i in range(len(cv_read_count)):
        start, end = cv_read_count[i]
        if start > 5000:
            yolo_detect(start, end)


config = configparser.ConfigParser()
config.read("train_config.ini", encoding='UTF-8')
csv_url = config.get('file-url', 'csv-url')


def video_lump_classification():
    """
    这个函数主要用于处理将有信号灯的那几帧视频进行保存。
    视频从30s开始进行出站，如果相差很大的距离的时候，就是代表着这个时刻是有信号灯的，那么向前数10s，都视为有信号灯，将其进行保存。
    :return:
    """
    videos_src_path = 'D:/衡阳到长沙/衡阳-岳阳.mp4'
    timer = []  # 系统时间
    real_distance = []  # 相对距离
    speed_hours = []  # 时速
    frame_distance = []  # 这里是最后返回的，每帧多少米。存储的是每秒的距离
    hour_speed = []  # 在每秒的时候的时速。
    with open(csv_url, mode='r') as f:
        data = csv.reader(f)
        for row in data:
            timer.append(row[2])
            real_distance.append(row[5])
            speed_hours.append(row[8])
    start_time = datetime.strptime(str(timer[1]), '%H:%M:%S')  # 起始时间。
    for i in range(2, len(real_distance)):
        if real_distance[i] != '' and real_distance[i - 1] != '':
            if int(real_distance[i]) > int(real_distance[i - 1]) and abs(
                    int(real_distance[i - 1]) - int(real_distance[i])) > 50 and i > 2158:
                cap = cv2.VideoCapture(videos_src_path)  # 读取视频
                end_time = datetime.strptime(str(timer[i]), '%H:%M:%S')  # 这一刻的结束时间
                s = end_time - start_time
                second = s.seconds
                # 然后通过计算时间。
                frame_start = ((second + 30) - 18) * 50  # 这是跳
                frame_end = frame_start + 1200
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                # 视频的宽度
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # 视频的高度
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # 视频的帧率  视频的编码  定义视频输出
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')

                out = cv2.VideoWriter(str(i) + "out.avi", fourcc, fps, (width, height))

                #  out = cv2.VideoWriter(str(i) + 'output.avi', fourcc, 50.0, (1920, 1080))
                while frame_start < frame_end:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        #  frame = cv2.flip(frame, 0)
                        #  cv2.imshow('frame', frame)
                        #  cv2.waitKey(0)
                        out.write(frame)
                        if frame_start >= frame_end:
                            cap.release()
                            out.release()
                            print('END')
                            break
                        frame_start += 1
                        print(frame_start)


def class_videofile():
    """
    这个函数主要是对视频进行分类，用系统文件进行操作。
    :return:
    """
    # 首先获取该目录下的所有视频文件名字。
    video_name = os.listdir('D:/衡阳到长沙/video_part')
    print(video_name)


def three_lump_replace():
    """
    想法是，首先都是需要手工标记的。
    那么在手工标记的时候就告诉是什么灯。然后在根据一半手工，一半是yolo标记的进行替换。

    :return:
    8/3 @gxl
    """


def xml_handwork_corrodiance():
    """
    获取xml中的坐标。然后进行保存。
    首先first存储第一个灯
    second存储第二个灯
    three存储第三个灯

    :return:
    8/3 @gxl
    """
    file_path = "E:/labelhandwork/"
    label_name = os.listdir(file_path)
    xml_value = []
    for i in range(len(label_name)):
        singal_label_name = file_path + label_name[i]
        frame_count = label_name[i].strip().split('.')[0]
        frame_count = int(frame_count.strip().split('_')[1])
        dom = xml.dom.minidom.parse(singal_label_name)
        root = dom.documentElement
        object_name = root.getElementsByTagName('object')
        classid = []
        cord = []
        for j in range(len(object_name)):
            classid.append(object_name[j].getElementsByTagName('name')[0].firstChild.data)
            bndbox = object_name[j].getElementsByTagName('bndbox')[0]
            xmin = int(bndbox.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(bndbox.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(bndbox.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(bndbox.getElementsByTagName('ymax')[0].firstChild.data)
            cord.append((xmin, ymin, abs(xmax - xmin), abs(ymax - ymin)))
        xml_value.append((frame_count, classid, cord))
    #  print(xml_value)
    xml_value = sorted(xml_value)
    first_xml_value = []
    seconde_xml_value = []
    three_xml_value = []
    for i in range(len(xml_value)):
        frame_count, classid, cordinate = xml_value[i]
        for j in range(len(classid)):
            x, y, p, q = cordinate[j]
            temp = [frame_count, x, y, p, q]
            if j == 0:
                first_xml_value.append(temp)
            elif j == 1:
                seconde_xml_value.append(temp)
            else:
                three_xml_value.append(temp)
    xml_value_all = [first_xml_value, seconde_xml_value, three_xml_value]
    return xml_value_all


def get_yolodetect_tostore():
    """
    获取三个灯的yolo检测，然后进行寻找到合适的值，进行储存。
    :return:
    8/4 @gxl
    """
    first_boxs = []
    second_boxs = []
    file_path = "txt_file/yolo_83.txt"
    for txt in open(file_path):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        first_box = []
        second_box = []
        for j in range(8):
            if j < 4:
                first_box.append(int(box[j]))
            else:
                second_box.append(int(box[j]))
        first_box.insert(0, frame)
        second_box.insert(0, frame)
        first_boxs.append(first_box)
        second_boxs.append(second_box)
    video_path = "E:/衡阳到长沙/video_part/2247out.avi"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_boxs[0][0])
    while len(first_boxs) >= 1:
        ret, frame = cap.read()
        if ret:
            first_box = first_boxs.pop(0)
            frame_count = first_box.pop(0)
            second_box = second_boxs.pop(0)
            second_box.pop(0)
            first_corrdinate = findcenter(frame, first_box)
            second_corrdinate = findcenter(frame, second_box)
            with open("txt_file/corrdinate84.txt", "a") as file:
                file.write(str(frame_count) + ";")
                for i in range(0, len(first_corrdinate)):
                    file.write(str(first_corrdinate[i]) + ",")
                for i in range(0, len(first_corrdinate)):
                    file.write(str(second_corrdinate[i]) + ",")
                file.write("\n")
    cap.release()


def get_doublelump_corrdinate():
    """
    获取已经存储找到边界的两种灯的corrdinate
    :return:
    8/4 @gxl
    """
    file_path = "txt_file/corrdinate84.txt"
    first_boxs = []
    second_boxs = []
    for txt in open(file_path):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        first_box = []
        second_box = []
        for j in range(8):
            if j < 4:
                first_box.append(int(box[j]))
            else:
                second_box.append(int(box[j]))
        first_box.insert(0, frame)
        second_box.insert(0, frame)
        first_boxs.append(first_box)
        second_boxs.append(second_box)
    all_boxs = [first_boxs, second_boxs]
    return all_boxs


def two_path_yolo_handwork_together():
    yolo_boxs = get_doublelump_corrdinate()
    xml_handwork = xml_handwork_corrodiance()
    handwork_first_boxs = predict_frame_tobond(xml_handwork[0])
    handwork_second_boxs = predict_frame_tobond(xml_handwork[1])
    first_half_yolo_handwork = half_yoloandhandwork(handwork_first_boxs, yolo_boxs[0])
    second_half_yolo_handwork = half_yoloandhandwork(handwork_second_boxs, yolo_boxs[1])
    """
    for i in range(len(first_half_yolo_handwork)):
        print(first_half_yolo_handwork[i])
    for j in range(len(second_half_yolo_handwork)):
        print(second_half_yolo_handwork[j])
    """
    return [first_half_yolo_handwork, second_half_yolo_handwork]

if __name__ == '__main__':
    two_path_yolo_handwork_together()
    '''
    frame_detect = [[73721, 887, 612, 13, 15], [73771, 881, 608, 12, 17, ], [73821, 869, 601, 15, 20],
                    [73871, 856, 591, 17, 24], [73921, 836, 574, 22, 29]]
    # frame_detect=predict_frame()
    # for i in range(len(frame_detect)):
    #   print(frame_detect[i])
    detect_yolo = replace_frame_smooth()
    frame_detect = half_yoloandhandwork(frame_detect, detect_yolo)
    for i in range(len(frame_detect)):
        print(frame_detect[i])
    '''
    #  detect_takeoff()
    #  video_lump_classification()
    """
    xml_handwork = xml_handwork_corrodiance()
    for i in range(len(xml_handwork[0])):
        print(xml_handwork[0][i])
    xml = predict_frame_tobond(xml_handwork[0])
    for j in range(len(xml)):
        print(xml[j])
    #  print(predict_frame_tobond(xml_handwork[0]))
    """
    '''
    imagepath = 'D:/train_photo_all/_30747.jpg'
    image = cv2.imread(imagepath)
    cv2.rectangle(image, (570, 304), (570 + 59, 304 + 92), (0, 255, 0), 3)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    '''
    #  yolo_detect_pre()
    '''
    videos_src_path = 'D:/衡阳到长沙/衡阳-岳阳.mp4'
    cap = cv2.VideoCapture(videos_src_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 61903)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow("image3",frame)
        cv2.waitKey(0)
        boxes = [572, 340, 62, 86]
        corrdinate = findcenter_test(frame, boxes)
        print(corrdinate)
        frame = replace_image(frame, corrdinate)
        cv2.imshow('image', frame)
        cv2.waitKey(0)
        '''
