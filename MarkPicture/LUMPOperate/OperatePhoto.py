"""
用来操作图片坐标。比如预测，等。
"""
import os
import numpy as np
import cv2
import yolov4_detect
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import linear_model
from LUMPOperate import PhotoReplace
from LUMPOperate.PhotoReplace import findcenter
from opencvyolo_0502 import yolov3_detect, finln_out, findnet


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
            flag += 1
        count = frame_detct[len(frame_detct) - 1][0]
        temp_len = len(x_list)
        for q in range(temp_len):
            frame_detct.append([count + 1, x_list.pop(0), y_list.pop(0), h_list.pop(0), w_list.pop(0)])
            count += 1
        frame_detct.append(object_frame[i])
    """
    这里是做了一个长宽比的绑定。
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
    """
    return frame_detct


def predict_frame_tobond_doublelump(object_frame):
    """
    两个信号灯的时候，进行一个预测。
    :param object_frame:
    :return:
    """
    frame_detct = []
    frame_detct.append(object_frame[0])
    for i in range(1, len(object_frame)):
        x_list, y_list, h_list, w_list = [], [], [], []
        x_list1, y_list1, h_list1, w_list1 = [], [], [], []
        temp_len = object_frame[i][0] - object_frame[i - 1][0]
        x, y, h, w = object_frame[i][1] - object_frame[i - 1][1], object_frame[i][2] - object_frame[i - 1][2], \
                     object_frame[i][3] - object_frame[i - 1][3], object_frame[i][4] - object_frame[i - 1][4]
        x1, y1, h1, w1 = object_frame[i][5] - object_frame[i - 1][5], object_frame[i][6] - object_frame[i - 1][6], \
                         object_frame[i][7] - object_frame[i - 1][7], object_frame[i][8] - object_frame[i - 1][8]
        x_len, y_len, h_len, w_len = x / temp_len, y / temp_len, h / temp_len, w / temp_len
        x_len1, y_len1, h_len1, w_len1 = x1 / temp_len, y1 / temp_len, h1 / temp_len, w1 / temp_len
        x_flag, y_flag, h_flag, w_flag = 0, 0, 0, 0
        x_flag1, y_flag1, h_flag1, w_flag1 = 0, 0, 0, 0
        flag = 1
        while flag < temp_len:
            x_list.append(object_frame[i - 1][1] + round(x_flag))
            y_list.append(object_frame[i - 1][2] + round(y_flag))
            h_list.append(object_frame[i - 1][3] + round(h_flag))
            w_list.append(object_frame[i - 1][4] + round(w_flag))
            x_list1.append(object_frame[i - 1][5] + round(x_flag1))
            y_list1.append(object_frame[i - 1][6] + round(y_flag1))
            h_list1.append(object_frame[i - 1][7] + round(h_flag1))
            w_list1.append(object_frame[i - 1][8] + round(w_flag1))
            x_flag += x_len
            y_flag += y_len
            w_flag += w_len
            h_flag += h_len
            x_flag1 += x_len1
            y_flag1 += y_len1
            w_flag1 += w_len1
            h_flag1 += h_len1
            flag += 1
        count = frame_detct[len(frame_detct) - 1][0]
        temp_len = len(x_list)
        for q in range(temp_len):
            frame_detct.append([count + 1, x_list.pop(0), y_list.pop(0), h_list.pop(0), w_list.pop(0),
                                x_list1.pop(0), y_list1.pop(0), h_list1.pop(0), w_list1.pop(0)])
            count += 1
        frame_detct.append(object_frame[i])
    print(frame_detct)
    return frame_detct


def half_yoloandhandwork1(handwork, yolo_corrdinate):
    """
    思路：在25到35之间找到比较合适的间隔比较小的帧数。
    然后进行替换。
    :param handwork:
    :param yolo_corrdinate:
    :return:
    @gxl 9/19
    """
    edge_px_before = 25  # 边缘像素点
    edge_px_after = 35
    half_yolo_handwork = []
    #  yolo_corrdinate_smooth = yolo_frame_smooth(yolo_corrdinate, edge_px)
    yolo_corrdinate_smooth = yolo_corrdinate
    # 这里保存最终的结果。一预半是yolo，一半是手工标注的预测
    #  handwork_pro = predict_frame_tobond(handwork)  # 测的标记
    handwork_pro = handwork  # 测的标记
    handedge_before = 0
    handedge_after = 0
    flag = 1
    for i in range(len(handwork_pro)):
        if handwork_pro[i][3] > edge_px_before:  #
            if flag == 1:
                handedge_before = i
                flag = 0
            if handwork_pro[i][3] == edge_px_after:
                handedge_after = i
                break
        else:
            half_yolo_handwork.append(handwork_pro[i])
    yolo_edge = 0
    for i in range(0, len(yolo_corrdinate)):
        if yolo_corrdinate[i][0] == handwork_pro[handedge_before][0]:
            yolo_edge = i
            break
    min_px = 10000  # 最小的间隔。
    min_px_pos = handedge_before
    i = handedge_before
    yolo_edge_after = yolo_edge
    while i <= handedge_after:
        sum = 0
        for j in range(1, len(handwork_pro[i])):
            sum += abs(handwork_pro[i][j] - yolo_corrdinate[yolo_edge][j])
        if min_px >= sum:
            min_px = sum
            min_px_pos = i
        yolo_edge += 1
        i += 1
    print('==', handwork_pro[i])
    print('++', yolo_corrdinate[yolo_edge])
    for h in range(handedge_before, min_px_pos + 1):
        half_yolo_handwork.append(handwork_pro[h])
        yolo_edge_after += 1
    for h in range(yolo_edge_after, len(yolo_corrdinate)):
        half_yolo_handwork.append(yolo_corrdinate[h])
    return half_yolo_handwork


def half_yoloandhandwork_doublelump(handwork, yolo_corrdinate):
    """
    #这是两个灯的情况下。 手工与yolo进行合并。
    思路：在25到35之间找到比较合适的间隔比较小的帧数。
    然后进行替换。
    :param handwork:
    :param yolo_corrdinate:
    :return:
    @gxl 9/19
    """
    edge_px_before = 25  # 边缘像素点
    edge_px_after = 35
    half_yolo_handwork = []
    #  yolo_corrdinate_smooth = yolo_frame_smooth(yolo_corrdinate, edge_px)
    yolo_corrdinate_smooth = yolo_corrdinate
    # 这里保存最终的结果。一预半是yolo，一半是手工标注的预测
    #  handwork_pro = predict_frame_tobond(handwork)  # 测的标记
    handwork_pro = handwork  # 测的标记
    handedge_before = 0
    handedge_after = 0
    # 这是第二个灯的时候。
    handedge_after_doublelump = 0
    handedge_before_doublelump = 0
    flag = 1
    flag_doublelump = 1
    # 这是第一个灯的时候。
    for i in range(len(handwork_pro)):
        if handwork_pro[i][3] > edge_px_before:  #
            if flag == 1:
                handedge_before = i
                flag = 0
            if handwork_pro[i][3] == edge_px_after:
                handedge_after = i
                break
        else:
            half_yolo_handwork.append(handwork_pro[i])
    yolo_edge = 0
    for i in range(0, len(yolo_corrdinate)):
        if yolo_corrdinate[i][0] == handwork_pro[handedge_before][0]:
            yolo_edge = i
            break
    min_px = 10000  # 最小的间隔。
    min_px_pos = handedge_before
    i = handedge_before
    yolo_edge_after = yolo_edge
    while i <= handedge_after:
        sum = 0
        for j in range(1, len(handwork_pro[i])):
            sum += abs(handwork_pro[i][j] - yolo_corrdinate[yolo_edge][j])
        if min_px >= sum:
            min_px = sum
            min_px_pos = i
        yolo_edge += 1
        i += 1
    print('==', handwork_pro[i])
    print('++', yolo_corrdinate[yolo_edge])
    for h in range(handedge_before, min_px_pos + 1):
        half_yolo_handwork.append(handwork_pro[h])
        yolo_edge_after += 1
    for h in range(yolo_edge_after, len(yolo_corrdinate)):
        half_yolo_handwork.append(yolo_corrdinate[h])
    return half_yolo_handwork


def half_yoloandhandwork(handwork, yolo_corrdinate):
    """
    一半手工一半yolo去实现
    handwork: 手工标注预测的那几帧
    yolo_corrdinate: yolo预测的所有桢
    :return:*
    需要在yolo与手工处做一个平滑，这样才能够不显得比较大的抖动。
    @gxl  9/9
    """
    edge_px = 25  # 边缘像素点
    half_yolo_handwork = []
    #  yolo_corrdinate_smooth = yolo_frame_smooth(yolo_corrdinate, edge_px)
    yolo_corrdinate_smooth = yolo_corrdinate
    # 这里保存最终的结果。一预半是yolo，一半是手工标注的预测
    #  handwork_pro = predict_frame_tobond(handwork)  # 测的标记
    handwork_pro = handwork  # 测的标记
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
    # 在这里做一个平滑过渡。9/9，做个10帧的平滑过渡。
    #  前5帧的话，一半预测，一半yolo，后5帧1/4预测，3/4的yolo
    for j in range(temp_yolo_count, len(yolo_corrdinate_smooth)):
        # if j - temp_yolo_count < 5:
        #     for h in range(1, len(yolo_corrdinate_smooth[j])):
        #         yolo_corrdinate_smooth[j][h] = int((handwork_pro[temp][h] + yolo_corrdinate_smooth[j][h]) / 2)
        #         temp += 1
        """
        elif 5 <= j - temp_yolo_count < 10:
            for h in range(1, len(yolo_corrdinate_smooth[j])):
                yolo_corrdinate_smooth[j][h] = int(handwork_pro[temp][h]/4 + (yolo_corrdinate_smooth[j][h]/4)*3)
                temp += 1
        """
        half_yolo_handwork.append(yolo_corrdinate_smooth[j])
    #  print('temp=', temp, 'temp_yolo', temp_yolo_count)
    """
    for i in range(len(half_yolo_handwork)):
        print(half_yolo_handwork[i])
    """
    return half_yolo_handwork


def yolo_v4_detect(url, frame_count):
    """
    利用的是最新的yolov4进行探测。在指定的帧内。
    :param url:
    :param frame_count:
    :return:
    """
    cap = cv2.VideoCapture(url)
    #  cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = frame_count
    #  cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    # net = findnet()
    # ln, out = finln_out(net)
    while cap.isOpened():

        # cap.set(cv2.CAP_PROP_POS_FRAMES,6000)
        # print(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        print("i=", i)
        if ret:

            # boxes, conf = yolov3_detect(frame, net, ln, out)
            boxes, conf = yolov4_detect.yolov4_detect(frame)
            print(boxes, conf)
            if len(conf) > 0:

                with open("txt/yololump.txt", "a") as file:
                    file.write(str(i) + ";")
                    for p in range(len(boxes)):
                        for j in range(len(boxes[p])):
                            file.write(str(boxes[p][j]) + ",")
                    file.write(";")
                    for h in range(len(conf)):
                        file.write(str(conf[h]) + ",")
                    file.write("\n")
            """
            else:
                cap.release()
                break
            """
            i += 1
        else:
            break


def yolo_detect(url, frame_count):
    """
    主要是利用yolov3进行探测。在指定的帧内。
    :param start:  开始的帧
    :param end: 结束的帧
    :return:
    """

    # video_path = 'E:/衡阳到长沙/衡阳-岳阳.mp4'
    cap = cv2.VideoCapture(url)
    #  cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = frame_count
    #  cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    net = findnet()
    ln, out = finln_out(net)
    while cap.isOpened():

        # cap.set(cv2.CAP_PROP_POS_FRAMES,6000)
        # print(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        print("i=", i)
        if ret:

            boxes, conf = yolov3_detect(frame, net, ln, out)
            print(boxes, conf)
            if len(conf) > 0:

                with open("txt/yololump.txt", "a") as file:
                    file.write(str(i) + ";")
                    for p in range(len(boxes)):
                        for j in range(len(boxes[p])):
                            file.write(str(boxes[p][j]) + ",")
                    file.write(";")
                    for h in range(len(conf)):
                        file.write(str(conf[h]) + ",")
                    file.write("\n")
            """
            else:
                cap.release()
                break
            """
            i += 1
        else:
            break


def yolo_detct_to_smooth(ratio, video_url, frame_count):
    """
    对yolo探测出来的结果进行一个平滑处理。以及寻找到中心点。好进行替换操作。
    需要对这里进行改造一下。video_url 进行写死操作。
    :return:
    """
    yolo_detect = []
    url = 'txt/yololump.txt'
    for txt in open(url):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        yolo_detect.append((frame, (box[0], box[1], box[2], box[3])))
    # 在这里
    # yolo_detect = get_pos_for_txt(url)
    # flag = 1
    # while flag < len(yolo_detect):
    #     temp_len = yolo_detect[flag][0] - yolo_detect[flag - 1][0]
    #     if temp_len > 1:
    #         temp_frame = [yolo_detect[flag - 1][0] + 1, yolo_detect[flag - 1][1], yolo_detect[flag - 1][2],
    #                       yolo_detect[flag - 1][3], yolo_detect[flag - 1][4]]
    #         yolo_detect.insert(flag, temp_frame)
    #     else:
    #         flag += 1

    #  获取到了之后，进行探测处理。
    cap = cv2.VideoCapture(video_url)
    order_frame1 = int(yolo_detect[0][0]) - int(frame_count)

    order_frame = int(yolo_detect[0][0])
    """
    以下是测试部分。
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(order_frame1))

    # while 181850 + 680 != yolo_detect[0][0]:
    #     yolo_detect.pop(0)
    # order_frame = 181850 + 680
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 680)
    # frame_count1 = 680
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(yolo_detect) > 0 and order_frame == yolo_detect[0][0]:
                (singalump_frame, boxes) = yolo_detect.pop(0)  # 这表明获取到了这一帧的照片了
                frame = PhotoReplace.findcenter(frame, boxes)  # 在这里代表获取坐标
                if order_frame == 688:
                    print('688frame: ', frame)
                with open("txt/yolosmooth.txt", "a") as file:
                    file.write(str(order_frame) + ";")
                    for i in range(len(frame)):
                        file.write(str(frame[i]) + ",")
                    file.write("\n")
            order_frame += 1
        if len(yolo_detect) == 0:
            break
    cap.release()
    #  对yolosmooth 必须做一个连续的smooth。
    corrdinate = get_pos_for_txt("txt/yolosmooth.txt")

    #  将yolosmooth内容进行清除。
    #  将得到的数据进行最小二乘法进行拟合。
    intercal = 30
    open('txt/yolosmooth.txt', 'w').close()
    edge = 0
    for i1 in range(len(corrdinate)):
        if corrdinate[i1][3] > 40:
            edge = i1
            break
    for h in range((len(corrdinate) - edge) // intercal):
        temp = edge + h * intercal
        frame_int = []
        frame_nointerval = []
        x = []
        y = []
        w = []
        h = []
        for j in range(temp, temp + intercal, 2):
            frame_int.append(int(corrdinate[j][0]) - int(frame_count) + 1)
            frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 1)
            frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 2)
            x.append(int(corrdinate[j][1]))
            y.append(int(corrdinate[j][2]))
            w.append(int(corrdinate[j][3]))
            h.append(int(corrdinate[j][4]))
        x1 = clf(frame_int, x, frame_nointerval)
        y1 = clf(frame_int, y, frame_nointerval)
        w1 = clf(frame_int, w, frame_nointerval)
        h1 = clf(frame_int, h, frame_nointerval)
        p = 0
        print(x, x1)
        print(y, y1)
        print(w, w1)
        print(h, h1)
        for j in range(temp, temp + intercal):
            corrdinate[j][1] = int(round(x1[p]))
            corrdinate[j][2] = round(y1[p])
            corrdinate[j][3] = round(w1[p])
            corrdinate[j][4] = round(h1[p])
            p += 1
    x = []
    y = []
    w = []
    h = []
    frame_int = []
    frame_nointerval = []
    for j in range(len(corrdinate) - intercal, len(corrdinate), 3):
        frame_int.append(int(corrdinate[j][0]) - int(frame_count) + 1)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 1)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 2)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 3)
        x.append(int(corrdinate[j][1]))
        y.append(int(corrdinate[j][2]))
        w.append(int(corrdinate[j][3]))
        h.append(int(corrdinate[j][4]))
    x1 = clf(frame_int, x, frame_nointerval)
    y1 = clf(frame_int, y, frame_nointerval)
    w1 = clf(frame_int, w, frame_nointerval)
    h1 = clf(frame_int, h, frame_nointerval)
    p = 0
    for j in range(len(corrdinate) - intercal, len(corrdinate)):
        corrdinate[j][1] = int(round(x1[p]))
        corrdinate[j][2] = round(y1[p])
        corrdinate[j][3] = round(w1[p])
        corrdinate[j][4] = round(h1[p])
        p += 1
    store_pos_txt('txt/nosmooth.txt', corrdinate)
    # corr = yolo_frame_smooth(corrdinate, 40, ratio)
    store_pos_txt('txt/yolosmooth.txt', corrdinate)


def yolo_detct_to_smooth1(ratio, video_url, frame_count):
    """
    对yolo探测出来的结果进行一个平滑处理。以及寻找到中心点。好进行替换操作。
    需要对这里进行改造一下。video_url 进行写死操作。
    :return:
    """
    yolo_detect = []
    url = 'txt/yololump.txt'
    for txt in open(url):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        yolo_detect.append((frame, (box[0], box[1], box[2], box[3])))
    #  获取到了之后，进行探测处理。
    cap = cv2.VideoCapture(video_url)
    order_frame1 = int(yolo_detect[0][0]) - int(frame_count)

    order_frame = int(yolo_detect[0][0])
    """
    以下是测试部分。
    """
    # cap.set(cv2.CAP_PROP_POS_FRAMES, int(order_frame1))
    temp_frame = 657
    while 181850 + temp_frame != yolo_detect[0][0]:
        yolo_detect.pop(0)
    order_frame = 181850 + temp_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, temp_frame)
    frame_count1 = temp_frame
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(yolo_detect) > 0 and order_frame == yolo_detect[0][0]:
                (singalump_frame, boxes) = yolo_detect.pop(0)  # 这表明获取到了这一帧的照片了
                frame = PhotoReplace.findcenter1(frame, boxes, frame_count1)  # 在这里代表获取坐标
                if frame_count1 == 675:
                    print('675frame:', frame)
                    # break
                with open("txt/yolosmooth.txt", "a") as file:
                    file.write(str(order_frame) + ";")
                    for i in range(len(frame)):
                        file.write(str(frame[i]) + ",")
                    file.write("\n")
            order_frame += 1
            frame_count1 += 1
        if len(yolo_detect) == 0:
            break
    cap.release()


def yolo_detect_to_smooth_doublelump1120(ratio, video_url, frame_count):
    """
    两个灯的平滑。
    主要是检测两个灯的平滑，以及寻找到中心点，好进行替换操作。
    :param ratio:
    :param video_url:
    :param frame_count:
    :return:
    """
    yolo_detect = []
    lump_predict = get_pos_for_txt_doublelump('txt/lumppospredicate.txt')  # 获取预测的信号灯。
    url = 'txt/yololump.txt'
    lessBox = 0
    for txt in open(url):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        box1 = []
        box2 = []
        for i in range(len(box)):
            if i < 4:
                box1.append(int(box[i]))
            if 4 <= i < 8:
                box2.append(int(box[i]))
        if len(box2) != 4:
            # if flag_first == 1:  # 这样最起码保证了有8个pos
            lessBox += 1
            # 这里计算的是有几个不足的pos
        yolo_detect.append([frame, box1, box2])
    # 应该从预测开始计算。
    # 这里计算的是，假设有一些是没有box2的。那么就用预测的来使用。
    yolo_detect_final = []

    predict_frame = yolo_detect[0][0]  # 寻找预测的帧
    predict_pos = 0
    for i in range(len(yolo_detect)):
        frame_i, box_i1, box_i2 = yolo_detect[i]
        flag_break = 0
        for h in range(len(box_i1)):
            if box_i1[h] < 0:
                flag_break = 1
                break
        for h in range(len(box_i2)):
            if box_i2[h] < 0:
                flag_break = 1
                break
        if flag_break == 1:
            break
        if len(box_i2) < 3:
            boxi3 = []
            if frame_i>333352:
                print(frame_i)
            while predict_frame != frame_i:
                predict_frame += 1
                predict_pos +=1
            if predict_frame > lump_predict[-1][0]:
                boxi3 = [box_i1[0], box_i1[1] + box_i1[3] + 21, box_i1[2], box_i1[3]]
            else:
                pre_box = lump_predict[predict_pos]
                if box_i1[3] < 40:
                    boxi3 = [pre_box[5], pre_box[6], pre_box[7], pre_box[8]]
                elif 40 <= box_i1[3] < 80:
                    boxi3 = [box_i1[0], ((box_i1[1] + box_i1[3]) + pre_box[6]) // 2, (box_i1[2] + pre_box[7]) // 2,
                             (box_i1[3] + pre_box[8]) // 2]
                else:
                    boxi3 = [box_i1[0], ((box_i1[1] + box_i1[3]) * 0.7 + pre_box[6] * 0.3),
                             (box_i1[2] + pre_box[7]) // 2,
                             (box_i1[3] + pre_box[8]) // 2]
            yolo_detect_final.append([frame_i, box_i1, boxi3])
        else:
            yolo_detect_final.append([frame_i, box_i1, box_i2])
    flag = 1
    while flag < len(yolo_detect_final):
        # print('+', yolo_detect_final[flag])
        # print('+', yolo_detect_final[flag - 1])
        temp_len = yolo_detect_final[flag][0] - yolo_detect_final[flag - 1][0]
        if temp_len > 1:
            temp_frame = [yolo_detect_final[flag - 1][0] + 1, yolo_detect_final[flag - 1][1],
                          yolo_detect_final[flag - 1][2]]
            yolo_detect_final.insert(flag, temp_frame)
        else:
            flag += 1
    cap = cv2.VideoCapture(video_url)
    order_frame1 = int(yolo_detect_final[0][0]) - int(frame_count)

    order_frame = int(yolo_detect_final[0][0])
    """
    以下是测试部分。
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(order_frame1))

    # while 181850 + 680 != yolo_detect[0][0]:
    #     yolo_detect.pop(0)
    # order_frame = 181850 + 680
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 680)
    # frame_count1 = 680
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(yolo_detect_final) > 0 and order_frame == yolo_detect_final[0][0]:
                (singalump_frame, boxes, boxes1) = yolo_detect_final.pop(0)  # 这表明获取到了这一帧的照片了
                if singalump_frame == 333401:
                    print(singalump_frame)
                frame1 = PhotoReplace.findcenter(frame, boxes)  # 在这里代表获取boxes坐标
                frame2 = PhotoReplace.findcenter(frame, boxes1)  # 在这里代表获取boxes1坐标
                with open("txt/yolosmooth.txt", "a") as file:
                    file.write(str(order_frame) + ";")
                    for i in range(len(frame1)):
                        file.write(str(frame1[i]) + ",")
                    for i in range(0, len(frame1)):
                        file.write(str(frame2[i]) + ",")
                    file.write("\n")
            order_frame += 1
        if len(yolo_detect_final) == 0:
            break
    cap.release()
    #  对yolosmooth 必须做一个连续的smooth。
    corrdinate = get_pos_for_txt_doublelump("txt/yolosmooth.txt")

    #  将yolosmooth内容进行清除。
    #  将得到的数据进行最小二乘法进行拟合。
    intercal = 30
    open('txt/yolosmooth.txt', 'w').close()
    edge = 0
    for i1 in range(len(corrdinate)):
        if corrdinate[i1][3] > 40:
            edge = i1
            break
    for h in range((len(corrdinate) - edge) // intercal):
        temp = edge + h * intercal
        frame_int = []
        frame_nointerval = []
        x = []
        y = []
        w = []
        h = []
        x1 = []
        y1 = []
        w1 = []
        h1 = []
        for j in range(temp, temp + intercal, 2):
            frame_int.append(int(corrdinate[j][0]) - int(frame_count) + 1)
            frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 1)
            frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 2)
            x.append(int(corrdinate[j][1]))
            y.append(int(corrdinate[j][2]))
            w.append(int(corrdinate[j][3]))
            h.append(int(corrdinate[j][4]))
            x1.append(int(corrdinate[j][5]))
            y1.append(int(corrdinate[j][6]))
            w1.append(int(corrdinate[j][7]))
            h1.append(int(corrdinate[j][8]))
        x2 = clf(frame_int, x, frame_nointerval)
        y2 = clf(frame_int, y, frame_nointerval)
        w2 = clf(frame_int, w, frame_nointerval)
        h2 = clf(frame_int, h, frame_nointerval)
        x3 = clf(frame_int, x1, frame_nointerval)
        y3 = clf(frame_int, y1, frame_nointerval)
        w3 = clf(frame_int, w1, frame_nointerval)
        h3 = clf(frame_int, h1, frame_nointerval)
        p = 0
        print(x, x2)
        print(y, y2)
        print(w, w2)
        print(h, h2)
        print(x1, x3)
        print(y1, y3)
        print(w1, w3)
        print(h1, h3)
        for j in range(temp, temp + intercal):
            corrdinate[j][1] = int(round(x2[p]))
            corrdinate[j][2] = round(y2[p])
            corrdinate[j][3] = round(w2[p])
            corrdinate[j][4] = round(h2[p])
            corrdinate[j][5] = int(round(x3[p]))
            corrdinate[j][6] = round(y3[p])
            corrdinate[j][7] = round(w3[p])
            corrdinate[j][8] = round(h3[p])
            p += 1
    x = []
    y = []
    w = []
    h = []
    x1 = []
    y1 = []
    w1 = []
    h1 = []
    frame_int = []
    frame_nointerval = []
    for j in range(len(corrdinate) - intercal, len(corrdinate), 3):
        frame_int.append(int(corrdinate[j][0]) - int(frame_count) + 1)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 1)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 2)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 3)
        x.append(int(corrdinate[j][1]))
        y.append(int(corrdinate[j][2]))
        w.append(int(corrdinate[j][3]))
        h.append(int(corrdinate[j][4]))
        x1.append(int(corrdinate[j][5]))
        y1.append(int(corrdinate[j][6]))
        w1.append(int(corrdinate[j][7]))
        h1.append(int(corrdinate[j][8]))
    x2 = clf(frame_int, x, frame_nointerval)
    y2 = clf(frame_int, y, frame_nointerval)
    w2 = clf(frame_int, w, frame_nointerval)
    h2 = clf(frame_int, h, frame_nointerval)
    x3 = clf(frame_int, x1, frame_nointerval)
    y3 = clf(frame_int, y1, frame_nointerval)
    w3 = clf(frame_int, w1, frame_nointerval)
    h3 = clf(frame_int, h1, frame_nointerval)
    p = 0
    for j in range(len(corrdinate) - intercal, len(corrdinate)):
        corrdinate[j][1] = int(round(x2[p]))
        corrdinate[j][2] = round(y2[p])
        corrdinate[j][3] = round(w2[p])
        corrdinate[j][4] = round(h2[p])
        corrdinate[j][5] = int(round(x3[p]))
        corrdinate[j][6] = round(y3[p])
        corrdinate[j][7] = round(w3[p])
        corrdinate[j][8] = round(h3[p])
        p += 1
    store_pos_txt('txt/nosmooth.txt', corrdinate)
    # corr = yolo_frame_smooth(corrdinate, 40, ratio)
    store_pos_txt('txt/yolosmooth.txt', corrdinate)


def yolo_detct_to_smooth_doublelump(ratio, video_url, frame_count):
    """
    #  特别注明： 这里是两个灯的平滑。
    对yolo探测出来的结果进行一个平滑处理。以及寻找到中心点。好进行替换操作。
    需要对这里进行改造一下。video_url 进行写死操作。
    :return:
    @gxl 9/27
    """
    yolo_detect = []
    lump_predict = get_pos_for_txt_doublelump('txt/lumppospredicate.txt')
    url = 'txt/yololump.txt'
    lessBox = 0
    for txt in open(url):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        box1 = []
        box2 = []
        for i in range(len(box)):
            if i < 4:
                box1.append(int(box[i]))
            if 4 <= i < 8:
                box2.append(int(box[i]))
        if len(box2) != 4:
            if frame == 333401:
                print(frame)
            # if flag_first == 1:  # 这样最起码保证了有8个pos
            lessBox += 1
            # flag_first = 1
            # for h in range(len(lump_predict)):
            #     temp_1 = lump_predict[h][0]
            #     temp_2 = lump_predict[h]
            #     if temp_1 == frame:
            #         box2 = [lump_predict[h][5], lump_predict[h][6], lump_predict[h][7], lump_predict[h][8]]
            #         flag_first = 0
            #         break
            # if flag_first == 1:
            #     framelast, box1last, box2last = yolo_detect[len(yolo_detect) - 1]
            #     # box2 = box2last
            #     box2 = [box2last[0], box1last[1] + box1last[3] + 5, box1last[2], box1last[3]]
            #
            #     # flag_first = 0
            # # else:
            # #     framelast, box1last, box2last = yolo_detect[len(yolo_detect) - 1]
            # #     box2 = box2last
        yolo_detect.append([frame, box1, box2])
    yolo_detect_final = []
    if lessBox / len(yolo_detect) > 0.3:
        for i in range(len(yolo_detect)):
            frame_i, box_i1, box_i2 = yolo_detect[i]
            flag_break = 0
            for h in range(len(box_i1)):
                if box_i1[h] < 0:
                    flag_break = 1
                    break
            for h in range(len(box_i2)):
                if box_i2[h] < 0:
                    flag_break = 1
                    break
            if flag_break == 1:
                break
            if len(box_i2) < 3:
                boxi3 = []
                if box_i1[3] < 40:
                    boxi3 = [box_i1[0], box_i1[1] + box_i1[3] + 5, box_i1[2], box_i1[3]]
                else:
                    boxi3 = [box_i1[0], box_i1[1] + box_i1[3] + 20, box_i1[2], box_i1[3]]
                yolo_detect_final.append([frame_i, box_i1, boxi3])
            else:
                yolo_detect_final.append([frame_i, box_i1, box_i2])
    else:
        for i in range(len(yolo_detect)):
            frame_i, box_i1, box_i2 = yolo_detect[i]
            flag_break = 0
            for h in range(len(box_i1)):
                if box_i1[h] < 0:
                    flag_break = 1
                    break
            for h in range(len(box_i2)):
                if box_i2[h] < 0:
                    flag_break = 1
                    break
            if flag_break == 1:
                break
            if len(box_i2) < 3:
                if i != 0:
                    framelast, box1last, box2last = yolo_detect_final[len(yolo_detect_final) - 1]
                    box_i2 = box2last
                else:
                    box_i2 = [box_i1[0], box_i1[1] + box_i1[3] + 5, box_i1[2], box_i1[3]]
            yolo_detect_final.append([frame_i, box_i1, box_i2])
    # yolo_detect = yolo_detect_final
    # 获取的应该是  frame，box1，box2
    flag = 1
    while flag < len(yolo_detect_final):
        print('+', yolo_detect_final[flag])
        print('+', yolo_detect_final[flag - 1])
        temp_len = yolo_detect_final[flag][0] - yolo_detect_final[flag - 1][0]
        if temp_len > 1:
            temp_frame = [yolo_detect_final[flag - 1][0] + 1, yolo_detect_final[flag - 1][1],
                          yolo_detect_final[flag - 1][2]]
            yolo_detect_final.insert(flag, temp_frame)
        else:
            flag += 1

    #  获取到了之后，进行探测处理。
    cap = cv2.VideoCapture(video_url)
    order_frame1 = int(yolo_detect_final[0][0]) - int(frame_count)

    order_frame = int(yolo_detect_final[0][0])
    """
    以下是测试部分。
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(order_frame1))

    # while 181850 + 680 != yolo_detect[0][0]:
    #     yolo_detect.pop(0)
    # order_frame = 181850 + 680
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 680)
    # frame_count1 = 680
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(yolo_detect_final) > 0 and order_frame == yolo_detect_final[0][0]:
                (singalump_frame, boxes, boxes1) = yolo_detect_final.pop(0)  # 这表明获取到了这一帧的照片了
                if singalump_frame == 333401:
                    print(singalump_frame)
                frame1 = PhotoReplace.findcenter(frame, boxes)  # 在这里代表获取boxes坐标
                frame2 = PhotoReplace.findcenter(frame, boxes1)  # 在这里代表获取boxes1坐标
                with open("txt/yolosmooth.txt", "a") as file:
                    file.write(str(order_frame) + ";")
                    for i in range(len(frame1)):
                        file.write(str(frame1[i]) + ",")
                    for i in range(0, len(frame1)):
                        file.write(str(frame2[i]) + ",")
                    file.write("\n")
            order_frame += 1
        if len(yolo_detect_final) == 0:
            break
    cap.release()
    #  对yolosmooth 必须做一个连续的smooth。
    corrdinate = get_pos_for_txt_doublelump("txt/yolosmooth.txt")

    #  将yolosmooth内容进行清除。
    #  将得到的数据进行最小二乘法进行拟合。
    intercal = 30
    open('txt/yolosmooth.txt', 'w').close()
    edge = 0
    for i1 in range(len(corrdinate)):
        if corrdinate[i1][3] > 40:
            edge = i1
            break
    for h in range((len(corrdinate) - edge) // intercal):
        temp = edge + h * intercal
        frame_int = []
        frame_nointerval = []
        x = []
        y = []
        w = []
        h = []
        x1 = []
        y1 = []
        w1 = []
        h1 = []
        for j in range(temp, temp + intercal, 2):
            frame_int.append(int(corrdinate[j][0]) - int(frame_count) + 1)
            frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 1)
            frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 2)
            x.append(int(corrdinate[j][1]))
            y.append(int(corrdinate[j][2]))
            w.append(int(corrdinate[j][3]))
            h.append(int(corrdinate[j][4]))
            x1.append(int(corrdinate[j][5]))
            y1.append(int(corrdinate[j][6]))
            w1.append(int(corrdinate[j][7]))
            h1.append(int(corrdinate[j][8]))
        x2 = clf(frame_int, x, frame_nointerval)
        y2 = clf(frame_int, y, frame_nointerval)
        w2 = clf(frame_int, w, frame_nointerval)
        h2 = clf(frame_int, h, frame_nointerval)
        x3 = clf(frame_int, x1, frame_nointerval)
        y3 = clf(frame_int, y1, frame_nointerval)
        w3 = clf(frame_int, w1, frame_nointerval)
        h3 = clf(frame_int, h1, frame_nointerval)
        p = 0
        print(x, x2)
        print(y, y2)
        print(w, w2)
        print(h, h2)
        print(x1, x3)
        print(y1, y3)
        print(w1, w3)
        print(h1, h3)
        for j in range(temp, temp + intercal):
            corrdinate[j][1] = int(round(x2[p]))
            corrdinate[j][2] = round(y2[p])
            corrdinate[j][3] = round(w2[p])
            corrdinate[j][4] = round(h2[p])
            corrdinate[j][5] = int(round(x3[p]))
            corrdinate[j][6] = round(y3[p])
            corrdinate[j][7] = round(w3[p])
            corrdinate[j][8] = round(h3[p])
            p += 1
    x = []
    y = []
    w = []
    h = []
    x1 = []
    y1 = []
    w1 = []
    h1 = []
    frame_int = []
    frame_nointerval = []
    for j in range(len(corrdinate) - intercal, len(corrdinate), 3):
        frame_int.append(int(corrdinate[j][0]) - int(frame_count) + 1)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 1)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 2)
        frame_nointerval.append(int(corrdinate[j][0]) - int(frame_count) + 3)
        x.append(int(corrdinate[j][1]))
        y.append(int(corrdinate[j][2]))
        w.append(int(corrdinate[j][3]))
        h.append(int(corrdinate[j][4]))
        x1.append(int(corrdinate[j][5]))
        y1.append(int(corrdinate[j][6]))
        w1.append(int(corrdinate[j][7]))
        h1.append(int(corrdinate[j][8]))
    x2 = clf(frame_int, x, frame_nointerval)
    y2 = clf(frame_int, y, frame_nointerval)
    w2 = clf(frame_int, w, frame_nointerval)
    h2 = clf(frame_int, h, frame_nointerval)
    x3 = clf(frame_int, x1, frame_nointerval)
    y3 = clf(frame_int, y1, frame_nointerval)
    w3 = clf(frame_int, w1, frame_nointerval)
    h3 = clf(frame_int, h1, frame_nointerval)
    p = 0
    for j in range(len(corrdinate) - intercal, len(corrdinate)):
        corrdinate[j][1] = int(round(x2[p]))
        corrdinate[j][2] = round(y2[p])
        corrdinate[j][3] = round(w2[p])
        corrdinate[j][4] = round(h2[p])
        corrdinate[j][5] = int(round(x3[p]))
        corrdinate[j][6] = round(y3[p])
        corrdinate[j][7] = round(w3[p])
        corrdinate[j][8] = round(h3[p])
        p += 1
    store_pos_txt('txt/nosmooth.txt', corrdinate)
    # corr = yolo_frame_smooth(corrdinate, 40, ratio)
    store_pos_txt('txt/yolosmooth.txt', corrdinate)


def get_yolo_and_hand_pos():
    """
    获取yolo 与手工标记的坐标进行合并。
    :return:
    """
    # 首先获取yolo的坐标。
    yolo_url = 'txt/yolosmooth.txt'
    hand_url = 'txt/lumppospredicate.txt'
    yolo_detect = get_pos_for_txt(yolo_url)
    hand_predict = get_pos_for_txt(hand_url)
    #  在获取了两个pos之后，进行合并获得新的pos。
    hand_and_yolo = half_yoloandhandwork1(hand_predict, yolo_detect)
    txt_url = 'txt/yoloandhand.txt'
    store_pos_txt(txt_url, hand_and_yolo)  # 进行储存


def get_yolo_and_hand_pos_doublelump():
    """
    获取yolo 与手工标记的坐标进行合并。
    :return:
    """
    # 首先获取yolo的坐标。
    yolo_url = 'txt/yolosmooth.txt'
    hand_url = 'txt/lumppospredicate.txt'
    yolo_detect = get_pos_for_txt_doublelump(yolo_url)
    hand_predict = get_pos_for_txt_doublelump(hand_url)
    #  在获取了两个pos之后，进行合并获得新的pos。
    hand_and_yolo = half_yoloandhandwork_doublelump(hand_predict, yolo_detect)
    txt_url = 'txt/yoloandhand.txt'
    store_pos_txt(txt_url, hand_and_yolo)  # 进行储存


def store_pos_txt(url, frame_pos):
    """
    无论是yolo还是预测的将其进行储存。url即为储存地址。
    frame_pos即为需要存储的pos。
    :param url:储存地址。
    :param frame_pos:为需要存储的pos
    :return:
    """
    for j in range(0, len(frame_pos)):
        with open(url, "a") as file:
            file.write(str(frame_pos[j][0]) + ";")
            for i in range(1, len(frame_pos[j])):
                file.write(str(frame_pos[j][i]) + ",")
            file.write("\n")


def get_pos_for_txt(url):
    """
    这个主要是从存储的url中获取需要的pos
    :param url:存储的url
    :return: 返回的是获取的pos
    """
    yolo_detect_pos = []
    for txt in open(url):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        yolo_detect_pos.append([frame, int(box[0]), int(box[1]), int(box[2]), int(box[3])])
    return yolo_detect_pos


def get_pos_for_txt_doublelump(url):
    """
    这个主要是从存储的url中获取需要的pos
    这是两个灯的时候。
    :param url:存储的url
    :return: 返回的是获取的pos
    """
    yolo_detect_pos = []
    for txt in open(url):
        all_yolo = txt.strip().split(";")
        frame = int(all_yolo[0])
        box = all_yolo[1].strip().split(",")
        box.pop(len(box) - 1)
        yolo_detect_pos.append([frame, int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3])),
                                int(float(box[4])), int(float(box[5])), int(float(box[6])), int(float(box[7]))])
    return yolo_detect_pos


def replace_video_to_store(videourl, frame_count):
    """
    将更改的视频进行保存到另一个地方。
    :param videourl: 视频的url
    :param frame_count: 视频在第几帧。
    :return:
    """
    #  首先获取的是已经合并的pos
    save_url = "E:/save_video/replaceVideo/"
    pos = get_pos_for_txt('txt/yoloandhand.txt')
    video_count = int(pos[0][0]) - frame_count + 1  # 实际的在该视频的帧数。
    cap = cv2.VideoCapture(videourl)  # 读取视频
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频的高度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频的帧率  视频的编码  定义视频输出
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #  存储应该是lujing+out+framecount.avi
    out = cv2.VideoWriter(save_url + "out" + str(frame_count) + ".avi", fourcc, fps, (width, height))
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(pos) > 0 and video_count == i:
                pos_i = pos.pop(0)
                if i == 650:
                    print(i)
                print(pos_i)
                frame = PhotoReplace.replace_image(i, frame, pos_i)  # 进行替换操作之后。
            out.write(frame)  # 将需要的每一帧进行储存。
            i += 1
            if len(pos) >= 1:
                video_count = int(pos[0][0]) - frame_count + 1  # 实际的在该视频的帧数。
        else:
            break
    cap.release()
    out.release()
    # 视频的宽度


def replace_video_to_store_doublelump(videourl, frame_count):
    """
    将更改的视频进行保存到另一个地方。
    :param videourl: 视频的url
    :param frame_count: 视频在第几帧。
    :return:
    """
    #  首先获取的是已经合并的pos
    save_url = "E:/save_video/replaceVideo/"
    pos = get_pos_for_txt_doublelump('txt/yoloandhand.txt')
    video_count = int(pos[0][0]) - frame_count + 1  # 实际的在该视频的帧数。
    cap = cv2.VideoCapture(videourl)  # 读取视频
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频的高度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频的帧率  视频的编码  定义视频输出
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #  存储应该是lujing+out+framecount.avi
    out = cv2.VideoWriter(save_url + "out" + str(frame_count) + ".avi", fourcc, fps, (width, height))
    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if len(pos) > 0 and video_count == i:
                pos_i = pos.pop(0)
                print(pos_i)
                if pos_i[0] == 287376:
                    print(pos_i)
                frame = PhotoReplace.replace_image_doublelump(i, frame, pos_i)  # 进行替换操作之后。
            out.write(frame)  # 将需要的每一帧进行储存。
            i += 1
            if len(pos) >= 1:
                video_count = int(pos[0][0]) - frame_count + 1  # 实际的在该视频的帧数。
        else:
            break
    cap.release()
    out.release()
    # 视频的宽度


def yolo_frame_smooth(yolo_detect, edge_px, ratio):
    """
    对yolo目标检测之后过滤的帧进行一个平滑处理
    :return:
    ratio: 从小到大的长宽比
    """
    # 首先基于宽度，将宽去做一个平滑处理
    # edge_px=40#边缘的像素点。
    temp_edge = 0
    for i in range(len(yolo_detect)):
        if yolo_detect[i][3] >= edge_px:
            temp_edge = i
            break
    # 从这个开始，基于宽度进行预测\
    min_ratio = ratio[1] - 0.2
    max_ratio = ratio[1] + 0.2
    # 最大和最小的高宽比
    # 进行高度修改

    # for i in range(temp_edge, len(yolo_detect)):
    #     if yolo_detect[i][3] > 90:
    #         if yolo_detect[i][4] / yolo_detect[i][3] < (ratio[1] - 0.02) or \
    #                 yolo_detect[i][4] / yolo_detect[i][3] > (ratio[1] + 0.02):
    #             yolo_detect[i][4] = int(round(yolo_detect[i][3] * ratio[1]))
    #     else:
    #
    #         if yolo_detect[i][4] / yolo_detect[i][3] < min_ratio or yolo_detect[i][4] / yolo_detect[i][3] > max_ratio:
    #             # 假设这个不行的话，那就用宽度乘以1.3
    #             # yolo_detect[i][4] = int(round(yolo_detect[i][3] * (max_ratio + min_ratio) / 2))
    #             yolo_detect[i][4] = int(round(yolo_detect[i][3] * ratio[1]))

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
    # for i in range(temp_edge, len(yolo_detect) - 3):
    i = temp_edge
    while i < len(yolo_detect) - 3:
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
    # min_ratio = ratio[1] - 0.2
    # max_ratio = ratio[1] + 0.2
    # # 最大和最小的高宽比
    # # 进行高度修改
    #
    # for i in range(temp_edge, len(yolo_detect) - 1):
    #     if yolo_detect[i][4] / yolo_detect[i][3] < min_ratio or yolo_detect[i][4] / yolo_detect[i][3] > max_ratio:
    #         # 假设这个不行的话，那就用宽度乘以1.3
    #         # yolo_detect[i][4] = int(round(yolo_detect[i][3] * (max_ratio + min_ratio) / 2))
    #         yolo_detect[i][4] = int(round(yolo_detect[i][3] * max_ratio))

    # 使用滑动窗口
    # yolo_detect1 = yolo_smooth_to_height(yolo_detect, ratio[1], temp_edge)
    # for i in range(temp_edge, len(yolo_detect1)):
    #     if yolo_detect1[i][3] > 90:
    #         if yolo_detect1[i][4] / yolo_detect1[i][3] < (ratio[1] - 0.02) or \
    #                 yolo_detect1[i][4] / yolo_detect1[i][3] > (ratio[1] + 0.02):
    #             yolo_detect1[i][4] = int(round(yolo_detect1[i][3] * ratio[1]))
    #     else:
    #
    #         if yolo_detect1[i][4] / yolo_detect1[i][3] < min_ratio or yolo_detect1[i][4] / yolo_detect1[i][3] > max_ratio:
    #             # 假设这个不行的话，那就用宽度乘以1.3
    #             # yolo_detect[i][4] = int(round(yolo_detect[i][3] * (max_ratio + min_ratio) / 2))
    #             yolo_detect1[i][4] = int(round(yolo_detect1[i][3] * ratio[1]))
    return yolo_detect


def yolo_smooth_to_width(yolo_detect, ratio, edge):
    """
    使用平滑处理。如果增加的应该是按照宽度进行比例的增加。
    这里主要是高度。
    :param yolo_detect:
    :return:
    @gxl 9/18
    """

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
    for i in range(edge, len(yolo_detect) - 1):
        temp = yolo_detect[i][3] - yolo_detect[i - 1][3]
        tempplus = yolo_detect[i + 1][3] - yolo_detect[i][3]
        tempintevral = yolo_detect[i + 1][3] - yolo_detect[i - 1][3]
        if flag == 0:  # 当递减的时候。
            if temp > 0:
                #  当不是递减的时候。
                yolo_detect[i][3] = yolo_detect[i - 1][3] - int(
                    round(abs(yolo_detect[i - 1][4] - yolo_detect[i][4]) / ratio))
            #  主要解决突然降低的情况。
            else:
                if tempplus > 0:
                    if tempintevral < 0:  # 间隔一个还是递减
                        #  在i这里是突然降低。
                        yolo_detect[i][3] = ((yolo_detect[i + 1][3] +
                                              int(round(abs(yolo_detect[i + 1][4] - yolo_detect[i][4]) / ratio)))
                                             + (yolo_detect[i - 1][3] -
                                                int(round(abs(yolo_detect[i - 1][4] - yolo_detect[i][4]) / ratio)))) \
                                            // 2


        else:
            temp = yolo_detect[i][3] - yolo_detect[i - 1][3]
            tempplus = yolo_detect[i + 1][3] - yolo_detect[i][3]
            tempintevral = yolo_detect[i + 1][3] - yolo_detect[i - 1][3]
            if temp < 0:
                #  当不是递增的时候。
                yolo_detect[i][3] = yolo_detect[i - 1][3] + int(
                    round(abs(yolo_detect[i - 1][4] - yolo_detect[i][4]) / ratio))
            #  主要解决突然降低的情况。
            else:
                if tempplus < 0:
                    if tempintevral > 0:  # 间隔一个还是递增
                        yolo_detect[i][3] = ((yolo_detect[i + 1][3] -
                                              int(round(abs(yolo_detect[i + 1][4] - yolo_detect[i][4]) / ratio)))
                                             + (yolo_detect[i - 1][3] +
                                                int(round(abs(yolo_detect[i - 1][4] - yolo_detect[i][4]) / ratio)))) \
                                            // 2
    return yolo_detect


def yolo_smooth_to_height(yolo_detect, ratio, edge):
    """
    使用平滑处理。如果增加的应该是按照宽度进行比例的增加。
    这里主要是高度。
    :param yolo_detect:
    :return:
    @gxl 9/18
    """

    # 当flag=0是递减的时候，当flag=1的时候，就是递增
    temp = yolo_detect[0][4] - yolo_detect[len(yolo_detect) // 2][4]
    if temp > 0:
        flag = 0
    elif temp < 0:
        flag = 1
    else:
        temp = yolo_detect[0][4] - yolo_detect[len(yolo_detect) // 4 * 3][4]
        if temp > 0:
            flag = 0
        else:
            flag = 1
    for i in range(edge, len(yolo_detect) - 1):
        temp = yolo_detect[i][4] - yolo_detect[i - 1][4]
        tempplus = yolo_detect[i + 1][4] - yolo_detect[i][4]
        tempintevral = yolo_detect[i + 1][4] - yolo_detect[i - 1][4]
        if flag == 0:  # 当递减的时候。
            if temp > 0:
                #  当不是递减的时候。
                yolo_detect[i][4] = yolo_detect[i - 1][4] - int(
                    round(abs(yolo_detect[i - 1][3] - yolo_detect[i][3]) * ratio))
            #  主要解决突然降低的情况。
            else:
                if tempplus > 0:
                    if tempintevral < 0:  # 间隔一个还是递减
                        #  在i这里是突然降低。
                        yolo_detect[i][4] = yolo_detect[i - 1][4] - int(round(abs(yolo_detect[i - 1][3] -
                                                                                  yolo_detect[i][3]) * ratio))



        else:
            # temp = yolo_detect[i][4] - yolo_detect[i - 1][4]
            # tempplus = yolo_detect[i + 1][4] - yolo_detect[i][4]
            # tempintevral = yolo_detect[i + 1][4] - yolo_detect[i - 1][4]
            if temp < 0:
                #  当不是递增的时候。
                yolo_detect[i][4] = yolo_detect[i - 1][4] + int(
                    round(abs(yolo_detect[i - 1][3] - yolo_detect[i][3]) * ratio))
            #  主要解决突然降低的情况。
            else:
                if tempplus < 0:
                    if tempintevral > 0:  # 间隔一个还是递增
                        yolo_detect[i][4] = yolo_detect[i - 1][4] + int(round(
                            abs(yolo_detect[i - 1][3] - yolo_detect[i][3]) * ratio))

    return yolo_detect


def get_raito_length_width(cord):
    """
    获取长宽比。
    :param cord: 输入的标记
    :return:
    @gxl  9/8
    """
    i = len(cord) - 1
    ratio = []
    while i > len(cord) - 2 and i > 0:
        #  求出长宽比
        ratio_temp = float(int(cord[i][4]) / int(cord[i][3]))
        ratio.append(ratio_temp)
        i -= 1
    return [min(ratio), max(ratio)]


def clearData():
    """
    在一次操作之后对所有的数据资源进行释放。这样下次才可以继续
    :return:
    """
    # 首先将txt中的文件都进行释放了。
    open('txt/yolosmooth.txt', 'w').close()
    open('txt/lumppospredicate.txt', 'w').close()
    # open('txt/yololump.txt', 'w').close()
    open('txt/yoloandhand.txt', 'w').close()
    # 对json数据进行释放。
    """
    json_url = 'openlabel/output/PASCAL_VOC/'
    label_jsons = os.listdir(json_url)
    for label_json in label_jsons:
        if label_json.endswith(".json"):
            os.remove(os.path.join(json_url, label_json))
    """
    #  对图片数据进行释放。
    images = os.listdir('openlabel/input/')
    for image in images:
        if image.endswith(".jpg"):
            os.remove(os.path.join('openlabel/input/', image))


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
        for i in range(len(frame_detect) - 10):  # 最后20帧不进行平滑
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


def clf(x, y, frame):
    """
    最小二乘法
    :return:
    """
    X1 = []
    X2 = []
    for x1 in x:
        temp_x2 = x1 ** 2
        temp_x3 = x1 ** 3
        # temp_x4 = x1 ** 4
        # X1.append([x1, temp_x2, temp_x3, temp_x4])
        X1.append([x1, temp_x2, temp_x3])
        # X1.append([x1])
    for x1 in frame:
        temp_x2 = x1 ** 2
        temp_x3 = x1 ** 3
        # temp_x4 = x1 ** 4
        # X1.append([x1, temp_x2, temp_x3, temp_x4])
        X2.append([x1, temp_x2, temp_x3])
        # X1.append([x1])
    x1 = np.array(X1)
    x2 = np.array(X2)
    y1 = np.array(y)
    X = sm.add_constant(x1)
    X2 = sm.add_constant(x2)
    st = sm.OLS(y1, X).fit()  # 方法二
    y_pred = st.predict(X2)
    return y_pred


if __name__ == '__main__':
    yolo = get_pos_for_txt('txt/yololump.txt')
    temp_edge = 0
    for i in range(len(yolo)):
        if yolo[i][3] > 50:
            temp_edge = i
            break
    x = []
    y = []
    for i in range(temp_edge, len(yolo)):
        x.append(int(yolo[i][0]) - 181850 + 1)
        y.append(int(yolo[i][1]))
    clf(x, y)
