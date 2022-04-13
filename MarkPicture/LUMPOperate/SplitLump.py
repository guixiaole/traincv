"""
把信号灯给分割出来。
"""
import csv
import json
import os
from datetime import datetime

import cv2

video_path = "E:/衡阳到长沙/衡阳-岳阳.mp4"


def saveImage_calssification():
    """
    将每个视频的图片进行存储。存储在根目录下的ShowImage目录下面。
    :return:
    """
    video_path = "E:/save_video/"
    video_name = os.listdir(video_path)
    frame_save = []
    for i in range(len(video_name)):
        video_path_part = os.path.join(video_path, video_name[i])
        cap = cv2.VideoCapture(video_path_part)  # 读取视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
        while True:
            ret, frame = cap.read()
            if ret:
                frame_save.append(frame)
            break
        cap.release()
    #  保存了所有的视频的图片
    video_path_image = video_path + "ShowImage"
    if not os.path.exists(video_path_image):
        os.makedirs(video_path_image)
        #  创建文件夹
    for i in range(0, len(frame_save)):
        cv2.imwrite(video_path_image + "/%s.jpg" % video_name[i], frame_save[i])
        #  进行储存照片


def video_lump_classification(video_name, csv_url, save_url):
    """
    这个函数主要用于处理将有信号灯的那几帧视频进行保存。
    按照距离开始保存吧。
    视频从30s开始进行出站，如果相差很大的距离的时候，就是代表着这个时刻是有信号灯的，那么向前数10s，都视为有信号灯，将其进行保存。
    :return:
    """
    #  video_name = 'E:/衡阳到长沙/衡阳-岳阳.mp4'
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
    temp_i = []
    nolumpstart = 1
    nolumpend = 1
    nolump_saveurl = save_url + 'nolump/'
    start_time = datetime.strptime(str(timer[1]), '%H:%M:%S')  # 起始时间。
    for i in range(2, len(real_distance)):
        if real_distance[i] != '' and real_distance[i - 1] != '':
            if int(real_distance[i]) > int(real_distance[i - 1]) and abs(
                    int(real_distance[i]) - int(real_distance[i - 1])) > 100 and i > 300:
                #  再往前拉。当大于150米的时候就停止。
                temp_i.append(i)
                flag_pos = i - 1  # 就是往前拉的时候，什么时候大于150就停止。
                while real_distance[flag_pos] == '' or int(real_distance[flag_pos]) <= 300:
                    flag_pos -= 1
                flag_pos -= 1
                cap = cv2.VideoCapture(video_name)  # 读取视频
                end_time = datetime.strptime(str(timer[i]), '%H:%M:%S')  # 这一刻的结束时间\
                start_km_time = datetime.strptime(str(timer[flag_pos]), '%H:%M:%S')  # 从这一刻开始录视频
                s_km = start_km_time - start_time
                km_second = s_km.seconds
                s = end_time - start_time
                second = s.seconds
                # 然后通过计算时间。
                frame_start = km_second * 50 + 1500  # 这是开始的那一帧
                frame_end = second * 50 + 1700  # 结束的时候多加200帧，即4s钟。这样好区分。
                nolumpend = frame_start - 1

                video_store_nolump(video_name, nolump_saveurl, nolumpstart, nolumpend)  # 将没有的信号灯的视频进行分割储存。
                nolumpstart = frame_end + 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
                # 视频的宽度
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # 视频的高度
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # 视频的帧率  视频的编码  定义视频输出
                fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                #  存储应该是lujing+out+framecount.avi
                out = cv2.VideoWriter(save_url + "out" + str(km_second * 50) + ".avi", fourcc, fps, (width, height))

                #  out = cv2.VideoWriter(str(i) + 'output.avi', fourcc, 50.0, (1920, 1080))
                print("i=", i)
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
    print("temp_i:", temp_i)


def video_lump_store_classification1(video_name, csv_url, save_url):
    """
    当记录说明为 出战，进站，过信号灯时。然后往前走三百米，然后将其信号灯进行保存。
    保存名字为 序号+out+framecount.avi
    :return:
    """
    timer = []  # 系统时间
    real_distance = []  # 相对距离
    speed_hours = []  # 时速
    frame_distance = []  # 这里是最后返回的，每帧多少米。存储的是每秒的距离
    hour_speed = []  # 在每秒的时候的时速。
    i = 0
    save_csv = []  # 需要保存的数据，即含有出站，进站，过信号机的数据。
    with open(csv_url, mode='r') as f:
        data = csv.reader(f)
        for row in data:
            if row[1] == '进站' or row[1] == '出站' or row[1] == '过信号机':
                save_csv.append([i, row[2], row[5], row[8]])
            timer.append(row[2])
            real_distance.append(row[5])
            speed_hours.append(row[8])
            i += 1
        # 将需要的数据进行保存。
    # temp_i = []
    nolumpstart = 1
    nolumpend = 0
    nolump_saveurl = save_url + 'nolump/'
    start_time = datetime.strptime(str(timer[1]), '%H:%M:%S')  # 起始时间。
    for i in range(0, len(save_csv)):
        pos = save_csv[i][0] - 1
        while real_distance[pos] == "" or int(real_distance[pos]) <= 300:
            if pos > 1:
                pos -= 1
            else:  # 边界问题，预防出了边界。
                break
        if pos != 1:
            pos -= 1
        # temp_i.append(i)
        cap = cv2.VideoCapture(video_name)
        end_time = datetime.strptime(str(timer[int(save_csv[i][0])]), '%H:%M:%S')  # 这一刻的结束时间\
        start_km_time = datetime.strptime(str(timer[pos]), '%H:%M:%S')  # 从这一刻开始录视频
        s_km = start_km_time - start_time
        km_second = s_km.seconds
        s = end_time - start_time
        second = s.seconds
        # 然后通过计算时间。
        frame_start = km_second * 50 + 1500  # 这是开始的那一帧
        frame_end = second * 50 + 1700  # 结束的时候多加200帧，即4s钟。这样好区分。
        nolumpend = frame_start - 1
        video_store_nolump(video_name, nolump_saveurl, nolumpstart, nolumpend, i)  # 将没有的信号灯的视频进行分割储存。
        nolumpstart = frame_end + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        # 视频的宽度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 视频的高度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 视频的帧率  视频的编码  定义视频输出
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #  存储应该是i+lujing+out+framecount.avi
        out = cv2.VideoWriter(save_url + str(i) + "out" + str(km_second * 50) + ".avi", fourcc, fps, (width, height))
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
        # print("temp_i:", temp_i)


def video_store_nolump(video_name, save_url, start, end, i):
    """
    存储没有信号灯的视频部分。
    传入的视频为开始帧和结束帧。
    :return:
    """
    cap = cv2.VideoCapture(video_name)  # 读取视频
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # 视频的高度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 视频的帧率  视频的编码  定义视频输出
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #  存储应该是lujing+out+framecount.avi
    out = cv2.VideoWriter(save_url + str(i) + "out" + str(start) + ".avi", fourcc, fps, (width, height))

    #  out = cv2.VideoWriter(str(i) + 'output.avi', fourcc, 50.0, (1920, 1080))

    while start < end:
        while cap.isOpened():
            ret, frame = cap.read()
            #  frame = cv2.flip(frame, 0)
            #  cv2.imshow('frame', frame)
            #  cv2.waitKey(0)
            out.write(frame)
            if start >= end:
                cap.release()
                out.release()
                print('END')
                break
            start += 1
            print(start)


def store_image_to_input(video_url, frame_count):
    """
    将选中的照片放入input中。
    :return:
    @gxl 8/17
    """
    store_url = 'openlabel/input/'
    cap = cv2.VideoCapture(video_url)  # 读取视频
    frames_num = cap.get(7)  # 获取视频总帧数、
    frames_range = int(frames_num // 50)
    for i in range(frames_range):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * 50)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(store_url + "%d.jpg" % (frame_count + i * 50), frame)
    cap.release()


def get_output_labeljson():
    """
    获取标注好的json文件。当获取完毕之后，就开始删除所有的json文件。
    :return: 返回的是已经设置好的坐标。
    @gxl  8/25
    """
    json_url = 'openlabel/output/PASCAL_VOC/'
    txt_url = 'txt/lumppospredicate.txt'
    label_jsons = os.listdir(json_url)
    cord = []
    for label_json in label_jsons:
        label_url = json_url + label_json
        label_count = int(label_json[:-5])
        with open(label_url, 'r') as f:
            dict_str = json.loads(f.read())
            dict_shap_all = dict_str['shapes']
            if len(dict_shap_all) >= 2:
                # 这里暂时写死了，后期可能会改，目前只有两个灯的实现。
                pos = dict_shap_all[0]['points']
                xmax = round(pos[0][0])
                ymax = round(pos[0][1])
                xmin = round(pos[1][0])
                ymin = round(pos[1][1])
                w = abs(int(xmax) - int(xmin)) + 1
                h = abs(int(ymax) - int(ymin)) + 1
                pos1 = dict_shap_all[1]['points']
                xmax1 = round(pos1[0][0])
                ymax1 = round(pos1[0][1])
                xmin1 = round(pos1[1][0])
                ymin1 = round(pos1[1][1])
                w1 = abs(int(xmax1) - int(xmin1)) + 1
                h1 = abs(int(ymax1) - int(ymin1)) + 1
                cord.append((label_count, xmax, ymax, w, h, xmax1, ymax1, w1, h1))
            else:
                dict_str = dict_str['shapes'][0]
                pos = dict_str['points']
                print(pos)
                xmax = round(pos[0][0])
                ymax = round(pos[0][1])
                xmin = round(pos[1][0])
                ymin = round(pos[1][1])
                print(label_count, xmax, ymax, xmin, ymin)
                w = abs(int(xmax) - int(xmin)) + 1
                h = abs(int(ymax) - int(ymin)) + 1
                cord.append((label_count, xmax, ymax, w, h))
    print(cord)
    #  获得所有标注的坐标之后，需要将xml文件进行删除。
    """
    暂时先注释掉。后面再
    for label_json in label_jsons:
        if label_json.endswith(".json"):
            os.remove(os.path.join(json_url, label_json))
    """
    # #  把预测的数据进行存储。
    # 为什么要储存啊？？？？迷惑
    # for j in range(0, len(cord)):
    #     with open(txt_url, "a") as file:
    #         file.write(str(cord[j][0]) + ";")
    #         for i in range(1, len(cord[j])):
    #             file.write(str(cord[j][i]) + ",")
    #         file.write("\n")
    return cord



if __name__ == '__main__':
    get_output_labeljson()
