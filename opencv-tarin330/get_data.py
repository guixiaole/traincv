# 关于操作video
from datetime import datetime
from idlelib.multicall import r
import csv
import cv2
import xlrd
import configparser
import os
import shutil
from opencvyolo_0502 import yolov3_detect, findnet, finln_out

config = configparser.ConfigParser()
config.read("train_config.ini", encoding='UTF-8')
csv_url = config.get('file-url', 'csv-url')


# 在这里获得每一帧的多少距离。以秒来存储
# 这里写的是之前的数据
def getxlsxdata1():
    data = xlrd.open_workbook(r"train_0807.xlsx")
    table = data.sheets()[0]
    times = table.col_values(2)  # 获取时间
    speed = table.col_values(5)  # 获取相对应的距离
    speed_hours = table.col_values(8)  # 获取相对应的速度
    ncols = table.nrows  # 获取相对应的列数
    # 通过列数来得到
    alldistance = 0  # 总共行驶了多少距离
    frame = 50  # 设置每秒多少帧
    frame_distance = []
    hour_speed = []

    for i in range(2, ncols - 1):

        d1 = datetime.strptime(str(times[i]), '%H:%M:%S')
        d2 = datetime.strptime(str(times[i - 1]), '%H:%M:%S')
        s = d1 - d2  # 相对应的时间
        time_minus = s.seconds
        speed_minus = int(speed[i - 1]) - int(speed[i])  # 相对应的路程

        for j in range(time_minus):  # 计算每帧多少距离，只存前几秒
            distance = (speed_minus / time_minus) / 50

            frame_distance.append(round(distance, 4))  # 计算出来了每帧多少距离。
        for h in range(time_minus):
            hour_speed.append(table.col_values(8)[i])
        # 这里计算出了每帧多少秒，计算出每帧多少秒之后，然后通过给出的距离
        # 计算增帧还是减帧,增多少帧，减多少帧
        # 这边设置一个速度60km/h
        # 四舍五入速度。
    # print(hour_speed)
    # print(len(hour_speed),len(frame_distance))
    # speed_hour = 60
    # speed_frame = (speed_hour * 1000) / 3600 / 50  # 给出的每帧多少米
    # speed_frame = round(speed_frame, 4)

    # 将给定的速度除以每帧多少米。
    # 随着时间过去。而来动态的呈现出来
    # print(speed_frame)
    # print(frame_distance,hour_speed)
    return frame_distance, hour_speed


def getxlsxdata():
    """
    这里的代码应该是写一个每帧多少米
    :return: 返回的应该是一个列表，里面存了每帧多少米,以及在这一秒的时间的瞬时速度
    7/18
    """
    timer = []  # 系统时间
    real_distance = []  # 相对距离
    speed_hours = []  # 时速
    frame_distance_list = []  # 这里是最后返回的，每帧多少米。存储的是每秒的距离
    hour_speed_list = []  # 在每秒的时候的时速。
    flag = 1
    with open(csv_url, mode='r') as f:
        data = csv.reader(f)
        for row in data:
            if row[5] != '' and flag != 1:
                timer.append(row[2])
                real_distance.append(int(row[5]))
                speed_hours.append(int(row[8]))
            else:
                flag = 0
    h_flag = 1
    i = 1
    while i < len(timer):
        d1 = datetime.strptime(str(timer[i]), '%H:%M:%S')
        d2 = datetime.strptime(str(timer[i - 1]), '%H:%M:%S')
        s = d1 - d2  # 相对应的时间
        time_minus = s.seconds
        if time_minus == 0:
            if real_distance[i] == "":
                timer.pop(i)
                real_distance.pop(i)
                speed_hours.pop(i)
            else:
                timer.pop(i - 1)
                real_distance.pop(i - 1)
                speed_hours.pop(i - 1)
        else:
            i += 1
    distance_error = 0

    flagnext = 0  # 当distance_error 出现的时候，给下一个出现的time
    for i in range(2, len(timer)):
        # 算法思路、
        # s = v0t+1/2at^2  时间间隔按照其中给的思路。设置一个误差变量，此刻的误差给下一个时刻。
        # 假设到最后误差超过一定的值，那么就平均分配。
        #
        d1 = datetime.strptime(str(timer[i]), '%H:%M:%S')
        d2 = datetime.strptime(str(timer[i - 1]), '%H:%M:%S')
        s = d1 - d2  # 相对应的时间
        time_minus = s.seconds
        if time_minus != 0:
            distance_minus = int(real_distance[i - 1]) - int(real_distance[i])  # 相对应的路程
            if distance_minus < 0 and int(real_distance[i]) - int(real_distance[i - 1]) > 50:
                # 假设已经到了信号灯这里，那么将i处的距离改成0
                distance_minus = int(real_distance[i - 1])
            # 计算st 即本该走了多少路。
            a = ((int(speed_hours[i]) - int(speed_hours[i - 1])) / time_minus) / 3.6
            st = (int(speed_hours[i - 1]) * time_minus) / 3.6 + 0.5 * a * time_minus * time_minus  # 本该走的理论值
            if abs(st - distance_minus) < 5:  # 误差不大的时候，采用的是csv给出的
                # 当小于5的时候，用的距离即为csv
                # distance_error 也曾在负数的情况。
                if flagnext == 1:
                    #  误差补偿 大于0指的是还有路程没有算进去，
                    #  小于0指的是多算了路程。
                    if distance_error > 0:
                        if 0 < st - distance_minus < distance_error:
                            distance_minus += (st - distance_minus)
                            distance_error -= (st - distance_minus)
                        else:
                            if st - distance_minus < 0:
                                pass
                            else:
                                distance_minus += distance_error
                                distance_error = 0
                                flagnext = 0
                    else:
                        if 0 > st - distance_minus > distance_error:
                            distance_minus += (st - distance_minus)
                            distance_error -= (st - distance_minus)
                        else:
                            if st - distance_minus < 0:
                                distance_minus += distance_error
                                distance_error = 0
                                flagnext = 0
                flagdistance = distance_minus
                sj_error = (distance_minus - st) / time_minus  # 单个里面的误差
                for j in range(time_minus):
                    if a == 0:  # 当加速度为0的时候，即匀速运动的时候。
                        distance = (distance_minus / time_minus) / 50
                    else:  # 非匀速运动的时候，就需要按秒进行
                        if j != time_minus - 1:  # 用加速度去计算的时候，会有误差。
                            vj = a * j + float(speed_hours[i - 1]) / 3.6
                            sj = (vj + 0.5 * a) + sj_error
                            flagdistance -= sj
                            distance = sj / 50
                        else:
                            distance = flagdistance / 50
                    if distance < 0:
                        distance = (distance_minus / time_minus) / 50
                    frame_distance_list.append(distance)
            else:  # 假设是st过大的话，其实不用改，但是如果是distance_minus过大则需要修改
                # 如果相差过大的情况，就用自己算出来的结果。
                distance_error += (distance_minus - st)
                flagnext = 1
                # distance_error 在什么时候用呢？
                flagdistance = st
                for j in range(time_minus):
                    if a == 0:  # 当加速度为0的时候，即匀速运动的时候。
                        distance = (st / time_minus) / 50
                    else:  # 非匀速运动的时候，就需要按秒进行
                        if j != time_minus - 1:
                            vj = a * j + speed_hours[i - 1] / 3.6
                            sj = vj + 0.5 * a
                            flagdistance -= sj
                            distance = sj / 50
                        else:
                            distance = flagdistance / 50
                    frame_distance_list.append(distance)

            for h in range(time_minus):
                hour_speed_list.append(speed_hours[i])
    return frame_distance_list, hour_speed_list


def getxlsxdata2():
    """
    错误的版本，即原始的csv有错误，但是没有进行过更改。
    这里的代码应该是写一个每帧多少米
    :return: 返回的应该是一个列表，里面存了每帧多少米,以及在这一秒的时间的瞬时速度
    7/18
    """
    timer = []  # 系统时间
    real_distance = []  # 相对距离
    speed_hours = []  # 时速
    frame_distance = []  # 这里是最后返回的，每帧多少米。存储的是每秒的距离
    hour_speed = []  # 在每秒的时候的时速。
    with open(csv_url, mode='r') as f:
        data = csv.reader(f)
        for row in data:
            if row[5] != '':
                timer.append(row[2])
                real_distance.append(row[5])
                speed_hours.append(row[8])
    h_flag = 1
    for i in range(2, len(timer)):
        d1 = datetime.strptime(str(timer[i]), '%H:%M:%S')
        d2 = datetime.strptime(str(timer[i - 1]), '%H:%M:%S')
        s = d1 - d2  # 相对应的时间
        time_minus = s.seconds
        if time_minus != 0:

            if time_minus == 0 or real_distance[i] == '' or real_distance[i - 1] == '':
                # if timer[i - 1] == timer[i - 2]:
                #     speed_minus = int(real_distance[i - 2]) - int(real_distance[i])  # 相对应的路程
                # else:
                speed_minus = 0
            else:
                speed_minus = int(real_distance[i - 1]) - int(real_distance[i])  # 相对应的路程
            if speed_minus < 0:
                # 假设已经到了信号灯这里，那么将i处的距离改成0
                speed_minus = int(real_distance[i - 1])
            for j in range(time_minus):  # 计算每帧多少距离,这里存储的是每帧多少米，按照秒来存储的。
                distance = (speed_minus / time_minus) / 50
                if distance == 0.0:
                    distance = frame_distance[len(frame_distance) - 1]
                frame_distance.append(round(distance, 4))
                h_flag += 1
            for h in range(time_minus):
                hour_speed.append(speed_hours[i])
    return frame_distance, hour_speed


# 需要加速多少
def speed_up(speed_now, speed_pass):  # 给出需要的速度，和当前的速度。
    # 其中大概按照每一帧的播放速度是10毫秒来进行计算（可能不准确，大概按照这个速度）
    # 正常播放速度是50帧每秒。所以最快的播放速度是100帧每秒，大概是正常速度的加速2倍，超过四倍则进行抽帧
    # 减速播放则可以无限的播放某一帧，则不需要增帧
    # 给出的速度都是以km/h为单位的
    beisu = float(int(speed_now) / int(speed_pass))
    beishu = float(50 * beisu)  # 诶，步步都丢失了精度,这里计算出来每秒多少帧
    delay = float(1000 / beishu) - 10  # 这就是要延时的速度  最多两倍速。
    if delay < 0:
        # delay = 0
        #  假设延迟，先采用四舍五入的做法。
        if round(beisu) == 1:
            beisu = 2
        else:
            beisu = round(beisu) + 1
    # 这里还没有给出需要抽帧的操作
    # 这里还是要计算抽帧的操作。

    return delay, int(beishu), beisu


def acceleration(speed_final, speed_now, frame):  # 加速度
    """
    给的是火车的km/h的加速度
    当现在的速度与目标速度不匹配时。求出下一帧的速度
    动车的加速度大概在1m/s2左右
    那就按照1m/s2的加速度来设置吧
    这样的话，按照当前的每一帧加多少速度，
    """
    speed_now1 = speed_now
    if int(speed_final) != int(speed_now):
        # 按照1m/s2的速度换算成km/h2
        # 每一帧增加的速度大概是

        speed = float(1 / frame)  # 大概是每一帧增加了多少m/s
        # 1m/s=3.6km/h
        speed = speed * 3.6  # 每一帧增加的速度
        # 判断是加速度还是减速度
        if speed_final > speed_now:  # 加速度
            speed_now1 = speed_now + speed
            if speed_now1 > speed_final:
                speed_now1 = speed_final
        else:
            speed_now1 = speed_now - speed
            if speed_now1 < speed_final:
                speed_now1 = speed_final

    return speed_now1


def distance_frame(speed):
    """
    计算出一帧的时间走了多少路程。
    根据s = v0t+at^2/2
    其中t为1s/50=0.02s
    其中a=1m/s^2
    :param speed: 速度为km/h 需要对其进行换算。
    1 千米/时(km/h)=0.277777778 米/秒(m/s)
    :return:
    """
    speed_s = speed * 0.277777778
    s = speed_s * 0.02 + 0.2 * 0.2 / 2
    # 即为在加速的时候，
    return s


def jump_frame_for_distance(s, frame_distance):
    """

    :param s:代表的是总路程
    :param frame_distance: 代表的是每帧的距离。
    :return:返回的是跳的帧数以及剩余的误差距离。
    """
    if frame_distance != 0:
        jump_frame = s // frame_distance
        last_distance = float(s - (frame_distance * jump_frame))
        return jump_frame, last_distance
    else:
        return 1, 0


def class_videofile():
    """
    这个函数主要是对视频进行分类，用系统文件进行操作。
    :return:
    """
    # 首先获取该目录下的所有视频文件名字。

    file_path = "E:/衡阳到长沙/video_part/"
    file_path_three = "E:/衡阳到长沙/video_part/three"
    video_name = os.listdir(file_path)
    net = findnet()
    ln, out = finln_out(net)
    for i in range(len(video_name)):

        video_path = os.path.join(file_path, video_name[i])
        #  获取到了该目录下的所有的文件
        #  所有的视频的总帧数应该是24*50=1200
        #  需要识别的应该在900帧的样子。
        print('video_path=', video_path)
        cap = cv2.VideoCapture(video_path)  # 读取视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
        while True:
            while cap.isOpened():

                ret, frame = cap.read()
                if ret:
                    boxes, conf, classID = yolov3_detect(frame, net, ln, out)
                    if len(conf) > 0:
                        # 当有的时候，就是暂停。
                        print(boxes, conf, classID)
                        for j in range(len(classID)):
                            if classID[j] == 2:
                                cap.release()
                                video_three_path = os.path.join(file_path_three, video_name[i])
                                shutil.move(video_path, video_three_path)
                                break
                        break
                else:
                    break
            break
    print(video_name)


def video_detetct():
    """
    测试yolo探测的结果。
    :return:
    """
    file_path_three = "E:/衡阳到长沙/video_part/three/1258out.avi"
    cap = cv2.VideoCapture(file_path_three)
    i = 850
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    net = findnet()
    ln, out = finln_out(net)

    while cap.isOpened():
        ret, frame = cap.read()
        boxes, conf, classID = yolov3_detect(frame, net, ln, out)
        print('frame_count=', i, 'boxes=', boxes, 'conf=', conf, 'classID=', classID)
        i += 1


def videtoimage():
    file_path = "E:/衡阳到长沙/video_part/"
    file_path_three = "E:/衡阳到长沙/video_part/three"
    video_name = os.listdir(file_path)
    for i in range(len(video_name)):
        video_path = os.path.join(file_path, video_name[i])
        cap = cv2.VideoCapture(video_path)  # 读取视频
        cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
        frame_count = 1000
        j = 1

        #  每个保存三帧。
        while cap.isOpened():
            while j <= 1:
                ret, frame = cap.read()
                cv2.imwrite('E:/train_photo/' + "_%d.jpg" % (i * frame_count), frame)
                j += 1
                frame_count += 1
            break
        cap.release()

# if __name__ == '__main__':
# class_videofile()
# print(speed_up(60,60))
# frame_distance, house_speed = getxlsxdata()
# with open("hashmap.txt", "a") as file:
#     for i in range(len(frame_distance)):
#         file.write("framedistance:"+str(frame_distance[i]) + ";")
#         file.write("house_speed:"+str(house_speed[i]) + ";")
#         file.write("\n")

#  video_detetct()
# class_videofile()
if __name__ == '__main__':
    getxlsxdata1()