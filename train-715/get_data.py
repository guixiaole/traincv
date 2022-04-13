# 关于操作video
from datetime import datetime
import cv2
import xlrd


# 在这里获得每一帧的多少距离。以秒来存储

def getxlsxdata():
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

            frame_distance.append(round(distance, 4))
        for h in range(time_minus):
            hour_speed.append(table.col_values(8)[i])
        # 这里计算出了每帧多少秒，计算出每帧多少秒之后，然后通过给出的距离
        # 计算增帧还是减帧,增多少帧，减多少帧
        # 这边设置一个速度60km/h
        # 四舍五入速度。
    # print(hour_speed)
    # print(len(hour_speed),len(frame_distance))
    speed_hour = 60
    speed_frame = (speed_hour * 1000) / 3600 / 50  # 给出的每帧多少米
    speed_frame = round(speed_frame, 4)
    # 我的天？？？？我这里代码写死了是干嘛啊，根本没用上啊，测试吗？？？

    # 将给定的速度除以每帧多少米。
    # 随着时间过去。而来动态的呈现出来
    # print(speed_frame)
    # print(frame_distance,hour_speed)
    return frame_distance, hour_speed


# 需要加速多少
def speed_up(speed_now, speed_pass):  # 给出需要的速度，和当前的速度。
    # 其中大概按照每一帧的播放速度是10毫秒来进行计算（可能不准确，大概按照这个速度）
    # 正常播放速度是50帧每秒。所以最快的播放速度是100帧每秒，大概是正常速度的加速2倍，超过四倍则进行抽帧
    # 减速播放则可以无限的播放某一帧，则不需要增帧
    # 给出的速度都是以km/h为单位的
    beishu = float(int(speed_now) / int(speed_pass))
    beishu = float(50 * beishu)  # 诶，步步都丢失了精度,这里计算出来每秒多少帧
    delay = float(1000 / beishu) - 10  # 这就是要延时的速度
    if delay < 0:
        delay = 0
    # 这里还没有给出需要抽帧的操作
    return delay, int(beishu)


def acceleration(speed_final, speed_now, frame):  # 加速度
    """
    给的是火车的km/h的加速度
    当现在的速度与目标速度不匹配时。求出下一帧的速度
    动车的加速度大概在1m/s2左右
    那就按照1m/s2的加速度来设置吧
    这样的话，按照当前的每一帧加多少速度，
    """
    if int(speed_final) != int(speed_now):
        # 按照1m/s2的速度换算成km/h2
        # 每一帧增加的速度大概是
        speed = float(1 / frame)  # 大概是每一帧增加了多少m/s
        # 1m/s=3.6km/h
        speed = speed * 3.6  # 每一帧增加的速度
        # 判断是加速度还是减速度
        if speed_final > speed_now:  # 加速度
            speed_now = speed_now + speed
            if speed_now > speed_final:
                speed_now = speed_final
        else:
            speed_now = speed_now - speed
            if speed_now < speed_final:
                speed_now = speed_final

    return speed_now


# print(speed_up(60,60))
print(getxlsxdata())
