import csv
from datetime import datetime


def getxlsxdata():
    """
    这里的代码应该是写一个每帧多少米
    :return: 返回的应该是一个列表，里面存了每帧多少米,以及在这一秒的时间的瞬时速度
    7/18
    """
    csv_url = "E:\衡阳到长沙\9月22日Z138次衡阳至岳阳全程记录(最终版.csv"
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
        if i == 5:
            print(i)
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
                        if distance_minus != 0:
                            distance = (distance_minus / time_minus) / 50
                        else:
                            if speed_hours[i] == speed_hours[i - 1]:
                                distance = frame_distance_list[-1]
                            else:
                                distance = frame_distance_list[-1]  # todo:这里暂时还没有解决，还没有好的想法。

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


if __name__ == '__main__':
    frame_distanc, hour_speed = getxlsxdata()
    for i in range(len(frame_distanc)):
        print(frame_distanc[i], ':', hour_speed[i])
