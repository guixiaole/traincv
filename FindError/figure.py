from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xlrd

data = xlrd.open_workbook(r"E:\\test1.xlsx")
table = data.sheets()[0]
times = table.col_values(2)  # 获取时间
guangya_temp = table.col_values(13)  # 获取时间
gangyatemp = table.col_values(15)  # 获取时间
jungan1gtemp = table.col_values(16)  # 获取时间
jungang2temp = table.col_values(17)  # 获取时间
# speed = table.col_values(5)  # 获取相对应的距离
# speed_hours = table.col_values(8)  # 获取相对应的速度
ncols = table.nrows  # 获取相对应的列数
# 通过列数来得到
print(times)
alldistance = 0  # 总共行驶了多少距离
frame = 50  # 设置每秒多少帧
frame_distance = []
hour_speed = []

for i in range(2, ncols - 1):
    print(times[i])
    d1 = datetime.strptime(str(times[i]), '%H:%M:%S')
    d2 = datetime.strptime(str(times[i - 1]), '%H:%M:%S')
    s = d1 - d2  # 相对应的时间
    time_minus = s.seconds
    for j in range(time_minus):  # 计算每帧多少距离，只存前几秒
        pass
