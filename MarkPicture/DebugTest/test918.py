from LUMPOperate.OperatePhoto import get_pos_for_txt, clf
import numpy as np

yolo = get_pos_for_txt('C:/Users/gxl/PycharmProjects/train/MarkPicture/txt/yololump.txt')
temp_edge = 0
for i in range(len(yolo)):
    if yolo[i][3] > 50:
        temp_edge = i
        break
print(temp_edge)
x = []
y = []
for i in range(len(yolo) - 10, len(yolo)):
    temp_x = int(yolo[i][0]) - 181850 + 1
    temp_x2 = temp_x ** 2
    temp_x3 = temp_x ** 3
    temp_x4 = temp_x ** 4
    x.append([temp_x, temp_x2, temp_x3, temp_x4])
    y.append(int(yolo[i][1]))

clf(x, y)
