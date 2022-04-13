from xml.dom.minidom import Document
import math
import cv2
import numpy as np
from typing import List
from numpy import *
from get_frame_singal import get_frame_singal
import os
import re
import copy


# import LUMPOperate.OperatePhoto as OperatePhoto


def findcenter(image, boxes):
    # cv2.imshow('image1',image)
    # cv2.imshow('image1',image)
    #  frame_count = 74065

    # cv2.imwrite('F:/train_photo/' + "_%d.jpg" % frame_count, image)
    # image_trans_path = "E:\\trans\\trans3.png"
    #  image_trans_path = "F:\\train_photo\\trans.png"
    # image_trans = cv2.imread(image_trans_path)
    # cv2.imshow('image_trans12', image_trans)
    # image=cv2.imread(imagepath)
    image_trans_path = "E:\\trans\\trans3.png"
    # image_trans_path = "F:\\train_photo\\trans.png"
    image_trans = cv2.imread(image_trans_path)
    # cv2.imshow('image_trans12', image_trans)
    # image=cv2.imread(imagepath)
    (x, y) = (int(boxes[0]), int(boxes[1]))  # 框左上角
    (w, h) = (int(boxes[2]), int(boxes[3]))  # 框宽高
    # (x, y) = (int(boxes[0]), int(boxes[1]))  # 框左上角
    # (w, h) = (int(boxes[2]), int(boxes[3]))  # 框宽高

    crop = image[y:(h + y), x:(w + x)]

    # cv2.imshow('first', crop)
    # cv2.waitKey(0)
    if w > 100 or h > 100:

        x = x - int(w * 0.075)
        y = y - int(h * 0.075)

        # w += 10
        # h += 10
        w += w * 0.15
        h += h * 0.15
        w = int(w)
        h = int(h)
    else:
        x = x - 7
        y = y - 7

        w += 14
        h += 14
    # x = x - int(w * 0.075)
    # y = y - int(h * 0.075)
    #
    # # w += 10
    # # h += 10
    # w += w * 0.15
    # h += h * 0.15
    # w = int(w)
    # h = int(h)
    crop = image[y:(h + y), x:(w + x)]
    # cv2.imshow('crop3', crop)
    # cv2.waitKey(0)
    # 在这里设置了一个问题，这里必须要改
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    crop = image[y:(h + y), x:(w + x)]  # 这只是框的大小
    # crop3=image[y+9:y+32,x+6:x+24]

    # 先把R通道的给提取出来
    #
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2YUV)
    # cv2.imshow('crop3',crop)
    img_r = [[0 for p in range(w)] for q in range(h)]
    for i in range(h):
        for j in range(w):
            img_r[i][j] = int(crop[i][j][0])

    # 获得R通道的矩阵之后，进行一个算法寻找

    # 这边去掉噪音的方法还是不行。
    # print('kaishi ')

    # for i in range(len(crop)):
    #     print(img_r[i])

    boundary = 90
    left, right = findleft_right(len(img_r) // 2, img_r, boundary)  # 寻找到左右之后
    left1, right1 = findleft_right(len(img_r) // 2 + 1, img_r, boundary)  # 寻找到左右之后
    left2, right2 = findleft_right(len(img_r) // 2 - 1, img_r, boundary)  # 寻找到左右之后
    left3, right3 = findleft_right(len(img_r) // 2 - 2, img_r, boundary)  # 寻找到左右之后
    left4, right4 = findleft_right(len(img_r) // 2 + 2, img_r, boundary)  # 寻找到左右之后

    left_list = [left, left1, left2, left3, left4]
    right_list = [right, right1, right2, right3, right4]
    mean_all = 8  # 大于某个数的时候就会删掉。

    # right_mean=minus_max(right_list,len(img_r[0]) - 1)
    # right_mean = int(round(mean(right_list)))
    high, low = findhigh_low(len(img_r[0]) // 2, img_r, boundary)  # 寻找到左右之后
    high1, low1 = findhigh_low(len(img_r[0]) // 2 + 1, img_r, boundary)  # 寻找到左右之后
    high3, low3 = findhigh_low(len(img_r[0]) // 2 + 2, img_r, boundary)  # 寻找到左右之后
    high2, low2 = findhigh_low(len(img_r[0]) // 2 - 1, img_r, boundary)  # 寻找到左右之后
    high4, low4 = findhigh_low(len(img_r[0]) // 2 - 2, img_r, boundary)  # 寻找到左右之后
    high_list = [high, high1, high2, high3, high4]
    low_list = [low, low1, low2, low3, low4]
    high_list.sort()
    low_list.sort()
    left_list.sort()
    right_list.sort()

    # print(right_mean)
    print(high_list, low_list)
    print(left_list, right_list)

    # high_mean = minus_max(high_list,len(img_r) - 1)
    if w > 100:
        high_mean = high_list[0]
        low_mean = low_list[0]
        left_mean = left_list[0]
        right_mean = right_list[0]
    else:
        high_mean = high_list[2]
        low_mean = low_list[2]
        left_mean = left_list[2]
        right_mean = right_list[2]
    # low_mean = minus_max(low_list,len(img_r) - 1)
    # low_mean = int(round(mean(low_list)))
    # left_mean = minus_max(left_list,len(img_r[0]) - 1)
    # left_mean = int(round(mean(left_list)))

    # 获取信号灯的大小之后，将代替换的照片进行压缩

    # cv2.imshow('image_trans',image_trans)

    shrink = cv2.resize(image_trans, (abs(right_mean - left_mean),
                                      abs(high_mean - low_mean)
                                      ),
                        interpolation=cv2.INTER_AREA)
    crop2 = crop[high_mean:low_mean, left_mean:right_mean]

    # print(boxes)
    # print('left_mean=',left_mean,'right_mean=',right_mean)
    # print('high_mean',high_mean,'low_mean=',low_mean)
    # 图像压缩之后，应该开始复制。
    # cv2.imshow('image1', image)
    # cv2.imshow('crop', crop)
    # cv2.imshow('crop2', crop2)
    i, j = 0, 0
    # print(shrink[0])
    # print(crop.shape)
    # print(shrink.shape)

    print(y + high_mean, x + left_mean)
    # for i in range(abs(right_mean - left_mean) - 1):
    #     for j in range(abs(high_mean - low_mean) - 1):
    #         if shrink[j][i][0] < 100:
    #             image[y + j + high_mean][x + i + left_mean] = shrink[j][i]

    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    ##现在可以试试进行替换看看
    # return image
    # 将坐标进行保存然后进行储存

    corrdinate = [x + left_mean, y + high_mean, abs(right_mean - left_mean) + 1, abs(high_mean - low_mean) + 1]
    corrdinate_test = [left_mean, high_mean, abs(right_mean - left_mean) + 1, abs(high_mean - low_mean) + 1]
    print(corrdinate_test)
    # print(corrdinate)
    return corrdinate


def findcenter1(image, boxes, frame_count1):
    # cv2.imshow('image1',image)
    # cv2.imshow('image1',image)
    #  frame_count = 74065

    # cv2.imwrite('F:/train_photo/' + "_%d.jpg" % frame_count, image)
    image_trans_path = "E:\\trans\\trans3.png"
    # image_trans_path = "F:\\train_photo\\trans.png"
    image_trans = cv2.imread(image_trans_path)
    cv2.imshow('image_trans12', image_trans)
    # image=cv2.imread(imagepath)
    (x, y) = (int(boxes[0]), int(boxes[1]))  # 框左上角
    (w, h) = (int(boxes[2]), int(boxes[3]))  # 框宽高
    # (x, y) = (int(boxes[0]), int(boxes[1]))  # 框左上角
    # (w, h) = (int(boxes[2]), int(boxes[3]))  # 框宽高

    crop = image[y:(h + y), x:(w + x)]

    cv2.imshow('first', crop)
    cv2.waitKey(0)
    # if w<100 and h<100:
    if w > 100 or h > 100:

        x = x - int(w * 0.075)
        y = y - int(h * 0.075)

        # w += 10
        # h += 10
        w += w * 0.15
        h += h * 0.15
        w = int(w)
        h = int(h)
    else:
        x = x - 7
        y = y - 7

        w += 14
        h += 14
    # x = x - int(w * 0.075)
    # y = y - int(h * 0.075)
    #
    # # w += 10
    # # h += 10
    # w += w * 0.15
    # h += h * 0.15
    # w = int(w)
    # h = int(h)
    crop = image[y:(h + y), x:(w + x)]
    cv2.imshow('crop3', crop)
    cv2.waitKey(0)
    # 在这里设置了一个问题，这里必须要改
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    crop = image[y:(h + y), x:(w + x)]  # 这只是框的大小
    # crop3=image[y+9:y+32,x+6:x+24]

    # 先把R通道的给提取出来
    #
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2YUV)
    # cv2.imshow('crop3',crop)
    img_r = [[0 for p in range(w)] for q in range(h)]
    for i in range(h):
        for j in range(w):
            img_r[i][j] = int(crop[i][j][0])

    # 获得R通道的矩阵之后，进行一个算法寻找

    # 这边去掉噪音的方法还是不行。
    # print('kaishi ')

    for i in range(len(crop)):
        print(img_r[i])

    boundary = 90
    left, right = findleft_right(len(img_r) // 2, img_r, boundary)  # 寻找到左右之后
    left1, right1 = findleft_right(len(img_r) // 2 + 1, img_r, boundary)  # 寻找到左右之后
    left2, right2 = findleft_right(len(img_r) // 2 - 1, img_r, boundary)  # 寻找到左右之后
    left3, right3 = findleft_right(len(img_r) // 2 - 2, img_r, boundary)  # 寻找到左右之后
    left4, right4 = findleft_right(len(img_r) // 2 + 2, img_r, boundary)  # 寻找到左右之后

    left_list = [left, left1, left2, left3, left4]
    right_list = [right, right1, right2, right3, right4]
    mean_all = 8  # 大于某个数的时候就会删掉。

    # right_mean=minus_max(right_list,len(img_r[0]) - 1)
    # right_mean = int(round(mean(right_list)))
    high, low = findhigh_low(len(img_r[0]) // 2, img_r, boundary)  # 寻找到左右之后
    high1, low1 = findhigh_low(len(img_r[0]) // 2 + 1, img_r, boundary)  # 寻找到左右之后
    high3, low3 = findhigh_low(len(img_r[0]) // 2 + 2, img_r, boundary)  # 寻找到左右之后
    high2, low2 = findhigh_low(len(img_r[0]) // 2 - 1, img_r, boundary)  # 寻找到左右之后
    high4, low4 = findhigh_low(len(img_r[0]) // 2 - 2, img_r, boundary)  # 寻找到左右之后
    high_list = [high, high1, high2, high3, high4]
    low_list = [low, low1, low2, low3, low4]
    high_list.sort()
    low_list.sort()
    left_list.sort()
    right_list.sort()

    # print(right_mean)
    print(high_list, low_list)
    print(left_list, right_list)

    # high_mean = minus_max(high_list,len(img_r) - 1)
    high_mean = high_list[2]
    low_mean = low_list[2]
    left_mean = left_list[2]
    right_mean = right_list[2]
    # low_mean = minus_max(low_list,len(img_r) - 1)
    # low_mean = int(round(mean(low_list)))
    # left_mean = minus_max(left_list,len(img_r[0]) - 1)
    # left_mean = int(round(mean(left_list)))

    # 获取信号灯的大小之后，将代替换的照片进行压缩

    # cv2.imshow('image_trans',image_trans)

    shrink = cv2.resize(image_trans, (abs(right_mean - left_mean),
                                      abs(high_mean - low_mean)
                                      ),
                        interpolation=cv2.INTER_AREA)
    crop2 = crop[high_mean:low_mean, left_mean:right_mean]

    # print(boxes)
    # print('left_mean=',left_mean,'right_mean=',right_mean)
    # print('high_mean',high_mean,'low_mean=',low_mean)
    # 图像压缩之后，应该开始复制。
    # cv2.imshow('image1', image)
    # cv2.imshow('crop', crop)
    # cv2.imshow('crop2', crop2)
    i, j = 0, 0
    # print(shrink[0])
    # print(crop.shape)
    # print(shrink.shape)

    print(y + high_mean, x + left_mean)
    for i in range(abs(right_mean - left_mean) - 1):
        for j in range(abs(high_mean - low_mean) - 1):
            if shrink[j][i][0] < 100:
                image[y + j + high_mean][x + i + left_mean] = shrink[j][i]

    cv2.imshow('image', image)
    cv2.waitKey(0)
    # cv2.imwrite('E:/save_video/replaceVideo/' + "_%d.jpg" % frame_count1, image)
    ##现在可以试试进行替换看看
    # return image
    # 将坐标进行保存然后进行储存

    corrdinate = [x + left_mean, y + high_mean, abs(right_mean - left_mean) + 1, abs(high_mean - low_mean) + 1]
    corrdinate_test = [left_mean, high_mean, abs(right_mean - left_mean) + 1, abs(high_mean - low_mean) + 1]
    print(corrdinate_test)
    # print(corrdinate)
    return corrdinate


def popzero(temp):
    i = 0
    while temp[i] == 0 and len(temp) > 0:
        temp.pop(i)
    return temp


# 这里的算法有问题，需要修改
# 不能只找最大的数字，需要看周围的数字误差是不是小于5
# 算法思路：寻找出最大的5个差值，然后再根据坐标从小到大进行选择，越小的坐标权值越大。
def findleft_right(leng_part, img_r: List[List[int]], boundary):  # 传入一个二维数组
    # 从二维数组中寻找到中心值
    max_value = 0
    left = 0
    if len(img_r[0]) > 100:
        outline = len(img_r[0]) // 10
    else:
        outline = len(img_r[0]) // 8
    # print(leng_part)
    flag = 0
    for i in range(1, len(img_r[0]) // 2):

        # value = abs(int(img_r[leng_part][i] - img_r[leng_part][i - 1]))
        if img_r[leng_part][i] <= boundary and img_r[leng_part][i - 1] >= boundary:
            left = i
            flag = 1
            break
    if abs(left - len(img_r[0]) // 2) <= outline:
        left = 1
    if left >= 30:
        left = 1
        '''
        if len(max_value)<5 :
            if img_r[leng_part][i]<100:
                max_value.append((value,i))
        else:
            if abs(value)>min(max_value)[0]:
                max_value.remove(min(max_value))
                max_value.append((value,i))
                '''
    # 这里已经挑选出了最大的五个数
    '''
    max_value.sort(key=takeSecond)#对第二个数进行排序，排序之后就开始进行
    #left=max_value[0][1]
    #print(max_value)

    for i in range(len(max_value)):
        vlaue,pos=max_value[i]
        flag=0
        for h in range(4,8):#往后数三到7个数
            if pos+h<len(img_r[0]):
                if abs(img_r[leng_part][pos+h]-img_r[leng_part][pos+h-1])>10:
                    flag=1
        if flag==0:
            left=pos
            break

'''
    '''
    max_value=0

    for j in range (len(img_r[0])-1,len(img_r[0])//2,-1):
        value = int(img_r[leng_part][j] - img_r[leng_part][j - 1])
        if abs(value)>max_value:
            max_value=abs(value)
            right=j
    '''
    right = len(img_r[0]) - 1
    max_value2 = []
    flag = 0
    for j in range(len(img_r[0]) - 1, len(img_r[0]) // 2, -1):
        imgrj = img_r[leng_part][j]
        imgrjminus = img_r[leng_part][j - 1]
        if imgrj >= boundary and imgrjminus <= boundary:
            flag = 1
            right = j
            break
    if abs(right - len(img_r[0]) // 2) <= outline:
        right = len(img_r[0]) - 2
    if len(img_r[0])-right>=30:
        right = len(img_r[0]) - 2
        '''
        value = abs(int(img_r[leng_part][j] - img_r[leng_part][j - 1]))
        if len(max_value2)<5:
            if img_r[leng_part][j-1]<100:
                max_value2.append((value,j))
        else:
            if value>min(max_value2)[0]:
                max_value2.remove(min(max_value2))
                max_value2.append((value,j))
    max_value2.sort(key=takeSecond,reverse = True)
    print(max_value2)
    for j in range (len(max_value2)):

        value,pos=max_value2[j]
        print(img_r[leng_part][pos])
        print(img_r[leng_part][pos-1])
        flag=0
        for h in range(4, 8):  # 往后数三到7个数
            if pos - h >0:
                if abs(img_r[leng_part][pos - h] - img_r[leng_part][pos - h + 1]) > 10:
                    flag = 1
        if flag == 0:
            right = pos
            break
    '''
    '''
    if left>=len(img_r[0])//2:
        left=1
    if right<len(img_r[0])-len(img_r[0])//4:
        right=len(img_r[0])-2
    '''
    # print('left=',left,'right=',right)
    return left, right


def findhigh_low(leng_part, img_r: List[List[int]], boundary):
    max_value = []
    top = 0
    # print(leng_part)
    if len(img_r) > 100:
        outline = len(img_r) // 10
    else:
        outline = len(img_r) // 8
    for i in range(1, len(img_r) // 2):
        if img_r[i][leng_part] <= boundary <= img_r[i - 1][leng_part]:
            top = i
            break
    if abs(top - len(img_r) // 2) <= outline:
        top = 1
    if top>=30:
        top = 1
    # print('top=',top,'len=',len(img_r))
    '''
        value = abs(int(img_r[i][leng_part] - img_r[i - 1][leng_part]))
        if len(max_value)<5 :
            if  img_r[i][leng_part]<100:
                max_value.append((value, i))
        else:
            if abs(value) > min(max_value)[0]:
                max_value.remove(min(max_value))
                max_value.append((value, i))
    # 这里已经挑选出了最大的五个数
    #max_value.sort(key=takeSecond)  # 对第二个数进行排序，排序之后就开始进行
    #print(max_value)
    for i in range(len(max_value)):
        vlaue,pos  = max_value[i]
        flag = 0
        for h in range(4, 8):  # 往后数三到7个数
            if pos + h < len(img_r[0]):
                if abs(img_r[pos + h][leng_part] - img_r[pos + h - 1][leng_part]) > 10:
                    flag = 1
        if flag == 0:
            left = pos
            break
    '''
    bottom = len(img_r) - 1
    max_value2 = []
    flag = 0
    for j in range(len(img_r) - 1, len(img_r) - len(img_r) // 4, -1):
        if img_r[j][leng_part] >= boundary >= img_r[j - 1][leng_part]:
            flag = 1
            bottom = j
            break
    '''
    if flag==0:
        while j>len(img_r)//2:
            if img_r[j][leng_part] > 100 and img_r[j - 1][leng_part] < 100:
                bottom = j
                break
            j-=1
    '''
    if abs(bottom - len(img_r) // 2) <= outline:
        bottom = len(img_r) - 2
    if len(img_r)-bottom>=30:
        bottom = len(img_r) - 2
    # print(img_r[bottom][leng_part])
    # print(img_r[bottom-1][leng_part])
    # print(img_r[109])
    '''
        value = abs(int(img_r[j][leng_part] - img_r[j - 1][leng_part]))
        if len(max_value2) < 5:
            if img_r[j-1][leng_part]<100:
                max_value2.append((value, j))
        else:
            if value > min(max_value2)[0]:
                max_value2.remove(min(max_value2))
                max_value2.append((value, j))
    max_value2.sort(key=takeSecond,reverse = True)
    #print(max_value2)
    for j in range(len(max_value2)):
        value,pos  = max_value2[j]
        flag = 0
        for h in range(4, 8):  # 往后数三到7个数
            if pos - h > 0:
                if abs(img_r[pos - h][leng_part] - img_r[pos - h + 1][leng_part]) > 10:
                    flag = 1
        if flag == 0:
            right = pos
            break
            '''
    '''
    if top>=len(img_r)//2:
        top=1
    if bottom<len(img_r)-len(img_r)//4:
        bottom=len(img_r)-2
    '''
    # print ('top=',top,'bottom=', bottom)
    return (top, bottom)
    '''
    max_value = 0
    high = 0
    #print(leng_part)
    for i in range(1, len(img_r) // 2):
        value = int(img_r[i][leng_part] - img_r[i - 1][leng_part])
        print(i,'-',img_r[i][leng_part],':',value)
        if abs(value) > max_value:
            max_value = abs(value)
            high = i
    max_value = 0
    low = len(img_r) - 1
    for j in range(len(img_r) - 1, len(img_r) // 2, -1):
        value = int(img_r[j][leng_part] - img_r[j - 1][leng_part])
        if abs(value) > max_value:
            max_value = abs(value)
            low = j
    return (high,low)
    '''


def minus_max(list_all, flag):
    temp = list_all[0]
    p = 0
    while p < len(list_all):
        if list_all[p] == 0 or list_all[p] == flag:
            list_all.pop(p)
        else:
            p += 1
    if len(list_all) <= 0:
        return temp
    return round(mean(list_all))


def replace_image(frame_count, image, box):
    """
    对图片进行替换的操作。
    :param box:  这里的box编码应该是。x,y,w,h
    :param image:
    :return: 
    """
    image_trans_path = "E:/trans/trans3.png"
    image_trans = cv2.imread(image_trans_path)
    shrink = cv2.resize(image_trans, (box[3], box[4]), interpolation=cv2.INTER_AREA)
    # if 96 <= frame_count <= 98:
    #     cv2.imshow('shrink1', shrink)
    #     cv2.waitKey(0)
    temp = copy.copy(shrink)
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)
    # print(temp)
    lr = findlr(temp, 100)
    print('lr value =', len(lr), 'box[4]', box[4], 'box[3]', box[3])
    if len(lr) > 9:
        for i in range(box[3]):
            for j in range(box[4]):
                if lr[j][0] <= i <= lr[j][1]:
                    image[box[2] + j][box[1] + i] = shrink[j][i]
                # if temp[j][i][0] < 100:
                # else:\
    else:
        for i in range(box[3]):
            for j in range(box[4]):
                if temp[j][i][0] >= 110:
                    pass
                else:
                    image[box[2] + j][box[1] + i] = shrink[j][i]

    # cv2.imshow('image1', image)

    return image


def replace_image_doublelump(frame_count, image, box):
    """
    对图片进行替换的操作。
    :param box:  这里的box编码应该是。x,y,w,h
    :param image:
    :return:
    """
    image_trans_path = "E:/trans/trans3.png"
    image_trans_path2 = "E:/trans/trans3.png"
    image_trans = cv2.imread(image_trans_path)
    image_trans2 = cv2.imread(image_trans_path2)
    shrink = cv2.resize(image_trans, (box[3], box[4]), interpolation=cv2.INTER_AREA)
    shrink2 = cv2.resize(image_trans, (box[7], box[8]), interpolation=cv2.INTER_AREA)
    # if 96 <= frame_count <= 98:
    #     cv2.imshow('shrink1', shrink)
    #     cv2.waitKey(0)
    temp = copy.copy(shrink)
    temp2 = copy.copy(shrink2)
    temp = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)
    temp2 = cv2.cvtColor(temp2, cv2.COLOR_RGB2YUV)
    # print(temp)
    lr = findlr(temp, 100)
    lr2 = findlr(temp2, 100)
    print('lr value =', len(lr), 'box[4]', box[4], 'box[3]', box[3])
    if len(lr) > 9:
        for i in range(box[3]):
            for j in range(box[4]):
                if lr[j][0] <= i <= lr[j][1]:
                    image[box[2] + j][box[1] + i] = shrink[j][i]
                # if temp[j][i][0] < 100:
                # else:\
    else:
        for i in range(box[3]):
            for j in range(box[4]):
                if temp[j][i][0] >= 110:
                    pass
                else:
                    image[box[2] + j][box[1] + i] = shrink[j][i]
    if len(lr2) > 9:
        for i in range(box[7]):
            for j in range(box[8]):
                if lr2[j][0] <= i <= lr2[j][1]:
                    image[box[6] + j][box[5] + i] = shrink2[j][i]
                # if temp[j][i][0] < 100:
                # else:\
    else:
        for i in range(box[7]):
            for j in range(box[8]):
                if temp2[j][i][0] >= 110:
                    pass
                else:
                    image[box[6] + j][box[5] + i] = shrink2[j][i]

    # cv2.imshow('image1', image)

    return image


def findlr(image, boundary):
    """
    在图像替换的时候，寻找到他的边界。
    :param image:
    :param boundary:
    :return:
    @gxl 9/20
    """
    lr = []
    for j in range(len(image)):
        left = len(image[0])
        right = 0
        for i in range(1, len(image[0])):
            temp = image[j][i - 1][0]
            temp1 = image[j][i][0]
            if image[j][i - 1][0] >= boundary > image[j][i][0]:
                left = i
                break
        for i in range(len(image[0]) - 2, 0, -1):
            temp2 = image[j][i + 1][0]
            temp3 = image[j][i][0]
            if image[j][i + 1][0] >= boundary > image[j][i][0]:
                right = i
                break
        lr.append([left, right])
    return lr
