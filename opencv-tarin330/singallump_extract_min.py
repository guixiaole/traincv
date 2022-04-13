import cv2
import numpy as np
from typing import List
from numpy import *
from get_frame_singal import get_frame_singal
# 这个算法是根据rgb里面相差最大的数值确定中心点，然后按照中心点进行复制。
from opencvyolo_0502 import finln_out, findnet, yolov3_detect


def findcenter(image, boxes):
    # cv2.imshow('image1',image)
    frame_count = 74065

    # cv2.imwrite('F:/train_photo/' + "_%d.jpg" % frame_count, image)
    image_trans_path = "F:\\train_photo\\trans.png"
    image_trans = cv2.imread(image_trans_path)
    # image=cv2.imread(imagepath)
    (x, y) = (int(boxes[0][0]), int(boxes[0][1]))  # 框左上角
    (w, h) = (int(boxes[0][2]), int(boxes[0][3]))  # 框宽高

    x = x - 2
    y = y - 2

    w += 4
    h += 4

    crop = image[y:(h + y), x:(w + x)]  # 这只是框的大小
    # crop3=image[y+9:y+32,x+6:x+24]
    # cv2.imshow('crop3',crop3)
    # 先把R通道的给提取出来
    #
    cv2.cvtColor(crop, cv2.COLOR_RGB2YUV)
    # cv2.imshow('crop3',crop)
    img_r = [[0 for p in range(w)] for q in range(h)]
    for i in range(h):
        for j in range(w):
            img_r[i][j] = int(crop[i][j][0])
    # 获得R通道的矩阵之后，进行一个算法寻找

    # 这边去掉噪音的方法还是不行。

    for i in range (len(crop)):
        print(img_r[i])

    left, right = findleft_right(len(img_r) // 2, img_r)  # 寻找到左右之后
    left1, right1 = findleft_right(len(img_r) // 2 + 1, img_r)  # 寻找到左右之后
    left2, right2 = findleft_right(len(img_r) // 2 - 1, img_r)  # 寻找到左右之后
    left3, right3 = findleft_right(len(img_r) // 2 - 2, img_r)  # 寻找到左右之后
    left4, right4 = findleft_right(len(img_r) // 2 + 2, img_r)  # 寻找到左右之后

    left_list = [left, left1, left2, left3, left4]
    right_list = [right, right1, right2, right3, right4]
    mean_all = 8  # 大于某个数的时候就会删掉。

    right_mean = minus_max(right_list, len(img_r[0]) - 1)
    high, low = findhigh_low(len(img_r[0]) // 2, img_r)  # 寻找到左右之后
    high1, low1 = findhigh_low(len(img_r[0]) // 2 + 1, img_r)  # 寻找到左右之后
    high3, low3 = findhigh_low(len(img_r[0]) // 2 + 2, img_r)  # 寻找到左右之后
    high2, low2 = findhigh_low(len(img_r[0]) // 2 - 1, img_r)  # 寻找到左右之后
    high4, low4 = findhigh_low(len(img_r[0]) // 2 - 2, img_r)  # 寻找到左右之后
    high_list = [high, high1, high2, high3, high4]
    low_list = [low, low1, low2, low3, low4]

    # print(high_list,low_list)
    # print(left_list,right_list)

    high_mean = minus_max(high_list, len(img_r) - 1)

    low_mean = minus_max(low_list, len(img_r) - 1)
    left_mean = minus_max(left_list, len(img_r[0]) - 1)

    # 获取信号灯的大小之后，将代替换的照片进行压缩

    # cv2.imshow('image_trans',image_trans)
    shrink = cv2.resize(image_trans, (abs(right_mean - left_mean),
                                      abs(high_mean - low_mean)
                                      ),
                        interpolation=cv2.INTER_AREA)
    crop2 = crop[high_mean:low_mean, left_mean:right_mean]
    print(boxes)
    print('left_mean=', left_mean, 'right_mean=', right_mean)
    print('high_mean', high_mean, 'low_mean=', low_mean)
    # 图像压缩之后，应该开始复制。
    # cv2.imshow('image1', image)
    cv2.imshow('crop', crop)
    # cv2.imshow('crop2', crop2)
    i, j = 0, 0
    # print(shrink[0])
    # print(crop.shape)
    # print(shrink.shape)
    for i in range(abs(right_mean - left_mean) - 1):
        for j in range(abs(high_mean - low_mean) - 1):
            if shrink[j][i][0] < 250:
                image[y + j + high_mean][x + i + left_mean] = shrink[j][i]

    cv2.imshow('image',image)
    cv2.waitKey(0)
    ##现在可以试试进行替换看看
    return image


# 这里的算法有问题，需要修改
# 不能只找最大的数字，需要看周围的数字误差是不是小于5
# 算法思路：寻找出最大的5个差值，然后再根据坐标从小到大进行选择，越小的坐标权值越大。
def findleft_right(leng_part, img_r: List[List[int]]):  # 传入一个二维数组
    # 从二维数组中寻找到中心值
    max_value = 0
    left = 0
    if len(img_r[0]) > 100:
        outline = len(img_r[0]) // (2/5)
    else:
        outline = len(img_r[0]) // 3
    # print(leng_part)
    flag = 0
    for i in range(1, len(img_r[0]) // 2):

        # value = abs(int(img_r[leng_part][i] - img_r[leng_part][i - 1]))
        if img_r[leng_part][i] < 100 and img_r[leng_part][i - 1] > 100:
            left = i
            flag = 1
            break
    if abs(left - len(img_r[0]) // 2) <= outline:
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
        if img_r[leng_part][j] > 100 and img_r[leng_part][j - 1] < 100:
            flag = 1
            right = j
            break
    if abs(right - len(img_r[0]) // 2) <= outline:
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
    return (left, right)


def findhigh_low(leng_part, img_r: List[List[int]]):
    max_value = []
    top = 0
    # print(leng_part)
    if len(img_r) > 100:
        outline = len(img_r) // (2/5)
    else:
        outline = len(img_r) // 3
    for i in range(1, len(img_r) // 2):
        if img_r[i][leng_part] < 100 and img_r[i - 1][leng_part] > 100:
            top = i
            break
    if abs(top - len(img_r) // 2) <= outline:
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
        if img_r[j][leng_part] > 100 and img_r[j - 1][leng_part] < 100:
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


def minus_max(list_all, flag):  # flag为该list最大长度
    temp = list_all[0]
    p = 0
    while p < len(list_all):
        if list_all[p] == 0 or list_all[p] == flag:
            list_all.pop(p)
        else:
            p += 1
    if len(list_all) <= 0:
        return temp
    list_mean = int(mean(list_all))
    min_list = max(list_all)
    final = 0
    for i in range(len(list_all)):
        if list_all[i] - list_mean < min_list:
            min_list = list_all[i] - list_mean
            final = i

    '''
    list_copy=[]
    for i in range(len(list_all)):
        list_copy.append(list_all[i])
    list_mean=mean(list_all)
    p = 0
    while (p < len(list_all)):
        if abs(list_all[p] - list_mean) > mean_all:
            list_all.pop(p)
        else:
            p += 1
    if len(list_all)==0:
        if max(list_copy)-min(list_copy)>mean_all:
            temp=max(list_copy)
            list_copy.remove(temp)
            if max(list_copy) - min(list_copy) > mean_all:
                list_mean=(temp+max(list_copy))//2
            else:
                list_mean = int(mean(list_copy))
    else:
        list_mean = int(mean(list_all))
    '''
    return list_all[final]


# 排序的时候选定第二个数进行排序
def takeSecond(elem):
    return elem[1]


imagepath = ""
if __name__ == '__main__':
    imagepath = "F:\\train_photo_copy\_141077.jpg"
    # net=findnet()
    # ln,out=finln_out(net)
    image = cv2.imread(imagepath)
    # boxs,conf=yolov3_detect(image,net,ln,out)
    # print(boxs)
    boxs = [[714,474,49,79]]
    findcenter(image, boxs)
    '''
    all_txt=get_frame_singal()
    while len(all_txt)>0:
        (singalump_frame, boxes, conf) = all_txt.pop(0)
        print("当前帧",singalump_frame)
        image1="F:\\train_photo_copy\_"+str(singalump_frame)+".jpg"
        image = cv2.imread(image1)
        findcenter(image, boxes)


    '''
    '''
    [74, 60, 40] [36, 37, 21]
    [21, 17, 21] [29, 35, 34]
    '''
    '''
    73889;845,584,19,25,;0.9999500513076782,
    '''