from xml.dom.minidom import Document
import math
import cv2
import numpy as np
from typing import List
from numpy import *
from get_frame_singal import  get_frame_singal
import os
import re
#import Image
#这个算法是根据rgb里面相差最大的数值确定中心点，然后按照中心点进行复制。
from opencvyolo_0502 import finln_out,findnet,yolov3_detect
def findcenter(image,boxes):
    #cv2.imshow('image1',image)
    # cv2.imshow('image1',image)
    frame_count = 74065

    # cv2.imwrite('F:/train_photo/' + "_%d.jpg" % frame_count, image)

    image_trans_path = "F:\\train_photo\\trans.png"
    image_trans = cv2.imread(image_trans_path)

    # image=cv2.imread(imagepath)
    (x, y) = (int(boxes[0][0]), int(boxes[0][1]))  # 框左上角
    (w, h) = (int(boxes[0][2]), int(boxes[0][3]))  # 框宽高
    # (x, y) = (int(boxes[0]), int(boxes[1]))  # 框左上角
    # (w, h) = (int(boxes[2]), int(boxes[3]))  # 框宽高

    crop = image[y:(h + y), x:(w + x)]
    # cv2.waitKey(0)
    #cv2.imshow('first', crop)
    # if w<100 and h<100:
    x = x - 1
    y = y - 1

    w += 5
    h += 5

    crop = image[y:(h + y), x:(w + x)]
    #cv2.imshow('crop3', crop)
    #cv2.waitKey(0)
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
    '''
    for i in range(len(crop)):
        print(img_r[i])
    '''
    boundary = 70
    left, right = findleft_right(len(img_r) // 2, img_r, boundary)  # 寻找到左右之后
    left1, right1 = findleft_right(len(img_r) // 2 + 1, img_r, boundary)  # 寻找到左右之后
    left2, right2 = findleft_right(len(img_r) // 2 - 1, img_r, boundary)  # 寻找到左右之后
    left3, right3 = findleft_right(len(img_r) // 2 - 2, img_r, boundary)  # 寻找到左右之后
    left4, right4 = findleft_right(len(img_r) // 2 + 2, img_r, boundary)  # 寻找到左右之后

    left_list = [left, left1, left2, left3, left4]
    right_list = [right, right1, right2, right3, right4]
    mean_all = 8  # 大于某个数的时候就会删掉。

    # right_mean=minus_max(right_list,len(img_r[0]) - 1)
    right_mean = int(round(mean(right_list)))
    high, low = findhigh_low(len(img_r[0]) // 2, img_r, boundary)  # 寻找到左右之后
    high1, low1 = findhigh_low(len(img_r[0]) // 2 + 1, img_r, boundary)  # 寻找到左右之后
    high3, low3 = findhigh_low(len(img_r[0]) // 2 + 2, img_r, boundary)  # 寻找到左右之后
    high2, low2 = findhigh_low(len(img_r[0]) // 2 - 1, img_r, boundary)  # 寻找到左右之后
    high4, low4 = findhigh_low(len(img_r[0]) // 2 - 2, img_r, boundary)  # 寻找到左右之后
    high_list = [high, high1, high2, high3, high4]
    low_list = [low, low1, low2, low3, low4]
    print(right_mean)
    # print(high_list,low_list)
    # print(left_list,right_list)

    # high_mean = minus_max(high_list,len(img_r) - 1)
    high_mean = int(round(mean(high_list)))

    # low_mean = minus_max(low_list,len(img_r) - 1)
    low_mean = int(round(mean(low_list)))
    # left_mean = minus_max(left_list,len(img_r[0]) - 1)
    left_mean = int(round(mean(left_list)))

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
    '''
    print(y + high_mean, x + left_mean)
    for i in range(abs(right_mean - left_mean) - 1):
        for j in range(abs(high_mean - low_mean) - 1):
            if shrink[j][i][0] < 100:
                image[y + j + high_mean][x + i + left_mean] = shrink[j][i]

    cv2.imshow('image', image)
    cv2.waitKey(0)
    '''
    ##现在可以试试进行替换看看
    # return image
    # 将坐标进行保存然后进行储存

    corrdinate = [x + left_mean, y + high_mean, abs(right_mean - left_mean) + 1, abs(high_mean - low_mean) + 1]
    corrdinate_test = [left_mean, high_mean, abs(right_mean - left_mean) + 1, abs(high_mean - low_mean) + 1]
    print(corrdinate_test)
    # print(corrdinate)
    return corrdinate


def findcenter_test(image,boxes):
    #cv2.imshow('image1',image)
    frame_count=74065

    #cv2.imwrite('F:/train_photo/' + "_%d.jpg" % frame_count, image)

    image_trans_path = "F:\\train_photo\\trans.png"
    image_trans = cv2.imread(image_trans_path)

    #image=cv2.imread(imagepath)
    (x, y) = (int(boxes[0]), int(boxes[1]))  # 框左上角
    (w, h) = (int(boxes[2]), int(boxes[3]))  # 框宽高
    #(x, y) = (int(boxes[0]), int(boxes[1]))  # 框左上角
    #(w, h) = (int(boxes[2]), int(boxes[3]))  # 框宽高

    box_truth=[253, 47, 166, 208]
    crop = image[y:(h + y), x:(w + x)]
    crop2 = image[box_truth[1]:(box_truth[3] + box_truth[1]), box_truth[0]:(box_truth[2] + box_truth[0])]
    #cv2.waitKey(0)
    cv2.imshow('first',crop)
    cv2.imshow('truth',crop2)
    #if w<100 and h<100:
    x=x-1
    y=y-1

    w+=5
    h+=5

    crop = image[y:(h + y), x:(w + x)]
    cv2.imshow('crop3', crop)
    cv2.waitKey(0)
    #在这里设置了一个问题，这里必须要改
    if x<0:
        x=0
    if y<0:
        y=0
    crop = image[y:(h + y), x:(w + x)]  # 这只是框的大小
    #crop3=image[y+9:y+32,x+6:x+24]


    #先把R通道的给提取出来
    #
    crop=cv2.cvtColor(crop,cv2.COLOR_RGB2YUV)
    #cv2.imshow('crop3',crop)
    img_r=[[0 for p in range(w)]for q in range (h)]
    for i in range (h):
        for j in range(w):
            img_r[i][j]=int(crop[i][j][0])

    #获得R通道的矩阵之后，进行一个算法寻找

    #这边去掉噪音的方法还是不行。
    #print('kaishi ')

    for i in range (len(crop)):
        print(img_r[i])

    boundary=70
    left,right=findleft_right(len(img_r)//2,img_r,boundary)#寻找到左右之后
    left1,right1=findleft_right(len(img_r)//2+1,img_r,boundary)#寻找到左右之后
    left2,right2=findleft_right(len(img_r)//2-1,img_r,boundary)#寻找到左右之后
    left3,right3=findleft_right(len(img_r)//2-2,img_r,boundary)#寻找到左右之后
    left4,right4=findleft_right(len(img_r)//2+2,img_r,boundary)#寻找到左右之后

    left_list=[left,left1,left2,left3,left4]
    right_list=[right,right1,right2,right3,right4]
    mean_all=8#大于某个数的时候就会删掉。


    #right_mean=minus_max(right_list,len(img_r[0]) - 1)
    right_mean=int(round(mean(right_list)))
    high,low=findhigh_low(len(img_r[0])//2,img_r,boundary)#寻找到左右之后
    high1,low1=findhigh_low(len(img_r[0])//2+1,img_r,boundary)#寻找到左右之后
    high3,low3=findhigh_low(len(img_r[0])//2+2,img_r,boundary)#寻找到左右之后
    high2,low2=findhigh_low(len(img_r[0])//2-1,img_r,boundary)#寻找到左右之后
    high4,low4=findhigh_low(len(img_r[0])//2-2,img_r,boundary)#寻找到左右之后
    high_list=[high,high1,high2,high3,high4]
    low_list=[low,low1,low2,low3,low4]
    print(right_mean)
    #print(high_list,low_list)
    #print(left_list,right_list)

    #high_mean = minus_max(high_list,len(img_r) - 1)
    high_mean = int(round(mean(high_list)))

    #low_mean = minus_max(low_list,len(img_r) - 1)
    low_mean = int(round(mean(low_list)))
    #left_mean = minus_max(left_list,len(img_r[0]) - 1)
    left_mean = int(round(mean(left_list)))

    #获取信号灯的大小之后，将代替换的照片进行压缩

    #cv2.imshow('image_trans',image_trans)

    shrink = cv2.resize(image_trans, (abs(right_mean-left_mean),
                                    abs(high_mean-low_mean)
                                      ),
                        interpolation=cv2.INTER_AREA)
    crop2=crop[high_mean:low_mean,left_mean:right_mean]

    #print(boxes)
    #print('left_mean=',left_mean,'right_mean=',right_mean)
    #print('high_mean',high_mean,'low_mean=',low_mean)
    #图像压缩之后，应该开始复制。
    #cv2.imshow('image1', image)
    #cv2.imshow('crop', crop)
    #cv2.imshow('crop2', crop2)
    i,j=0,0
    #print(shrink[0])
    #print(crop.shape)
    #print(shrink.shape)

    print(y+high_mean,x+left_mean)
    for i in range(abs(right_mean-left_mean)-1):
        for j in range(abs(high_mean-low_mean)-1):
            if shrink[j][i][0]<100:
                image[y +j+high_mean][x + i+left_mean] = shrink[j][i]

    cv2.imshow('image',image)
    cv2.waitKey(0)
    ##现在可以试试进行替换看看
    #return image
    #将坐标进行保存然后进行储存

    corrdinate=[x+left_mean,y+high_mean,abs(right_mean-left_mean)+1,abs(high_mean-low_mean)+1]
    corrdinate_test=[left_mean,high_mean,abs(right_mean-left_mean)+1,abs(high_mean-low_mean)+1]
    print(corrdinate_test)
    #print(corrdinate)
    return corrdinate

#这里的算法有问题，需要修改
#不能只找最大的数字，需要看周围的数字误差是不是小于5
#算法思路：寻找出最大的5个差值，然后再根据坐标从小到大进行选择，越小的坐标权值越大。
def findleft_right(leng_part,img_r:List[List[int]],boundary):#传入一个二维数组
    #从二维数组中寻找到中心值
    max_value=0
    left=0
    if len(img_r[0])>100:
        outline=len(img_r[0])//(5/2)
    else:
        outline=len(img_r[0])//(24/7)
    #print(leng_part)
    flag = 0
    for i in range (1,len(img_r[0])//2):

        #value = abs(int(img_r[leng_part][i] - img_r[leng_part][i - 1]))
        if img_r[leng_part][i]<boundary and img_r[leng_part][i-1]>boundary:
            left=i
            flag=1
            break
    if abs(left - len(img_r[0]) // 2)<=outline:
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
    #这里已经挑选出了最大的五个数
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
    right=len(img_r[0])-1
    max_value2=[]
    flag=0
    for j in range(len(img_r[0]) - 1, len(img_r[0]) // 2, -1):
        if img_r[leng_part][j]>boundary and img_r[leng_part][j-1]<boundary:
            flag=1
            right=j
            break
    if abs(right-len(img_r[0])//2)<=outline:
        right=len(img_r[0])-2
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
    #print('left=',left,'right=',right)
    return (left,right)
def findhigh_low(leng_part,img_r:List[List[int]],boundary):

    max_value = []
    top = 0
    #print(leng_part)
    if len(img_r)>100:
        outline=len(img_r)//(5/2)
    else:
        outline=len(img_r)//(24/7)
    for i in range(1, len(img_r) // 2):
        if img_r[i][leng_part]<boundary and img_r[i-1][leng_part]>boundary:
            top=i
            break
    if abs(top - len(img_r) // 2) <=outline:
        top = 1
    #print('top=',top,'len=',len(img_r))
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
    flag=0
    for j in range(len(img_r) - 1, len(img_r)-len(img_r)//4, -1):
        if img_r[j][leng_part]>boundary and img_r[j-1][leng_part]<boundary:
            flag=1
            bottom=j
            break
    '''
    if flag==0:
        while j>len(img_r)//2:
            if img_r[j][leng_part] > 100 and img_r[j - 1][leng_part] < 100:
                bottom = j
                break
            j-=1
    '''
    if abs(bottom-len(img_r)//2)<=outline:
        bottom=len(img_r)-2
    #print(img_r[bottom][leng_part])
    #print(img_r[bottom-1][leng_part])
    #print(img_r[109])
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
    return(top, bottom)
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
def minus_max(list_all,flag):
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
def minus_max1(list_all,flag):#flag为该list最大长度
    temp=list_all[0]
    p=0
    while p<len(list_all):
        if list_all[p]==0 or list_all[p]==flag:
            list_all.pop(p)
        else:
            p+=1
    if len(list_all)<=0:
        return temp
    list_mean=int(mean(list_all))
    min_list=max(list_all)
    final=0
    for i in range (len(list_all)):
        if list_all[i]-list_mean<min_list:
            min_list=list_all[i]-list_mean
            final=i

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
#排序的时候选定第二个数进行排序
def takeSecond(elem):
    return elem[1]
def replace_frame_smooth():
    '''
     使用窗口进行平滑处理
    :return:
    '''
    frame_detect = []
    filepath = "corrdinate_test.txt"
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



    #print(frame_detect)
    windows=3#使用窗口设置，前后各3个，总共7个
    for j in range(1,len(frame_detect[0])):#4个坐标进行检测
        # 先检查是递减还是递增，递减flag=0，递增则是1
        temp=frame_detect[0][j]-frame_detect[len(frame_detect)//2][j]
        if temp>0:
            flag=0
        elif temp<0:
            flag=1
        else:
            temp = frame_detect[0][j] - frame_detect[len(frame_detect) //4*3][j]
            if temp>0:
                flag=0
            else:
                flag=1
    #确定递减还是递增
        for i in range (len(frame_detect)-20):#最后20帧不进行平滑
            sum=0
            if i >=3 and i<len(frame_detect)-3:#前窗口是有的。
                sum+=frame_detect[i-3][j]+frame_detect[i-2][j]+frame_detect[i-1][j]
                sum += frame_detect[i + 1][j] + frame_detect[i + 2][j] + frame_detect[i + 3][j] + frame_detect[i][j]
            else:
                if i==1:
                    sum+=frame_detect[0][j]
                    sum += frame_detect[i + 1][j] + frame_detect[i + 2][j] + frame_detect[i + 3][j] + frame_detect[i][j]
                elif i==2:
                    sum+=frame_detect[0][j]+frame_detect[1][j]
                    sum+=frame_detect[i+1][j]+frame_detect[i+2][j]+frame_detect[i+3][j]+frame_detect[i][j]
                elif i==0:
                    sum += frame_detect[i + 1][j] + frame_detect[i + 2][j] + frame_detect[i + 3][j] + frame_detect[i][j]
                elif i==len(frame_detect)-1:
                    sum += frame_detect[i -1][j] + frame_detect[i -2][j] + frame_detect[-3][j] + frame_detect[i][j]
                elif i == len(frame_detect) - 2:
                    sum += frame_detect[i -1][j] + frame_detect[i -2][j] + frame_detect[-3][j] + frame_detect[i][j]+frame_detect[i+1][j]

                elif i == len(frame_detect) - 3:
                    sum += frame_detect[i - 1][j] + frame_detect[i - 2][j] + frame_detect[-3][j] + frame_detect[i][j] + \
                           frame_detect[i + 1][j]+frame_detect[i+2][j]
                #这边滑动窗口还是有问题啊。
            #但是滑动之后，应该就不会特别的明显。
            #然后就是求平均了
            avera=frame_detect[i][j]

            if i >=3 and i<len(frame_detect)-3:
                avera=sum//7#求平均
            elif i==1 or  i==len(frame_detect)-2:
                avera=sum//5
            elif i==0 or i==len(frame_detect)-1:
                avera=sum//4
            elif i==2 or i==len(frame_detect)-3:
                avera=sum//6
            if i!=0:
                if flag==1 and avera<frame_detect[i-1][j]:#递增的情况下。
                    avera=frame_detect[i-1][j]
                if flag==0 and avera>frame_detect[i-1][j]:
                    avera=frame_detect[i-1][j]
            frame_detect[i][j]=avera

    for i in range(len(frame_detect)):
        print(frame_detect[i])


    return frame_detect



def replace_frame_smooth1():
    '''
    将获取图片替换进行平滑处理。然后将帧数与坐标进行存储。
    :return:
    '''
    frame_detect=[]
    filepath="corrdinate.txt"
    #已经将坐标存储在了txt文件中。将其从txt文件中获取之后，进行平滑处理
    for txt in open(filepath):
        all = txt.strip().split(";")
        frame=int(all[0])
        box=all[1].strip().split(",")
        box.pop(len(box)-1)
        boxs=[]
        for i in range(len(box)):
            boxs.append(int(box[i]))
        boxs.insert(0,frame)
        frame_detect.append(boxs)
    #已经获取了所有的帧和目标检测出来的图片
    for j in range(1,len(frame_detect[0])):#4个坐标进行检测
        # 先检查是递减还是递增，递减flag=0，递增则是1
        temp=frame_detect[0][j]-frame_detect[len(frame_detect)//2][j]
        if temp>0:
            flag=0
        elif temp<0:
            flag=1
        else:
            temp = frame_detect[0][j] - frame_detect[len(frame_detect) //4*3][j]
            if temp>0:
                flag=0
            else:
                flag=1
        for i in range(1,len(frame_detect)-15):
            #判断是在递减或递增这个趋势里
            if flag==0:
                if frame_detect[i][j]-frame_detect[i-1][j]>0:
                    #表示有问题
                    temp_pos=i+1
                    temp= frame_detect[temp_pos][j]-frame_detect[i-1][j]
                    length=1
                    while (temp_pos<=len(frame_detect)-15 and temp>0):
                        length+=1
                        temp_pos=temp_pos+1

                        temp = frame_detect[temp_pos][j] - frame_detect[i - 1][j]
                    #找到了其中的数
                    for h in range (0,length):#进行修改
                        frame_detect[i+h][j]=(frame_detect[temp_pos][j]+frame_detect[i-1][j])//2
            elif flag==1:
                if frame_detect[i][j]-frame_detect[i-1][j]<0:
                    #表示有问题
                    temp_pos=i+1
                    temp= frame_detect[temp_pos][j]-frame_detect[i-1][j]
                    length = 1
                    while (temp_pos<len(frame_detect)-15 and  temp<0):
                        temp_pos+=1
                        temp = frame_detect[temp_pos][j] - frame_detect[i - 1][j]
                        length += 1
                    #找到了其中的数
                    for h in range (0,length):#进行修改
                        frame_detect[i+h][j]=(frame_detect[temp_pos][j]+frame_detect[i-1][j])//2


    for i in range(len(frame_detect)):
        print(frame_detect[i])

    return frame_detect
def replace_image(image,box):
    #这里仅仅只是将源文件传进来，然后再进行替换
    image_trans_path = "trans2.png"
    image_trans = cv2.imread(image_trans_path)
    shrink = cv2.resize(image_trans, (box[2],box[3]),
                        interpolation=cv2.INTER_AREA)
    temp=shrink
    temp=cv2.cvtColor(temp,cv2.COLOR_RGB2YUV)
    for i in range(box[2]-1):
        for j in range(box[3] - 1):
            if temp[j][i][0] < 100 or temp[j][i][1] < 100 or temp[j][i][2] < 100:
                image[box[1]+j][box[0] +i] = shrink[j][i]
    return image
def find_lump_center_pos(boxs,image,last_pos):
    '''
    找到信号灯的中心点，和find_crop_center_pos事从属关系。
    :param boxs:信号灯的坐标
    :param image:图片
    :param last_pos:上一帧的中心点
    :return:
    '''
   #此方法用来寻找到关于灯颜色的具体中间值。
    center_x = (boxs[0] + boxs[2]) // 2
    center_y = (boxs[1] + boxs[3]) // 2
    crop = image[boxs[1]:boxs[1] + boxs[3], boxs[0]:boxs[0] + boxs[2]]  # 将这个裁剪出来
    cv2.imshow('crop', crop)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2YUV)  # RGB转换为YUV
    #找到几个点，来进行寻找中心点。
    left_center_pos=boxs[3]//2
    right_center_pos=boxs[2]//2
    #假设这两个点开始寻找。corrdinate
    ipx_threshold=65#灯颜色像素点阈值的数
    boundary=find_crop_center_pos(crop,boxs,left_center_pos,right_center_pos,ipx_threshold)
    return boundary
def find_crop_center_pos(crop,boxs,left_center_pos,right_center_pos,ipx_threshold):
    #这个方法取消之前的递归想法
    #主要通过采取多个样点，然后进行比较，选出最有可能是灯的颜色的点
    #TODO:如何评价呢？评价函数如何去写呢？
    caiyang=[]#进行采样的点。

    if crop[left_center_pos][right_center_pos][0]>=ipx_threshold:
        caiyang.append([left_center_pos,right_center_pos])
    #对多个进行采样。
    flage=2
    flage_minus=0
    while flage<=4:
        if crop[left_center_pos+flage][right_center_pos+flage_minus][0]>=ipx_threshold:
            caiyang.append([left_center_pos+flage,right_center_pos+flage_minus])

        if crop[left_center_pos+flage_minus][right_center_pos+flage][0]>=ipx_threshold:
            caiyang.append([left_center_pos+flage_minus,right_center_pos+flage])

        if crop[left_center_pos+flage][right_center_pos+flage][0]>=ipx_threshold:
            caiyang.append([left_center_pos+flage,right_center_pos+flage])

        if crop[left_center_pos-flage][right_center_pos-flage_minus][0]>=ipx_threshold:
            caiyang.append([left_center_pos-flage,right_center_pos-flage_minus])

        if crop[left_center_pos-flage][right_center_pos-flage][0]>=ipx_threshold:
            caiyang.append([left_center_pos-flage,right_center_pos-flage])

        if crop[left_center_pos-flage_minus][right_center_pos-flage][0]>=ipx_threshold:
            caiyang.append([left_center_pos-flage_minus,right_center_pos+flage])

        flage_minus+=2
        flage+=2
    #获得想要的几个点
    max_boundary=[]
    cost_boundary=0
    for i in range(len(caiyang)):
        boundary=find_boundary(crop,boxs,caiyang[i])
        if boundary[2]*boundary[3]>cost_boundary:
            max_boundary=boundary
    return max_boundary
def final_frame_detect():
    '''
    此方法主要是解决最后几帧漂移的问题。
    :return:
    '''

def find_crop_center_pos1(crop,boxs,left_center_pos,right_center_pos,ipx_threshold,last_pos,flag,jianju):
    '''
    主要是找到信号灯颜色的中心点
    :param crop: 剪辑后的视频
    :param boxs: 边框
    :param left_center_pos: 假定中心点
    :param right_center_pos:假定中心点
    :param ipx_threshold:阈值
    :param last_pos: 上一次的
    :param flag:加减
    :param jianju: 加减的间距
    :return:
    '''
    left_center_right_boundary = 0  # 使用left_center
    left_center_left_boundary = 0  # 使用left_center
    right_center_left_boundary = 0  # 使用right_center
    right_center_right_boundary = 0  # 使用right_center
    if jianju > 4:
        # 说明小范围搜索都没有找到合适的点：
        if len(last_pos) == 2:  # 说明上一个点还有
            if crop[last_pos[0]][last_pos[1]][0] >=ipx_threshold:
                left_center_pos_=last_pos[0]
                right_center_pos_=last_pos[1]
                for i in range(left_center_pos_, boxs[3] - 1):
                    if crop[i][right_center_pos_][0] >= ipx_threshold:
                        if crop[i + 1][right_center_pos_][0] < ipx_threshold:
                            left_center_right_boundary = i
                            break
                for j in range(left_center_pos_, 0, -1):
                    if crop[j][right_center_pos_] > ipx_threshold :
                        if crop[j + 1][right_center_pos_][0] < ipx_threshold :
                            left_center_left_boundary = j
                left_center_pos = (left_center_left_boundary + left_center_right_boundary + 1) // 2
                for i in range(right_center_pos_, boxs[2] - 1):
                    if [left_center_pos_][i][0] >= ipx_threshold:
                        if [left_center_pos_][i + 1][0] < ipx_threshold:
                            right_center_right_boundary = i
                            break
                for j in range(right_center_pos_, 0, -1):
                    if [left_center_pos_][j][0] >= ipx_threshold:
                        if [left_center_pos_][j + 1][0] < ipx_threshold:
                            right_center_left_boundary = j
                            break
                right_center_pos = (right_center_right_boundary - right_center_left_boundary + 1) // 2

            return (left_center_pos, right_center_pos)
        else:
            #说明找了很久都没有找到，那就返回中心点吧
            return (left_center_pos, right_center_pos)

    if flag > 0:
        jianju = -jianju
    left_center_pos_=left_center_pos
    right_center_pos_=right_center_pos
    left_center_pos_+=jianju
    right_center_pos_+=jianju
    if left_center_pos_>=boxs[3]-1:
        left_center_pos_=boxs[3]-2
    if right_center_pos_>=boxs[2]-1:
        right_center_pos_=boxs[2]-2
    if crop[left_center_pos_][right_center_pos_][0] >= ipx_threshold:
        # 假设这点是阈值的话
        # 这一点主要是找到中点，假设
        # 假如这点找的不是中心点
        left_center_pos=left_center_pos_
        right_center_pos=right_center_pos_

        for i in range(left_center_pos_, boxs[3] - 1):
            if crop[i][right_center_pos_][0] >= ipx_threshold:
                if crop[i + 1][right_center_pos_][0] < ipx_threshold:
                    left_center_right_boundary = i
                    break
        for j in range(left_center_pos_, 0, -1):
            if crop[j][right_center_pos_][0] > ipx_threshold :
                if crop[j + 1][right_center_pos_][0] < ipx_threshold :
                    left_center_left_boundary = j
                    break
        left_center_pos = (left_center_left_boundary + left_center_right_boundary + 1) // 2
        print(left_center_pos)
        left_center_pos_=left_center_pos
        for h in range(right_center_pos_, boxs[2] - 1):
            if crop[left_center_pos_][h][0] >= ipx_threshold:
                if crop[left_center_pos_][h + 1][0] < ipx_threshold:
                    right_center_right_boundary = h
                    break
        for q in range(right_center_pos_, 0, -1):
            if crop[left_center_pos_][q][0] >= ipx_threshold:
                if crop[left_center_pos_][q + 1][0] < ipx_threshold:
                    right_center_left_boundary = q
                    break

      
        right_center_pos = (right_center_right_boundary - right_center_left_boundary + 1) // 2

    else:
        left_center_pos,right_center_pos=find_crop_center_pos(crop,boxs,left_center_pos,right_center_pos,ipx_threshold,last_pos,-flag,abs(jianju)+1)

    return (left_center_pos,right_center_pos)

def replace_lump_pos(boxs,image):
    '''
     寻找到灯中颜色的点 那点的坐标，然后进行相对应的替换
     思路是从中间开始从上往下开始找。
    :param boxs:  这是精细化后的信号灯的坐标
    :param image: 传进来的图片进行替换
    :return:

    '''
    center_x=(boxs[0]+boxs[2])//2
    center_y=(boxs[1]+boxs[3])//2

    #这个中心点不是真正的中心点，需要找到真正的中心点
    #找到了中心点，接下来就是从中心点开始向四周扩散，找到灯的边缘。
    #找到的这点是属于图片的中心的点。
    #先把这点打印出来看看是什么
    crop = image[boxs[1]:boxs[1] + boxs[3], boxs[0]:boxs[0] + boxs[2]]#将这个裁剪出来
    cv2.imshow('crop',crop)
    crop = cv2.cvtColor(crop, cv2.COLOR_RGB2YUV)#RGB转换为YUV
    '''
    flag = 0
    left_pos=boxs[3]//2
    if crop[left_pos][boxs[2]//2][0]<90:#假设找的那点不是中心点
        find_flag=1

        i=2
        temp_plus=left_pos
        temp_minus=left_pos
        while(find_flag):
            #先从上下开始找
            pos = boxs[2] // 2
            temp=pos
            while (pos<boxs[2]//(3/2)):#先往下找到中心点
                if crop[left_pos][pos][0]<90:
                    pos+=1
                else:
                    flag=1
                    break
            if flag==1:
                break
            else:
                pos=temp
                while (pos > boxs[2] // 3  ):  # 先往下找到中心点
                    if crop[left_pos][pos][0] < 90:
                        pos -= 1
                    else:
                        flag = 1
                        break
                if flag==1:
                    break
                else:
                    if i==1:
                        left_pos=temp_minus-1
                        temp_minus=left_pos
                        i=2
                    else:
                        left_pos=temp_plus+1
                        temp_plus=left_pos
                        if i==2:
                            i+=1
                        else:
                            i=1
    else:
        pos=boxs[2]//2
    #从中间往上下左右找
    #cv2.imshow('image', image)
    print(boxs)
    print(left_pos,pos)
    '''
    lump_box=find_lump_center_pos(boxs,image,[])

   # print('left_pos',left_pos,'right_pos',right_pos)
    img_r = [[0 for p in range(int(boxs[2]))] for q in range(boxs[3])]

    for p in range(boxs[3]):
        for w in range(boxs[2]):
            img_r[p][w] = int(crop[p][w][0])

    for i in range(len(crop)):
        print(img_r[i])

    #lump_box=find_boundary(crop,boxs,[left_pos,right_pos])
   # lump_box=[11,9,8,8]
    print(lump_box)
    crop2=image[boxs[1]+lump_box[1]:boxs[1]+lump_box[1]+lump_box[3], boxs[0]+lump_box[0]:boxs[0]+lump_box[0]+lump_box[2]]
    cv2.imshow('crop2',crop2)
    crop2=cv2.cvtColor(crop2,cv2.COLOR_RGB2YUV)

    print("middle")
    for i in range (len(crop2)):
        for j in range(len(crop2[0])):
            print(crop2[i][j][0],end=" ")
        print("\n")


    for i in range (lump_box[2]):
        for j in range(lump_box[3]):
            if crop[lump_box[1]+j][i+lump_box[0]][0]>=70:#是可以替换的
                image[lump_box[1]+j+boxs[1]][lump_box[0]+i+boxs[0]][0]=0
                image[lump_box[1]+j+boxs[1]][lump_box[0]+i+boxs[0]][1]=0
                image[lump_box[1]+j+boxs[1]][lump_box[0]+i+boxs[0]][2]=255


    #cv2.imshow('crop', crop)

    cv2.imshow('image1',image)
    cv2.waitKey(0)

    return (lump_box)
    '''
    document = Document()
    p = document.add_paragraph('')

    #with open("corrdinate2.txt", "a") as file:
    
        #file.write("now frame_count:"+str(frame_count)+"\n")
        #file.write("box:"+str(boxs[0])+","+str(boxs[1])+","+str(boxs[2])+","+str(boxs[3])+"\n")
    run=p.add_run("now frame_count:"+str(frame_count))
    run=p.add_run()
    run.add_break()
    run=p.add_run("box:"+str(boxs[0])+","+str(boxs[1])+","+str(boxs[2])+","+str(boxs[3]))
    run=p.add_run()
    run.add_break()
    for i in range(boxs[2]):
        for j in range(boxs[3]):
            #file.write(str(image[i][j][0])+" ")
            run = p.add_run(str(image[i][j][0])+" ")
            if i >boxs[2]//4 and i <boxs[2]//(4/3) and j >boxs[3]//4 and j<boxs[3]//(4/3):
                run.font.color.rgb = RGBColor(255, 0, 0)

            #r = run._element
            #r.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
            #file.write("\n")
        run = p.add_run()
        run.add_break()
    document.save('corrdinate2.docx')  # 关闭保存word
    '''
    #crop = image[boxs[1]:(boxs[3] + boxs[1]), boxs[0]:(boxs[2] + boxs[0])]
    #cv2.imshow('crop',crop)
    #cv2.waitKey(0)
    # 这只是框的大小
def find_boundary1(image,pos):
    '''
    寻找边界，已被废弃
    :param image:
    :param pos:
    :return:
    '''
    #首先向上
    x,y=pos[0],pos[1]
    upper,lower,left,right=0,0,0,0
    while x>len(image[0])//4:
        left = x
        if image[x][pos[1]][0]>=90 and image[x-1][pos[1]][0]<90:
            break
        else:
            x-=1
    left-=1
    x=pos[0]
    while x<len(image[0])//(4/3):
        right = x
        if image[x][pos[1]][0]>=90 and image[x+1][pos[1]][0]<90:
            break
        else:
            x+=1
    right+=1
    while y>len(image)//4:
        upper=y
        if image[pos[0]][y][0]>=90 and image[pos[0]][y-1][0]<90:
            break
        else:
            y-=1
    upper-=1
    y=pos[1]
    while y<(len(image)//(4/3)):
        lower=y
        if image[pos[0]][y][0]>=90 and image[pos[0]][y+1][0]<90:
            break
        else:
            y+=1
    lower+=1
    return [upper,left,abs(lower-upper+1),abs(right-left+1)]
def find_boundary(crop,boxs,pos):
    x, y = pos[0], pos[1]
    upper, lower, left, right = x, x, y, y
    flag_grid=[[False for _ in range(boxs[2]) ] for h in range(boxs[3])]
    queen=[]
    queen.append((x,y))
    flag_grid[x][y] = True
    while len(queen)>0:

        num = queen.pop(0)
        mins = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        for dx, dy in mins:
            nx, ny = num[0] + dx, num[1] + dy
            if 0 <= nx < boxs[3] and 0 <= ny < boxs[2] and crop[nx][ny][0] >=90 and flag_grid[nx][ny] == False:
                flag_grid[nx][ny] = True
                queen.append((nx, ny))
                print(crop[nx][ny][0])
                if nx<upper:
                    upper=nx
                if nx>lower:
                    lower=nx
                if ny<left:
                    left=ny
                if ny>right:
                    right=ny

    return [upper,left,abs(right-left)+1,abs(lower-upper)+1]
def store_coordinate():
    '''
    将视频有信号灯的进行存储
    :return:
    '''
    videoName="D:\火车项目\马皇--钦州港_2.MP4"
    cap = cv2.VideoCapture(videoName)
    order_frame=191613
    cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    frame_count=order_frame
    frame_singal = get_frame_singal()  # 指的是获取信号灯的那一帧
    while (cap.isOpened() == True):
        print("当前帧：", frame_count)
        ret, frame = cap.read()
        if ret:
            if len(frame_singal) > 0 and frame_count == frame_singal[0][0]:
                (singalump_frame, boxes, conf) = frame_singal.pop(0)  # 这表明获取到了这一帧的照片了
                frame = findcenter(frame, boxes)#在这里代表获取坐标
                with open("corrdinate_test.txt", "a") as file:
                    file.write(str(frame_count)+";")
                    for i in range (len(frame)):
                        file.write(str(frame[i])+",")
                    file.write("\n")
        frame_count+=1
if __name__ == '__main__':
    #imagepath = "F:\\train_photo_copy\_74089.jpg"
    #replace_frame_smooth()
    #net=findnet()
    #ln,out=finln_out(net)
    #image = cv2.imread(imagepath)
   # boxs,conf=yolov3_detect(image,net,ln,out)
    #print(boxs)

    #boxs = [[221,18,182,229]]
    #box1=findcenter(image,boxs)
    #image=replace_image(image,box1)
    #cv2.imshow('image',image)
    #cv2.waitKey(0)
    #findcenter(image,boxs)
    #replace_frame_smooth()
    #store_coordinate()

    image_trans_path = "F:\\train_photo\\trans2.png"
    image_trans = cv2.imread(image_trans_path)
    shrink = cv2.resize(image_trans, (20,22),interpolation=cv2.INTER_AREA)
    image_trans = cv2.cvtColor(image_trans, cv2.COLOR_RGB2YUV)
    for i in range(len(image_trans[0][0])):
        for j in range(len(image_trans[0])):
            for h in range(len(image_trans)):
                print(image_trans[h][j][i],end=' ')
            print(" ")
    print(image_trans[:,:,0])
    print(image_trans[:,:,1])
    print(image_trans[:,:,2])
    '''
    image1 = "F:\\train_photo_copy\_73737.jpg"
    image = cv2.imread(image1)
    boxes=[[881,605,17,17]]
    corrdinate=findcenter(image,boxes)#获取精细化坐标
    replace_lump_pos(corrdinate,image)


    '''
    '''
    all_txt=get_frame_singal()
    document = Document()
    while len(all_txt)>0:
        (singalump_frame, boxes, conf) = all_txt.pop(0)
        print("当前帧",singalump_frame)
        if singalump_frame%5!=0:
            image1="F:\\train_photo_copy\_"+str(singalump_frame)+".jpg"
            image = cv2.imread(image1)
            corrdinate = findcenter(image, boxes)
            lump_pos=replace_lump_pos(corrdinate,image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
      
            #replace_lump_pos(corrdinate,image,singalump_frame)
            p = document.add_paragraph(str(singalump_frame))

            # with open("corrdinate2.txt", "a") as file:
            # file.write("now frame_count:"+str(frame_count)+"\n")
            # file.write("box:"+str(boxs[0])+","+str(boxs[1])+","+str(boxs[2])+","+str(boxs[3])+"\n")
            run = p.add_run("now frame_count:" + str(singalump_frame))
            run = p.add_run()
            run.add_break()
            run = p.add_run("box:" + str(corrdinate[0]) + "," + str(corrdinate[1]) + "," + str(corrdinate[2]) + "," + str(corrdinate[3]))
            run = p.add_run()
            run.add_break()
            for i in range(corrdinate[3]):
                for j in range(corrdinate[2]):
                    # file.write(str(image[i][j][0])+" ")
                    run = p.add_run(str(image[i+corrdinate[1]][j+corrdinate[0]][0]) + " ")

                    if j >= lump_pos[0]and j <=lump_pos[0]+lump_pos[2]and i >=lump_pos[1] and i <=lump_pos[1]+lump_pos[3]and image[i+corrdinate[1]][j+corrdinate[0]][0]>=90 :


                        run.font.color.rgb = RGBColor(255, 0, 0)
                    
                    # r = run._element
                    # r.rPr.rFonts.set(qn('w:eastAsia'), u'宋体')
                    # file.write("\n")
                run = p.add_run()
                run.add_break()
            document.add_page_break()
        document.save('corrdinate2.docx')  # 关闭保存word

        '''

    '''

          crop = image[corrdinate[1]:corrdinate[1] + corrdinate[3], corrdinate[0]:corrdinate[0] + corrdinate[2]]

          cv2.imshow('crop', crop)
          cv2.waitKey(0)



          img_r = [[0 for p in range(int(corrdinate[2]))] for q in range(corrdinate[3])]
          img_rs = [[0 for p in range(int(corrdinate[2]))] for q in range(corrdinate[3])]
          for p in range(corrdinate[3]):
              for w in range (corrdinate[2]):
                  img_r[p][w] = int(crop[p][w][0])
                  img_rs[p][w] = int(image[p+corrdinate[1]][w+corrdinate[0]][0])
          print("middle")
          for i in range(len(crop)):
              print(img_r[i])
          print("end")
          for i in range(len(crop)):
              print(img_rs[i])
         '''

    '''
    [74, 60, 40] [36, 37, 21]
    [21, 17, 21] [29, 35, 34]
    '''
    '''
    73889;845,584,19,25,;0.9999500513076782,
    '''
