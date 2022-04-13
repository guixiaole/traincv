from xml.dom.minidom import Document
import math
import cv2
import numpy as np
from typing import List
from numpy import *
from get_frame_singal import  get_frame_singal
import os
import re

import math
def replace_image(image,box):
    '''
    进行图片的替换
    :param image:源图片
    :param box: 坐标
    :return:
    '''
    #这里仅仅只是将源文件传进来，然后再进行替换
    image_trans_path = "F:\\train_photo\\trans2.png"
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

def detect_smooth(frame_detect):
    '''
    目标检测的平滑过渡，主要功能是当有一些目标检测可能没有识别出来
    如果这一帧没有识别出来的话。那就给他加上最近的一帧的

    :return:
    '''
    flag=1
    while flag<len(frame_detect):
        temp_len=frame_detect[flag][0]-frame_detect[flag-1][0]
        if temp_len>1:
            temp_frame=[]
            for i in range (len(frame_detect[flag-1])):
                if i==0:
                    temp_frame.append(frame_detect[flag-1][i]+1)
                else:
                    temp_frame.append(frame_detect[flag - 1][i] )
            frame_detect.insert(flag,temp_frame)
        else:
            flag+=1
    #做一个平滑过渡
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
        for i in range(len(frame_detect) - 15):  # 最后20帧不进行平滑
            sum = 0
            if i >= 3 and i < len(frame_detect) - 3:  # 前窗口是有的。
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

            if i >= 3 and i < len(frame_detect) - 3:
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

def predict_frame(object_frame):
    '''
    假设对前两百帧进行一个预测
    :param object_frame:
    :return:
    '''
    #object_frame=[[73721,887,612,13,15],[73771,881,608,12,17,],[73821,869,601,15,20],[73871,856,591,17,24],[73921,836,574,22,29],[73971,803,545,28,37],[74021,734,485,47,60],[74065,567,330,87,115],[74089,225,23,175,216]]
    frame_detct=[]
    frame_detct.append(object_frame[0])
    for i in range(1,len(object_frame)):
        x_list,y_list,h_list,w_list=[],[],[],[]
        temp_len=object_frame[i][0]-object_frame[i-1][0]
        x,y,h,w=object_frame[i][1]-object_frame[i-1][1],object_frame[i][2]-object_frame[i-1][2],object_frame[i][3]-object_frame[i-1][3],object_frame[i][4]-object_frame[i-1][4]
        x_len,y_len,h_len,w_len=x/temp_len,y/temp_len,h/temp_len,w/temp_len
        x_flag,y_flag,h_flag,w_flag=0,0,0,0
        flag=1
        while flag<temp_len:
            x_list.append(object_frame[i-1][1]+round(x_flag))
            y_list.append(object_frame[i-1][2]+round(y_flag))
            h_list.append(object_frame[i-1][3]+round(h_flag))
            w_list.append(object_frame[i-1][4]+round(w_flag))
            x_flag+=x_len
            y_flag+=y_len
            w_flag+=w_len
            h_flag+=h_len
            '''
            if flag%x_len==0:
                if x_len>=0:
                    x_flag+=1
                else:
                    x_flag-=1
            if flag%y_len==0:
                if y_len>=0:
                    y_flag+=1
                else:
                    y_flag-=1
            if flag%h_len==0:
                if h_len>=0:
                    h_flag+=1
                else:
                    h_flag-=1
            if flag%w_len==0:
                if w_len>=0:
                    w_flag+=1
                else:
                    w_flag-=1
            '''
            flag+=1
        count = frame_detct[len(frame_detct)-1][0]
        temp_len=len(x_list)
        for q in range (temp_len):
            frame_detct.append([count+1,x_list.pop(0),y_list.pop(0),h_list.pop(0),w_list.pop(0)])
            count+=1
        frame_detct.append(object_frame[i])
    return frame_detct
def predict_frame_tobond(object_frame):
    '''
    假设对前两百帧进行一个预测
    :param object_frame:
    :return:
    '''
    #object_frame=[[73721,887,612,13,15],[73771,881,608,12,17,],[73821,869,601,15,20],[73871,856,591,17,24],[73921,836,574,22,29],[73971,803,545,28,37],[74021,734,485,47,60],[74065,567,330,87,115],[74089,225,23,175,216]]
    frame_detct=[]
    frame_detct.append(object_frame[0])
    for i in range(1,len(object_frame)):
        x_list,y_list,h_list,w_list=[],[],[],[]
        temp_len=object_frame[i][0]-object_frame[i-1][0]
        x,y,h,w=object_frame[i][1]-object_frame[i-1][1],object_frame[i][2]-object_frame[i-1][2],object_frame[i][3]-object_frame[i-1][3],object_frame[i][4]-object_frame[i-1][4]
        x_len,y_len,h_len,w_len=x/temp_len,y/temp_len,h/temp_len,w/temp_len
        x_flag,y_flag,h_flag,w_flag=0,0,0,0
        flag=1
        while flag<temp_len:
            x_list.append(object_frame[i-1][1]+round(x_flag))
            y_list.append(object_frame[i-1][2]+round(y_flag))
            h_list.append(object_frame[i-1][3]+round(h_flag))
            w_list.append(object_frame[i-1][4]+round(w_flag))
            x_flag+=x_len
            y_flag+=y_len
            w_flag+=w_len
            h_flag+=h_len
            '''
            if flag%x_len==0:
                if x_len>=0:
                    x_flag+=1
                else:
                    x_flag-=1
            if flag%y_len==0:
                if y_len>=0:
                    y_flag+=1
                else:
                    y_flag-=1
            if flag%h_len==0:
                if h_len>=0:
                    h_flag+=1
                else:
                    h_flag-=1
            if flag%w_len==0:
                if w_len>=0:
                    w_flag+=1
                else:
                    w_flag-=1
            '''
            flag+=1
        count = frame_detct[len(frame_detct)-1][0]
        temp_len=len(x_list)
        for q in range (temp_len):
            frame_detct.append([count+1,x_list.pop(0),y_list.pop(0),h_list.pop(0),w_list.pop(0)])
            count+=1
        frame_detct.append(object_frame[i])
    first_w=13
    first_h=15
    for i in range(0,len(frame_detct)):

        if frame_detct[i][3]==first_w:
            frame_detct[i][4]=first_h
        else:
            temp=frame_detct[i][3]-first_w
            first_h+=temp
            first_w+=temp
            frame_detct[i][4]=first_h
    return frame_detct
def half_yoloandhandwork(handwork,yolo_corrdinate):
    '''
    一半手工一半yolo去实现
    handwork: 手工标注预测的那几帧
    yolo_corrdinate: yolo预测的所有桢
    :return:*
    '''
    edge_px=30#边缘像素点
    half_yolo_handwork=[]
    yolo_corrdinate_smooth=yolo_frame_smooth(yolo_corrdinate,edge_px)
    #这里保存最终的结果。一半是yolo，一半是手工标注的预测
    handwork_pro=predict_frame_tobond(handwork)#预测的标记
    temp=0
    for i in range(len(handwork_pro)):
        if handwork_pro[i][3]>=edge_px:#
            temp = i
            break
        else:
            half_yolo_handwork.append(handwork_pro[i])
    if temp==0:
        frame_count_edge=handwork_pro[len(handwork_pro)-1][0]
    else:
        frame_count_edge=handwork_pro[temp][0]
    temp_yolo_count=0
    for i in range(len(yolo_corrdinate_smooth)):
        if yolo_corrdinate_smooth[i][0]==frame_count_edge:
            temp_yolo_count=i
            break
    for j in range(temp_yolo_count,len(yolo_corrdinate_smooth)):
        half_yolo_handwork.append(yolo_corrdinate_smooth[j])
    print('temp=',temp,'temp_yolo',temp_yolo_count)
    for i in range(len(half_yolo_handwork)):
        print(half_yolo_handwork[i])
    return half_yolo_handwork
def yolo_frame_smooth(yolo_detect,edge_px):
    '''
    对yolo目标检测之后过滤的帧进行一个平滑处理
    :return:
    '''
    #首先基于宽度，将宽去做一个平滑处理
    #edge_px=40#边缘的像素点。
    temp_edge=0
    for i in range (len(yolo_detect)):
        if yolo_detect[i][3]>=edge_px:
            temp_edge=i
            break
    #从这个开始，基于宽度进行预测\
    #当flag=0是递减的时候，当flag=1的时候，就是递增
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
    #滑动窗口应该采取3个左右，不应该采取过多
    for i in range(temp_edge,len(yolo_detect)-3):
        sum = 0
        if flag==0:
            if yolo_detect[i - 1][3] - yolo_detect[i][3]<0:
                #出现不寻常的时候就开始滑动窗口
                #默认前后都是正常的
                sum += yolo_detect[i - 1][3] + yolo_detect[i][3] + yolo_detect[i + 1][3]
            else:
                sum=frame_detect[i][3]*3
        else:
            if yolo_detect[i - 1][3] - yolo_detect[i][3] >0:
                sum += yolo_detect[i - 1][3] + yolo_detect[i][3] + yolo_detect[i + 1][3]
            else:
                sum = yolo_detect[i][3] * 3
        avera=int(round(sum/3))#使用四舍五入
        yolo_detect[i][3]=avera#对宽度进行修改之后，然后开始对高度修改
        #按照长宽比来进行比较
    min_ratio=1.26
    max_ratio=1.32
    #最大和最小的高宽比
    #进行高度修改
    for i in range(temp_edge, len(yolo_detect) - 3):
        if yolo_detect[i][4]/yolo_detect[i][3]<min_ratio or yolo_detect[i][4]/yolo_detect[i][3]>max_ratio:
            #假设这个不行的话，那就用宽度乘以1.3
            yolo_detect[i][4]=int(round(yolo_detect[i][3]*1.3))

        #使用滑动窗口
    return yolo_detect

def replace_frame_smooth():
    '''
     使用窗口进行平滑处理
    :return:
    '''
    frame_detect = []
    filepath = "corrdinate_judge.txt"
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
    frame_detect=detect_smooth(frame_detect)
    return frame_detect
def get_corrdinate():
    '''
    获取需要的坐标
    :return:
    '''
    frame_detect=[]
    filepath = "corrdinate_judge.txt"
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
    frame_detect=detect_smooth(frame_detect)
    return frame_detect
if __name__ == '__main__':
    frame_detect=[[73721,887,612,13,15],[73771,881,608,12,17,],[73821,869,601,15,20],[73871,856,591,17,24],[73921,836,574,22,29]]
    #frame_detect=predict_frame()
    #for i in range(len(frame_detect)):
     #   print(frame_detect[i])
    detect_yolo=replace_frame_smooth()
    frame_detect=half_yoloandhandwork(frame_detect,detect_yolo)
    for i in range(len(frame_detect)):
       print(frame_detect[i])