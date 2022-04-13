"""
这个类主要解决的是两个方面
第一：引用yolov3识别出信号灯。找出相对应的坐标
第二：识别出信号灯之后，使用opencv的边际检测，
然后进行替换，替换的图片为原始的
带有颜色的信号灯。
"""
# coding:utf-8
import numpy as np
import cv2
import os
import datetime

weightsPath = 'my_yolov3_50000.weights'  # 模型权重文件
configPath = "my_yolov3.cfg"  # 模型配置文件
labelsPath = "myData.names"  # 模型类别标签文件
LABELS = open(labelsPath).read().strip().split("\n")


def findnet():
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("END : %s" % time_now)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    time_end = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("END : %s" % time_end)
    return net


def finln_out(net):
    ln = net.getLayerNames()
    out = net.getUnconnectedOutLayers()  # 得到未连接层得序号  [[200] /n [267]  /n [400] ]
    x = []
    for i in out:  # 1=[200]
        x.append(ln[i[0] - 1])  # i[0]-1    取out中的数字  [200][0]=200  ln(199)= 'yolo_82'
    ln = x
    return (ln, out)


def yolov3_detect(image, net, ln, out):
    classIDs = []
    boxes = []
    confidences = []
    # image = cv2.imread(image)
    (H, W) = image.shape[0:2]

    # ln  =  ['yolo_82', 'yolo_94', 'yolo_106']  得到 YOLO需要的输出层

    # 从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
    # blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None, ddepth=None)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("END : %s" % time_now)
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True,
                                 crop=False)  # 构造了一个blob图像，对原图像进行了图像的归一化，缩放了尺寸 ，对应训练模型

    net.setInput(blob)  # 将blob设为输入？？？ 具体作用还不是很清楚
    time_mid = datetime.datetime.now().strftime('%H:%M:%S.%f')
    outInfo = net.getUnconnectedOutLayersNames()
    print("mid : %s" % time_mid)

    layerOutputs = net.forward(outInfo)  # ln此时为输出层名称  ，向前传播  得到检测结果

    time_end = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print("END : %s" % time_end)
    for output in layerOutputs:  # 对三个输出层 循环
        for detection in output:  # 对每个输出层中的每个检测框循环
            scores = detection[5:]  # detection=[x,y,h,w,c,class1,class2] scores取第6位至最后
            classID = np.argmax(scores)  # np.argmax反馈最大值的索引
            confidence = scores[classID]
            if confidence > 0.7:  # 过滤掉那些置信度较小的检测结果
                box = detection[0:4] * np.array([W, H, W, H])
                # print(box)
                (centerX, centerY, width, height) = box.astype("int")
                # 边框的左上角
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # 更新检测出来的框
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    return boxes, confidences  # 最后返回相对应的坐标和置信度，以及种类


def frame_to_trans(image, boxes):  # 进行照片的转换，暂时用做demo
    image_trans_path = "F:\\train_photo\\trans.png"
    image_trans = cv2.imread(image_trans_path)
    (x, y) = (boxes[0][0], boxes[0][1])  # 框左上角
    (w, h) = (boxes[0][2], boxes[0][3])  # 框宽高
    crop = image[y:(h + y), x:(w + x)]  # 这只是框的大小

    height_crop, width_crop = crop.shape[:2]
    print(height_crop, width_crop)
    # 将需要替换的照片进行同比例扩大缩小
    # shrink = cv2.resize(image_trans, (width_crop, height_crop), interpolation=cv2.INTER_AREA)
    x_mid = x + w // 2
    y_mid = y + h // 2
    edges = cv2.Canny(crop, 100, 200)
    cv2.imshow('edges', edges)
    cv2.waitKey(0)
    rows, cols, channels = crop.shape
    mid_row = rows // 2
    mid_col = cols // 2
    row_left, row_right = 0, 0
    col_left, col_right = 0, 0
    for i in range(rows):
        if edges[i][mid_col] != 0:
            row_left = i
            break
    for j in range(rows - 1, 0, -1):
        if edges[j][mid_col] != 0:
            row_right = j
            break
    for h in range(cols):
        if edges[mid_row][h] != 0:
            col_left = h
            break
    for q in range(cols - 1, 0, -1):
        if edges[mid_row][q] != 0:
            col_right = q
            break

    print(row_left, row_right)  # 获得两边的最大值。
    print(col_left, col_right)
    # 进行中心值的寻找
    # 将待替换的图片进行缩放
    print(col_right - col_left)
    print(row_right - row_left)

    shrink = cv2.resize(image_trans, ((col_right - col_left), (row_right - row_left)), interpolation=cv2.INTER_AREA)
    center_rows = (row_right + row_left) // 2
    center_cols = (col_left + col_right) // 2
    # cv2.imshow('image1', image)
    # cv2.copyTo(image,image_trans)
    # 对待替换的照片进行边缘检测
    edges_trans = cv2.Canny(shrink, 100, 200)
    # cv2.imshow('edges_trans', edges_trans)

    # 寻找到中心点后，然后进行复制。
    row_temp = row_left + y
    col_temp = col_left + x
    print(shrink[0][0][0])
    # 使用边缘检测后的待替换的照片进行替换
    '''
    for i in range ((col_right-col_left)):
        for j in range ((row_right-row_left)):
            if shrink[j][i][0]!=255 and shrink[j][i][1]!=255 and shrink[j][i][2]!=255  :
                image[row_temp+j][col_temp+i]=shrink[j][i]
    '''
    for i in range((col_right - col_left)):
        # flag=0
        left_trans = 0
        for j in range((row_right - row_left)):
            if edges_trans[j][i] != 0:
                flag = True
                while flag:
                    if edges_trans[j][i] == 0:
                        left_trans = j - 1
                        flag = False
                    j += 1
                break
        right_trans = row_right - row_left - 1
        for h in range(row_right - row_left - 1, -1, -1):
            if edges_trans[h][i] != 0:
                flag = True
                while flag:
                    if edges_trans[h][i] == 0:
                        right_trans = h - 1
                        flag = False
                    h -= 1
                break
        if left_trans == right_trans:
            right_trans = row_right - row_left - 1
            left_trans = 0
        print(left_trans, right_trans)
        while left_trans <= right_trans:
            image[row_temp + left_trans][col_temp + i] = shrink[left_trans][i]
            left_trans += 1
            '''
            if edges_trans[j][i]==0:
                if flag==1:
                    image[row_temp+j][col_temp+i]=shrink[j][i]
            else:
                flag+=1
    '''
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    return image


# 初始化一些参数
'''
if len(confidences)>0:
    (x, y) = (boxes[0][0], boxes[0][1])  # 框左上角
    (w, h) = (boxes[0][2], boxes[0][3])  # 框宽高
    crop=image[y:(h+y),x:(w+x)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([78, 43, 46])
    upper_blue = np.array([110, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    cv2.imshow('erode', erode)
    dilate = cv2.dilate(erode, None, iterations=1)
    cv2.imshow('dilate', dilate)

    # 遍历替换

    cv2.imshow('Mask', mask)
    cv2.imshow('img',crop)
    cv2.waitKey(0)

'''
if __name__ == '__main__':
    '''
    image1='F:\\train_photo\\train_photo_all\\_32790.jpg'
    image2='F:\\train_photo\\train_photo_all\\_74055.jpg'
    box1,conf1=yolov3_detect(image1,net)
    box2,conf2=yolov3_detect(image2,net)
    print(box1,box1[0][0]-box1[0][2],box1[0][1]-box1[0][3],conf1)
    print(box2,box2[0][0]-box2[0][2],box2[0][1]-box2[0][3],conf2)
'''
    image = "F:\\train_photo_copy\_73962.jpg"
    net = findnet()
    ln, out = finln_out(net)
    image = cv2.imread(image)
    boxs, conf = yolov3_detect(image, net, ln, out)
    # boxes=[[594,404,11,12]]
    print(boxs)
    frame_to_trans(image, boxs)
