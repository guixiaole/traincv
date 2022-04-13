'''
这个类主要解决的是两个方面
第一：引用yolov3识别出信号灯。找出相对应的坐标
第二：识别出信号灯之后，使用opencv的边际检测，
然后进行替换，替换的图片为原始的
带有颜色的信号灯。
'''
#coding:utf-8
import numpy as np
import cv2
import os
weightsPath='D:\darknet\my_yolov3_30000.weights'# 模型权重文件
configPath="D:\darknet\my_yolov3.cfg"# 模型配置文件
labelsPath = "D:\darknet\myData.names"# 模型类别标签文件
#初始化一些参数
LABELS = open(labelsPath).read().strip().split("\n")
boxes = []
confidences = []
classIDs = []

#加载 网络配置与训练的权重文件 构建网络
net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
#读入待检测的图像
image = cv2.imread('F:\\train_photo\\train_photo_all\\_32790.jpg')
#得到图像的高和宽
(H,W) = image.shape[0:2]


# 得到 YOLO需要的输出层
ln = net.getLayerNames()
out = net.getUnconnectedOutLayers()#得到未连接层得序号  [[200] /n [267]  /n [400] ]
x = []
for i in out:   # 1=[200]
    x.append(ln[i[0]-1])    # i[0]-1    取out中的数字  [200][0]=200  ln(199)= 'yolo_82'
ln=x
# ln  =  ['yolo_82', 'yolo_94', 'yolo_106']  得到 YOLO需要的输出层



#从输入图像构造一个blob，然后通过加载的模型，给我们提供边界框和相关概率
#blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None, ddepth=None)
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)#构造了一个blob图像，对原图像进行了图像的归一化，缩放了尺寸 ，对应训练模型

net.setInput(blob) #将blob设为输入？？？ 具体作用还不是很清楚
layerOutputs = net.forward(ln)  #ln此时为输出层名称  ，向前传播  得到检测结果

for output in layerOutputs:  #对三个输出层 循环
    for detection in output:  #对每个输出层中的每个检测框循环
        scores=detection[5:]  #detection=[x,y,h,w,c,class1,class2] scores取第6位至最后
        classID = np.argmax(scores)#np.argmax反馈最大值的索引
        confidence = scores[classID]
        if confidence >0.7:#过滤掉那些置信度较小的检测结果
            box = detection[0:4] * np.array([W, H, W, H])
            #print(box)
            (centerX, centerY, width, height)= box.astype("int")
            # 边框的左上角
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            # 更新检测出来的框
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)
print(boxes)
if len(confidences)>0:
    #替换的图片：
    image_trans=cv2.imread('F:\\train_photo\\trans1.png')

    (x, y) = (boxes[0][0], boxes[0][1])  # 框左上角
    (w, h) = (boxes[0][2], boxes[0][3])  # 框宽高
    crop=image[y:(h+y),x:(w+x)]
    height_crop,width_crop=crop.shape[:2]
    print(height_crop,width_crop)
    shrink = cv2.resize(image_trans, (width_crop, height_crop), interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv1 = cv2.cvtColor(shrink, cv2.COLOR_BGR2HSV)




    lower_blue = np.array([78, 43, 46])
    upper_blue = np.array([110, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask1 = cv2.inRange(hsv1, lower_blue, upper_blue)
    # 腐蚀膨胀
    erode = cv2.erode(mask, None, iterations=1)
    erode1 = cv2.erode(mask1, None, iterations=1)
    dilate = cv2.dilate(erode, None, iterations=1)
    dilate1 = cv2.dilate(erode1, None, iterations=1)

    cv2.imshow('dilate',dilate)

    dilate1_inv = cv2.bitwise_not(dilate1)
    cv2.imshow('dilate1_inv', dilate1_inv)

    # 遍历替换
    #dst=cv2.add(image_trans,dilate)
    #cv2.imshow('dst', dst)
    rows,cols,channels = crop.shape
    for i in range (rows):
        for j in range (cols):
            if dilate[i,j]==0:
                image[y+i,x+j]=shrink[i,j]
    cv2.imshow('image',image)
   # cv2.imshow('Mask', mask)
   #cv2.imshow('img',crop)
    cv2.waitKey(0)


print(boxes,confidences)

'''
idxs=cv2.dnn.NMSBoxes(boxes, confidences, 0.2,0.3)
box_seq = idxs.flatten()#[ 2  9  7 10  6  5  4]
if len(idxs)>0:
    for seq in box_seq:
        (x, y) = (boxes[seq][0], boxes[seq][1])  # 框左上角
        (w, h) = (boxes[seq][2], boxes[seq][3])  # 框宽高
        if classIDs[seq]==0: #根据类别设定框的颜色
            color = [0,0,255]
        else:
            color = [0,255,0]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)  # 画框
        text = "{}: {:.4f}".format(LABELS[classIDs[seq]], confidences[seq])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)  # 写字
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)

'''