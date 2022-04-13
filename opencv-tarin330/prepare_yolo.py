from opencvyolo_0502 import yolov3_detect, findnet, finln_out
# from opencvyolo_0502 import yolov3_detect,findnet
import time
import datetime
import cv2
import numpy as np

# 先用边缘检测检测出边缘来后。然后再通过检测到边缘之后，再确定中心点然后进行相对应的替换
imagepath = "F:\\train_photo\\train_photo_all\\_74055.jpg"
image_trans_path = "F:\\train_photo\\trans.png"
video_path = "E:/衡阳到长沙/video_part/2247out.avi"
image_trans = cv2.imread(image_trans_path)
cap = cv2.VideoCapture(video_path)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 73721)
i = 1
net = findnet()
ln, out = finln_out(net)
while cap.isOpened():

    # cap.set(cv2.CAP_PROP_POS_FRAMES,6000)
    # print(cv2.CAP_PROP_POS_MSEC)
    ret, frame = cap.read()
    print("i=", i)
    if ret:
        if i >= 0:
            boxes, conf, classID = yolov3_detect(frame, net, ln, out)
            print(boxes, conf)
            if len(conf) > 0:
                with open("txt_file/yolo_83.txt", "a") as file:
                    file.write(str(i) + ";")
                    for p in range(len(boxes)):
                        for j in range(len(boxes[p])):
                            file.write(str(boxes[p][j]) + ",")
                    file.write(";")
                    for h in range(len(conf)):
                        file.write(str(conf[h]) + ",")
                    file.write(";")
                    for g in range(len(classID)):
                        file.write(str(classID[g])+",")
                    file.write("\n")
        i += 1
    else:
        break
