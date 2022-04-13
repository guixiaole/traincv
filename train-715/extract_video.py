#! encoding: UTF-8

import os

import cv2


videos_src_path = 'D:/火车项目/马皇--钦州港_2.MP4'
videos_save_path = '/photo'

cap = cv2.VideoCapture(videos_src_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 73720)
frame_count = 73720
success = True
while(success):
    success, frame = cap.read()
    #print ('Read a new frame: ', success)
    if frame_count>=73720:
        if frame_count>0:

            cv2.imwrite('F:/train_photo_all/'+"_%d.jpg" % frame_count, frame)
    print(frame_count)
    frame_count = frame_count + 1

cap.release()