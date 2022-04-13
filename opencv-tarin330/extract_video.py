#! encoding: UTF-8
import datetime
import os

import cv2

videos_src_path = "E:/save_video/87out183500.avi"
videos_save_path = '/photo'

cap = cv2.VideoCapture(videos_src_path)
start_count = 1
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_count)
frame_count = start_count
success = True
time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
print(time_now)
while success:
    success, frame = cap.read()
    # print ('Read a new frame: ', success)
    if success:
        cv2.imwrite('E:/save_video/replaceVideo/' + "_%d.jpg" % frame_count, frame)
        frame_count = frame_count + 1
    print(frame_count)

cap.release()
