import datetime

import cv2
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import sys
import time
import numpy as np
import h5py

time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
print(time_now)
for i in range(300):
    frame = np.load('F:/numpy/'+str(i)+'.npy')
    time_now1 = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now1)
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    time_now1 = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now1)
    conforatimage = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                                 rgbImage.shape[0], QImage.Format_RGB888)
    # self.changePixmap.emit(frame)
    time_now1 = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now1)