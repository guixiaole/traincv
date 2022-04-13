import datetime
import os
import glob
from threading import Thread

import tables
import numpy as np
import cv2
import h5py
import mxnet as mx
from PIL import Image, ImageQt
import concurrent.futures
from PyQt5.QtGui import QImage
from qtpy import QtGui
import sys
import qimage2ndarray

videoName = 'D:/衡阳-岳阳.mp4'
cap = cv2.VideoCapture(videoName)


# 跳转到指定的帧
def savenumpy():
    step = 0
    order_frame = 0  # 跳到指定的某一帧
    # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    npfile = 'F:/numpy/test/'
    cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    hdf5_path = "test.hdf5"
    image_list = np.zeros((500, 1080, 1920, 3), dtype=np.uint8)
    while cap.isOpened():
        ret, frame = cap.read()
        # print(sys.getsizeof(frame))
        # np.save('F:/numpy/test1.npy', frame)
        if ret:
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_list[order_frame % 500] = rgbImage
            if order_frame % 500 == 0 and order_frame != 0:
                print('image_list:', sys.getsizeof(image_list))
                h5f = h5py.File('F:/numpy/test1' + '.h5', 'a')
                h5f.create_dataset('dataset', data=image_list, dtype=np.uint8, compression="gzip", compression_opts=9)
                h5f.close()

            #     if order_frame != 0:
            #         h5f.close()
            #     h5f = h5py.File('F:/numpy/' + str(order_frame) + '.hdf5', 'a')
            # h5f.create_dataset('dataset' + str(order_frame), data=rgbImage, dtype='int')
            # np.save(, frame)
            print(order_frame)
            order_frame += 1


def loadnumpy():
    order_frame = 0
    while order_frame < 5000:
        if order_frame % 500 == 0:
            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            print(time_now)
            if order_frame != 0:
                h5f.close()
            h5f = h5py.File('F:/numpy/' + str(order_frame) + '.h5', 'r')
            # h5f.close()

        # h5f = h5py.File('E:/save_video/numpy/' + str(0) + '.h5', 'r')
        rgbImage = h5f['dataset' + str(order_frame)][:]
        # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(order_frame)
        order_frame += 1


image_map = []


def phototest(imagefile):
    image = cv2.imread(imagefile)
    image_map.append(image)


def photo_load():
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        image_url = 'F:\\numpy\\photo'
        imagefile = glob.glob(image_url + '\\*.jpg')
        executor.map(phototest, imagefile)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    print(len(image_map))


def phototest1():
    image_url = 'D:/photo/_'
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    for i in range(500):
        image = cv2.imread(image_url + str(i) + '.jpg')
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)


def savePhoto():
    """
    改为存储图片试试。
    :return:
    """
    order_frame = 0  # 跳到指定的某一帧
    # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    photourl = 'F:/numpy/'
    cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(photourl + "_%d.jpg" % order_frame, frame)
            print(order_frame)
            order_frame += 1


def load_photo():
    photourl = 'F:/numpy/_'
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    for i in range(500):
        time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print(time_now)
        image = Image.open(photourl + str(i) + '.jpg')
        # print(type(image))
        time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print(time_now)
        qimg = ImageQt.toqimage(image)
        time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print(time_now)
        # conforatimage = ImageQt.toqpixmap(image)
        # time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        # print(time_now)

        # print(type(qimg))
        print(i)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)


def load_photo_Qimage():
    photourl = 'F:/numpy/_'
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    for i in range(500):
        image = QImage(photourl + str(i) + '.jpg')
        print(type(image))
        print(i)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)


def load_image_mxnet():
    photourl = 'F:/numpy/_'
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    for i in range(500):
        # rgbImage = mx.image.imdecode(open('F:/numpy/_' + str(i) + '.jpg', 'rb').read())
        rgbImage = mx.image.imread('F:/numpy/_' + str(i) + '.jpg')
        # rgbImage = cv2.imread('F:/numpy/_' + str(i) + '.jpg')
        time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print(time_now)
        image = qimage2ndarray.array2qimage(rgbImage)
        # rgbImage = rgbImage.asnumpy()
        time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
        print(time_now)
        # image = QImage(photourl + str(i) + '.jpg')
        # print(type(rgbImage))
        # print(type(conforatimage))
        print(i)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)


def opencv_load():
    # cap = cv2.VideoCapture(videoName)
    # 跳转到指定的帧
    step = 1
    order_frame = 1  # 跳到指定的某一帧
    # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # print(type(rgbImage))
            # rgbImage = frame
            # print("2 : %s" % time_now)
            # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
            # 2ms
            # rgbImage = frame
            # covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
            #                                 rgbImage.shape[0], QImage.Format_RGB888)
            if order_frame > 500:
                time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
                print(time_now)
                break
            # print(order_frame)
            order_frame += 1


frame_dict1 = {}
frame_dict2 = {}
frame_dict3 = {}


def run3():
    cap1 = cv2.VideoCapture(videoName)
    # 跳转到指定的帧
    step = 1
    order_frame = 101  # 跳到指定的某一帧
    # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    while cap1.isOpened():

        if len(frame_dict3) <= 501:
            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            # print("frame_dict3 : %s" % time_now)
            # with open("hashmap1.txt", "a") as file:
            #     file.write(str(time_now) + ";")
            #     file.write(str(order_frame) + ";")
            #     file.write("\n")
            if (order_frame - 1) % 100 == 0 and order_frame != 101:
                step += 2
                # print('run3 step:', step,'stop')
                cap1.set(cv2.CAP_PROP_POS_FRAMES, step * 200 + 1)
                order_frame = step * 200 + 1

            # print("当前帧：", order_frame)
            # if len(self.frame_list) < 100:
            # print('run3 order_frame:' + str(order_frame) + 'stop')
            ret, frame = cap1.read()
            if ret:
                # self.frame_list.append((order_frame, frame))
                frame_dict3[str(order_frame)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(type(rgbImage))
                # rgbImage = frame
                # print("2 : %s" % time_now)
                # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                # 2ms
                # rgbImage = frame
                # covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                #                                 rgbImage.shape[0], QImage.Format_RGB888)
                # # print("3 : %s" % time_now)
                # # # 7ms
                # # # print(type(covertToQtFormat))
                # p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                # # print(type(p))
                # self.frame_dict3[str(order_frame)] = rgbImage
                # self.frame_list.append((self.frame_count, frame))
                # order_frame += 1
                order_frame += 1
        else:
            # sleep(0.0001)
            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            print(time_now)
            break


def run1():
    cap1 = cv2.VideoCapture(videoName)
    # 跳转到指定的帧
    step = 1
    order_frame = 1  # 跳到指定的某一帧
    # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    while cap1.isOpened():

        if len(frame_dict3) <= 501:
            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            # print("frame_dict3 : %s" % time_now)
            # with open("hashmap1.txt", "a") as file:
            #     file.write(str(time_now) + ";")
            #     file.write(str(order_frame) + ";")
            #     file.write("\n")
            if (order_frame - 1) % 100 == 0 and order_frame != 1:
                step += 2
                # print('run3 step:', step,'stop')
                cap1.set(cv2.CAP_PROP_POS_FRAMES, step * 200 + 1)
                order_frame = step * 200 + 1

            # print("当前帧：", order_frame)
            # if len(self.frame_list) < 100:
            # print('run3 order_frame:' + str(order_frame) + 'stop')
            ret, frame = cap1.read()
            if ret:
                # self.frame_list.append((order_frame, frame))
                frame_dict3[str(order_frame)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(type(rgbImage))
                # rgbImage = frame
                # print("2 : %s" % time_now)
                # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                # 2ms
                # rgbImage = frame
                # covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                #                                 rgbImage.shape[0], QImage.Format_RGB888)
                # # print("3 : %s" % time_now)
                # # # 7ms
                # # # print(type(covertToQtFormat))
                # p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                # # print(type(p))
                # self.frame_dict3[str(order_frame)] = rgbImage
                # self.frame_list.append((self.frame_count, frame))
                # order_frame += 1
                order_frame += 1
        else:
            # sleep(0.0001)
            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            print(time_now)
            break


def run2():
    step = 1
    order_frame = 0  # 跳到指定的某一帧
    # cap.set(cv2.CAP_PROP_POS_FRAMES, order_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 3500)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    retval = cap.grab()
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    ret, frame = cap.retrieve(retval)
    time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    print(time_now)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    while cap.isOpened():

        if len(frame_dict3) <= 501:
            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            # print("frame_dict3 : %s" % time_now)
            # with open("hashmap1.txt", "a") as file:
            #     file.write(str(time_now) + ";")
            #     file.write(str(order_frame) + ";")
            #     file.write("\n")
            if (order_frame - 1) % 100 == 0 and order_frame != 201:
                step += 2
                # print('run3 step:', step,'stop')
                cap.set(cv2.CAP_PROP_POS_FRAMES, step * 200 + 1)
                order_frame = step * 200 + 1

            # print("当前帧：", order_frame)
            # if len(self.frame_list) < 100:
            # print('run3 order_frame:' + str(order_frame) + 'stop')
            ret, frame = cap.read()
            if ret:
                # self.frame_list.append((order_frame, frame))
                frame_dict1[str(order_frame)] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print(type(rgbImage))
                # rgbImage = frame
                # print("2 : %s" % time_now)
                # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
                # 2ms
                # rgbImage = frame
                # covertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                #                                 rgbImage.shape[0], QImage.Format_RGB888)
                # # print("3 : %s" % time_now)
                # # # 7ms
                # # # print(type(covertToQtFormat))
                # p = covertToQtFormat.scaled(1919, 1079, Qt.KeepAspectRatio)
                # # print(type(p))
                # self.frame_dict3[str(order_frame)] = rgbImage
                # self.frame_list.append((self.frame_count, frame))
                # order_frame += 1
                order_frame += 1
        else:
            # sleep(0.0001)

            time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
            print(time_now)
            break


if __name__ == '__main__':
    run2()
    # savenumpy()
    # loadnumpy()
    # phototest()
    # savePhoto()
    # time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    # print(time_now)
    # t1 = Thread(target=run1)
    # t2 = Thread(target=run2)
    # t3 = Thread(target=run3)
    # t1.start()
    # t2.start()
    # t3.start()
    # if len(frame_dict1)>500 and len(frame_dict2)>500 and len(frame_dict3)>500:
    #     time_now = datetime.datetime.now().strftime('%H:%M:%S.%f')
    #     print(time_now)
