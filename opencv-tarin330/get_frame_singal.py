# 探讨这个帧是否有信号灯
# 将txt文件中的信号灯进行保存，然后放入到list中
'''
with open("yolo1.txt") as file_object:
    lines = file_object.readlines()
for line in lines:
    data_line=line.strip("\n")
    print(data_line)
'''
# 这里获取的是所有的帧数
import numpy as np
import cv2

filepath = "corrdinate_newvideo.txt"


def get_frame_singal():
    object_frame = []
    for txt in open(filepath):
        all = txt.strip().split(";")
        frame = int(all.pop(0))
        length = len(all)
        confs = []
        boxs = []

        for i in range(length):
            if i < length // 2:
                box = all[i].strip().split(",")
                box.pop()
                for j in range(len(box)):
                    box[j] = int(box[j])
                boxs.append(box)
            else:
                conf = all[i].strip().split(",")
                conf.pop()
                for j in range(len(conf)):
                    conf[j] = float(conf[j])
                confs.append(conf)
                # h = all[i].replace(',','')
                # conf.append(float(h))
        object_frame.append((frame, boxs, confs))
    object_frame = detect_smooth(object_frame)
    return object_frame


# all_frame=get_frame_singal()
# print(all_frame[0][0])
# (i,box,conf)=all_frame.pop(0)
# print(i,box,conf)
def detect_smooth(object_frame):
    """
    目标检测的平滑过渡，主要功能是当有一些目标检测可能没有识别出来
    如果这一帧没有识别出来的话。那就给他加上最近的一帧的
    :return:
    """
    frame_count = 50  # 代表如果已经连续识别超过了50帧（1s的时间）的话，就代表这是信号灯。
    flag = 0
    i = 1
    while i < len(object_frame) :
        flag += 1
        # 获取每一帧，如果上一帧与下一帧相差大于1，表示有的没有识别出来。
        # 但是首先需要的是已经连续识别出来了frame_count帧
        # 复制的帧遵循着考进谁就复制谁的原则。
        if object_frame[i][0] - object_frame[i - 1][0] > 1:
            if object_frame[i][0] - object_frame[i - 1][0] > 10:  # 如果超过特定的帧数的话，怀疑是下一个视频
                if flag > frame_count:
                    flag = 0
                else:
                    flag1 = flag
                    count = 0
                    while flag:
                        object_frame.pop(i - flag1)
                        flag -= 1
                        count += 1
                    i = i - count - 1

            else:
                if flag > frame_count:  # 判定是连续的帧
                    minus_frame = object_frame[i][0] - object_frame[i - 1][0]
                    right_frame = object_frame[i]  # 左边的帧在插入的时候不会变，但是右边的帧则会变
                    p = 1
                    while p < minus_frame:
                        if p <= minus_frame // 2:  # 这个时候复制左边的。
                            # 由于存入的是元组
                            frame, box, conf = object_frame[i - 1]
                            frame = frame + p

                        else:
                            frame, box, confmp = right_frame
                            frame = object_frame[i - 1][0] + p
                        object_frame.insert(i - 1 + p, (frame, box, conf))

                        p += 1

                else:  # 假设这段连续的帧没有超过frame_count的话，则怀疑这段目标检测是假的
                    count = 0
                    flag1 = flag
                    while flag:
                        object_frame.pop(i - flag1)
                        flag = flag - 1
                        count += 1
                    i = i - count - 1

        i += 1
    return object_frame


if __name__ == '__main__':
    '''
    object_frame=get_frame_singal()
    #object_frame=detect_smooth(object_frame)
    for i in range (len(object_frame)):
        print(object_frame[i])
    '''
    src_img = [233, 233, 233]
    value_cost = src_img[0] + src_img[1] + src_img[2]

    image_trans_path = "F:\\train_photo\\trans2.png"
    image_trans = cv2.imread(image_trans_path)
    image = "F:\\train_photo_copy\\_73721.jpg"
    image = cv2.imread(image)

    box = [881, 603, 20, 22]
    image_trans = cv2.resize(image_trans, (box[2], box[3]), interpolation=cv2.INTER_AREA)
    r_img, g_img, b_img = image_trans[:, :, 0].copy(), image_trans[:, :, 1].copy(), image_trans[:, :, 2].copy()
    img = r_img + g_img + b_img
    print(img)
    print(image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]])

    r_img[img <= value_cost] = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2], 0]
    g_img[img <= value_cost] = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2], 1]
    b_img[img <= value_cost] = image[box[1]:box[1] + box[3], box[0]:box[0] + box[2], 2]
    image_trans = np.dstack([r_img, g_img, b_img])

    image[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = image_trans[:, :]
    # image=cv2.add(image,image_trans)
    cv2.imshow('image', image)
    cv2.waitKey(0)
