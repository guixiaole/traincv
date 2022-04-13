import copy

import cv2


def findlr(image, boundary):
    """
    在图像替换的时候，寻找到他的边界。
    :param image:
    :param boundary:
    :return:
    """
    lr = []
    for j in range(len(image)):
        left = 0
        right = len(image[0])
        for i in range(1, len(image[0])):
            temp = image[j][i - 1][0]
            temp1 = image[j][i][0]
            if image[j][i - 1][0] >= boundary > image[j][i][0]:
                left = i
                break
        for i in range(len(image[0]) - 2, 0, -1):
            temp2 = image[j][i + 1][0]
            temp3 = image[j][i][0]
            if image[j][i + 1][0] >= boundary > image[j][i][0]:
                right = i
                break
        lr.append([left, right])
    return lr


image_trans_path = "E:/trans/trans3.png"
image_trans = cv2.imread(image_trans_path)
cv2.imshow('image_trans', image_trans)
print(len(image_trans))
print(len(image_trans[0]))
for i in range(len(image_trans[0])):
    for j in range(len(image_trans)):
        if len(image_trans[0]) - (len(image_trans[0]) / 10) > i and i < len(image_trans[0]) / 10:
            if image_trans[j][i][0] >= 128 or image_trans[j][i][1] >= 128 or image_trans[j][i][2] >= 128:
                image_trans[j][i] = 255
        if len(image_trans) - (len(image_trans) / 10) > j and j < len(image_trans) / 10:
            if image_trans[j][i][0] >= 128 or image_trans[j][i][1] >= 128 or image_trans[j][i][2] >= 128:
                image_trans[j][i] = 255
cv2.imshow('image_trans1', image_trans)
shrink = cv2.resize(image_trans, (7, 9), interpolation=cv2.INTER_AREA)
# if 96 <= frame_count <= 98:
#     cv2.imshow('shrink1', shrink)
#     cv2.waitKey(0)

temp = copy.copy(shrink)
temo = copy.copy(shrink)
temp = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)
lr = findlr(temp, 100)
for i in range(len(shrink)):
    for j in range(len(shrink[0])):
        # if 80 - (80 / 10) > i and i < 80 / 10:
        #     if temp[j][i][0] >= 180 or temp[j][i][1] >= 180 or temp[j][i][2] >= 180:
        #         # if shrink[j][i][0] >= 100 and shrink[j][i][1] >= 100 and shrink[j][i][2] >= 100:
        #         shrink[j][i] = 0
        # if 150 - 150 / 10 > j and j < 150 / 10:
        #     if temp[j][i][0] >= 180 or temp[j][i][1] >= 180 or temp[j][i][2] >= 180:
        #         # if shrink[j][i][0] >= 100 and shrink[j][i][1] >= 100 and shrink[j][i][2] >= 100:
        #         shrink[j][i] = 0
        if lr[j][0] <= i <= lr[j][1]:
            if temp[j][i][0] >= 128 or temp[j][i][1] >= 128 or temp[j][i][2] >= 128:
                shrink[j][i] = 0
cv2.imshow('shrink', shrink)
# cv2.imshow('temo', temo)
# cv2.imshow('temp', temp)
cv2.waitKey(0)

for j in range(len(temp[0])):
    for h in range(len(temp)):
        print(temp[h][j][0], end=' ')
    print()
    print()
print("jiange")

print(findlr(temp, 100))
