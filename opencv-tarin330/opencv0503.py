from opencvyolo_0502 import yolov3_detect,findnet,finln_out
import time
import datetime
import cv2
import numpy as np
#先用边缘检测检测出边缘来后。然后再通过检测到边缘之后，再确定中心点然后进行相对应的替换
imagepath="F:\\train_photo\\train_photo_all\\_74055.jpg"
image_trans_path="F:\\train_photo\\trans.png"
image_trans=cv2.imread(image_trans_path)
net=findnet()
ln,out=finln_out(net)

boxes,conf=yolov3_detect(imagepath,net,ln,out)



#先将照片裁剪出来
image=cv2.imread(imagepath)
(x, y) = (boxes[0][0], boxes[0][1])  # 框左上角
(w, h) = (boxes[0][2], boxes[0][3])  # 框宽高
crop = image[y:(h + y), x:(w + x)]#这只是框的大小

height_crop, width_crop = crop.shape[:2]
print(height_crop, width_crop)
#将需要替换的照片进行同比例扩大缩小
#shrink = cv2.resize(image_trans, (width_crop, height_crop), interpolation=cv2.INTER_AREA)
x_mid=x+w//2
y_mid=y+h//2
edges = cv2.Canny(crop, 100, 200)
'''
cv2.imshow('edges',edges)
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)
lower_blue = np.array([78, 43, 46])
upper_blue = np.array([110, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imshow('mask',mask)
'''
rows, cols, channels = crop.shape
mid_row=rows//2
mid_col=cols//2
row_left,row_right=0,0
col_left,col_right=0,0
for i in range (rows):
    if edges[i][mid_col]!=0:
        row_left=i
        break
for j in range (rows-1,0,-1):
    if edges[j][mid_col]!=0:
        row_right=j
        break
for h in range (cols):
    if edges[mid_row][h]!=0:
        col_left=h
        break
for q in range (cols-1,0,-1):
    if edges[mid_row][q]!=0:
        col_right=q
        break

print(row_left,row_right)#获得两边的最大值。
print(col_left,col_right)
#进行中心值的寻找
#将待替换的图片进行缩放
shrink = cv2.resize(image_trans, ((col_right-col_left), (row_right-row_left)), interpolation=cv2.INTER_AREA)
center_rows=(row_right+row_left)//2
center_cols=(col_left+col_right)//2
cv2.imshow('image1',image)
#cv2.copyTo(image,image_trans)
#对待替换的照片进行边缘检测
edges_trans = cv2.Canny(shrink, 100, 200)
cv2.imshow('edges_trans',edges_trans)

#寻找到中心点后，然后进行复制。
row_temp=row_left+y
col_temp=col_left+x
print(shrink[0][0][0])
#使用边缘检测后的待替换的照片进行替换
'''
for i in range ((col_right-col_left)):
    for j in range ((row_right-row_left)):
        if shrink[j][i][0]!=255 and shrink[j][i][1]!=255 and shrink[j][i][2]!=255  :
            image[row_temp+j][col_temp+i]=shrink[j][i]
'''
for i in range ((col_right-col_left)):
    #flag=0
    left_trans=0
    for j in range ((row_right-row_left)):
        if edges_trans[j][i]!=0:
            flag = True
            while flag:
                if edges_trans[j][i] == 0:
                    left_trans=j-1
                    flag = False
                j += 1
            break
    right_trans = row_right-row_left-1
    for h in range (row_right-row_left-1,-1,-1):
        if edges_trans[h][i]!=0:
            flag=True
            while flag:
                if edges_trans[h][i]==0:
                    right_trans=h-1
                    flag=False
                h-=1
            break
    if left_trans==right_trans:
        right_trans = row_right - row_left - 1
        left_trans = 0
    print(left_trans,right_trans)
    while left_trans<=right_trans:
        image[row_temp + left_trans][col_temp + i] = shrink[left_trans][i]
        left_trans+=1
        '''
        if edges_trans[j][i]==0:
            if flag==1:
                image[row_temp+j][col_temp+i]=shrink[j][i]
        else:
            flag+=1
'''
cv2.imshow('edge',edges)
cv2.imshow('image',image)
cv2.imshow('shrink',shrink)
cv2.waitKey(0)
'''
if len(conf)>0:
    mid=(boxes[0][0]+boxes[0][2])/2
    max_pix=[]
    for i in range (boxes[0][1]+1,boxes[0][3]):
        minus=image[mid][i]-image[mid][i-1]
        max_pix.append((i,minus))
        print(i,minus)
        if len(max_pix)>2:
            max_pix.remove(min(max_pix))
    print(max_pix)
'''
print(boxes,conf)