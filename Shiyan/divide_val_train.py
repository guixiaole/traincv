import os
import random
import sys

file_url = 'E:/save_video/dataset/all/train/dataset4/'
file = os.listdir(file_url)
print(len(file))
# print(file[1][:-4])
trainfile = open('train.txt', 'a')
for i in range(len(file)):
    if file[i][-1] == 'g':
        trainfile.write(file[i][:-4])
        trainfile.write('\n')
trainfile.close()
print(file)
