import os

label = []
image = []
file_url = 'E:/save_video/dataset/all/train/dataset4/'
file = os.listdir(file_url)
for i in range(len(file)):
    if file[i][-1] == 'l':
        label.append(file[i])
    elif file[i][-1] == 'g':
        image.append(file[i])
label.sort()
image.sort()
# print(image[1][:-4])
i = 0
while i < len(image):
    if '9' >= label[i][-5] >= '0':
        pass
    else:

        # print(label[i])
        labeldelete = file_url+label[i]
        os.remove(labeldelete)
        label.pop(i)
        # i += 1
    if image[i][:-4] == label[i][:-4]:
        i += 1
    else:
        delete_url = file_url + image[i]
        # print(delete_url)
        image.pop(i)
        os.remove(delete_url)
print(image)
print(label)
