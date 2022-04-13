import os
import sys

import cv2
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import *
from LUMPOperate import SplitLump

from MarkPicture.Markpicture import Ui_MainWindow
from QClickableImage import QClickableImage


class video_Box(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.max_columns = 0
        self.setupUi(self)
        self.videoName = ""
        self.excelName = ""
        self.saveurl = "E:/save_video/"
        self.display_image_size = 200
        self.col = 0
        self.row = 0
        self.initial_path = None

    def on_left_clicked(self, image_id):
        print('left clicked - image id = ' + image_id)

    def on_right_clicked(self, image_id):
        print('right clicked - image id = ' + image_id)

    def openvideo(self):  # 打开视频的函数
        self.videoName, videoType = QFileDialog.getOpenFileName(self,
                                                                "打开视频",
                                                                "",
                                                                " *.mp4;;*.avi;;All Files (*)"
                                                                )
        print(self.videoName)

    def openexcel(self):
        self.excelName, excelType = QFileDialog.getOpenFileName(self,
                                                                "打开excel",
                                                                "",
                                                                "*.csv;;*.xls;;All Files (*)"
                                                                )

        print(self.excelName)

    def extractlump(self):
        if self.videoName == "" or self.excelName == "":
            QMessageBox.information(self, '提示', '请输入excel和视频')
        else:
            QMessageBox.information(self, '提示', '分割视频较久，请耐心等待')
            SplitLump.video_lump_classification(self.videoName, self.excelName, self.saveurl)
            SplitLump.saveImage_calssification()
            QMessageBox.information(self, '提示', '已经分割好了视频，请点击显示视频')

    def showlump(self):
        """
        将图片进行存储，然后
        :return:
        """
        video_path = "E:/save_video/ShowImage/"
        image_name = os.listdir(video_path)
        if len(image_name) < 10:
            #  表示已经存在了照片了。
            QMessageBox.information(self, '提示', '稍等，正在提取视频')
            SplitLump.saveImage_calssification()
        for i in range(len(image_name)):
            image_id = str(image_name[i])
            image_url = str(video_path + image_name[i])
            pixmap = QPixmap(image_url)
            self.addImage(pixmap, image_id)
            QApplication.processEvents()


    def clear_layout(self):
        for i in range(self.gridLayout.count()):
            self.gridLayout.itemAt(i).widget().deleteLater()

    def showlump2(self):
        """
        显示信号灯的片段，然后根据所选择的片段进行

        :return:
        @gxl 8/13
        """

        video_path = "E:/save_video/"
        video_name = os.listdir(video_path)
        frame_save = []
        for i in range(len(video_name)):
            video_path_part = os.path.join(video_path, video_name[i])
            cap = cv2.VideoCapture(video_path_part)  # 读取视频
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
            while True:
                ret, frame = cap.read()
                if ret:
                    frame_save.append(frame)
                break
            cap.release()
        #  保存了所有的视频的图片
        for i in range(len(frame_save)):
            image_id = frame_save[i]
            rgbImage = cv2.cvtColor(image_id, cv2.COLOR_BGR2RGB)
            # cv2.imwrite('F:/train_photo_copy_smooth/' + "_%d.jpg" % frame_count, frame)
            image_id = QtGui.QImage(rgbImage.data, rgbImage.shape[1],
                                    rgbImage.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap(image_id)
            self.addImage(pixmap, video_name[i])
            QApplication.processEvents()

    def addImage(self, pixmap, image_id):
        #  计算图像的列数
        nr_of_columns = self.get_nr_of_image_columns()
        #  这个布局内的数量
        nr_of_widgets = self.gridLayout.count()
        self.max_columns = nr_of_columns
        if self.col < self.max_columns:
            self.col += 1
        else:
            self.col = 0
            self.row += 1
        clickable_image = QClickableImage(self.display_image_size, self.display_image_size, pixmap, image_id)
        clickable_image.clicked.connect(self.on_left_clicked)
        clickable_image.rightClicked.connect(self.on_right_clicked)
        #  print(self.row, self.col)
        self.gridLayout.addWidget(clickable_image, self.row, self.col)

    def get_nr_of_image_columns(self):
        # 展示图片的区域，计算每排显示图片数。返回的列数-1是因为我不想频率拖动左右滚动条，影响数据筛选效率
        scroll_area_images_width = int(0.40 * self.width)
        if scroll_area_images_width > self.display_image_size:

            pic_of_columns = scroll_area_images_width // self.display_image_size  # 计算出一行几列；
        else:
            pic_of_columns = 1

        return pic_of_columns - 1


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = video_Box()
    window.show()
    sys.exit(app.exec_())
