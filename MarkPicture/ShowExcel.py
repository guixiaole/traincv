import os

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon
import MarkPicture.LUMPOperate.SplitLump as SplitLump
import MarkPicture.LUMPOperate.OperatePhoto as OperatePhoto
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pandas as pd
import numpy as np
import csv


class Ui_MainWindow(QMainWindow):

    def __init__(self):
        super(QtWidgets.QMainWindow, self).__init__()
        self.desktop = QApplication.desktop()
        self.height = self.desktop.height()
        self.width = self.desktop.width()
        self.setupUi(self)
        self.retranslateUi(self)
        self.lumpColor = 0  # 改颜色。 1为红色，2位黄色，3位绿色，0为还没有设置颜色
        self.videoName = ""
        self.excelName = ""  # videoName 和excelName

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.width, self.height)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.retranslateUi(MainWindow)

        self.tableWidget = QtWidgets.QTableWidget(self.centralWidget)
        self.tableWidget.setGeometry(QtCore.QRect(0, 60, self.width * 0.3, self.height * 0.8))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.tableWidget.setStyleSheet("selection-background-color:pink")
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.raise_()

        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(90, 20, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("打开excel")

        self.pushButton6 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton6.setGeometry(QtCore.QRect(360, 20, 75, 23))
        self.pushButton6.setObjectName("pushButton6")
        self.pushButton6.setText("打开视频")

        self.pushButton7 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton7.setGeometry(QtCore.QRect(270, 20, 75, 23))
        self.pushButton7.setObjectName("pushButton7")
        self.pushButton7.setText("分割视频")

        self.pushButton1 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton1.setGeometry(QtCore.QRect(180, 20, 75, 23))
        self.pushButton1.setObjectName("pushButton1")
        self.pushButton1.setText("显示")

        self.pushButton3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton3.setGeometry(QtCore.QRect(self.width * 0.4, 200, 100, 50))
        self.pushButton3.setObjectName("pushButton3")
        self.pushButton3.setText("红灯")

        self.pushButton4 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton4.setGeometry(QtCore.QRect(self.width * 0.4, 400, 100, 50))
        self.pushButton4.setObjectName("pushButton4")
        self.pushButton4.setText("黄灯")

        self.pushButton5 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton5.setGeometry(QtCore.QRect(self.width * 0.4, 600, 100, 50))
        self.pushButton5.setObjectName("pushButton5")
        self.pushButton5.setText("绿灯")

        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.tableWidget.doubleClicked.connect(self.tableWidget_double_clicked)
        # self.pushButton3.clicked.connect(self.tableWidget_double_clicked)

        self.pushButton.clicked.connect(self.openfile)  # 选择excel
        self.pushButton1.clicked.connect(self.creat_table_show1)  # 显示excel
        self.pushButton3.clicked.connect(self.setRedLump)  # 设置成灯的颜色
        self.pushButton4.clicked.connect(self.setYellowLump)
        self.pushButton5.clicked.connect(self.setGreenLump)
        self.pushButton6.clicked.connect(self.openvideo)
        self.pushButton7.clicked.connect(self.extractlump)

    # 改颜色。 1为红色，2位黄色，3位绿色，0为还没有设置颜色
    # self.lumpColor
    def openvideo(self):  # 打开视频的函数
        self.videoName, videoType = QFileDialog.getOpenFileName(self,
                                                                "打开视频",
                                                                "",
                                                                " *.mp4;;*.avi;;All Files (*)"
                                                                )
        print(self.videoName)

    def extractlump(self):
        if self.videoName == "" or self.excelName == "":
            QMessageBox.information(self, '提示', '请输入excel和视频')
        else:
            QMessageBox.information(self, '提示', '分割视频较久，请耐心等待')
            SplitLump.video_lump_store_classification1(self.videoName, self.excelName, self.saveurl)
            # SplitLump.video_lump_classification(self.videoName, self.excelName, self.saveurl)
            SplitLump.saveImage_calssification()
            QMessageBox.information(self, '提示', '已经分割好了视频，请点击显示视频')

    def setRedLump(self):
        self.lumpColor = 1
        QMessageBox.information(self, '提示', '已将红灯设置成需要替换的信号灯。')

    def setYellowLump(self):
        self.lumpColor = 2
        QMessageBox.information(self, '提示', '已将黄灯设置成需要替换的信号灯。')

    def setGreenLump(self):
        self.lumpColor = 3
        QMessageBox.information(self, '提示', '已将绿灯设置成需要替换的信号灯。')

    def tableWidget_double_clicked(self, index):
        if self.lumpColor == 0:
            QMessageBox.information(self, '提示', '请先确定右边需要修改的信号灯。')
        else:
            print('233333')
            table_column = index.column()
            table_row = index.row()
            current_item = self.tableWidget.item(table_row, table_column)
            print(current_item.text())
            # current_widget = self.tableWidget.cellWidget(table_row, table_column)

            print(table_row)
            video_url = "E:/save_video/"
            video_name = os.listdir(video_url)
            print(video_name)
            print(table_column)
            # 需要的url
            singal_video_url = ''
            for i in range(len(video_name)):
                if video_name[i][len(str(table_row))] == 'o' and video_name[i][:len(str(table_row))] == str(
                        table_row):  # 代表是这个url
                    singal_video_url = video_name[i]
                    break
            competed_video_url = video_url + singal_video_url
            frame_count = int(singal_video_url[len(str(table_row)) + 3:-4])  # 是在哪个帧开始的。
            # 暂时注释掉，这样比较快一点。
            SplitLump.store_image_to_input(video_url=competed_video_url, frame_count=frame_count)  # 将视频存储到标记的里面去。
            os.system('labelme ' + ' openlabel/input/' + ' --output' + ' openlabel/output/PASCAL_VOC/')
            OperatePhoto.clearData()
            # 这个是yolov3的detect，后续改为了yolov4
            # OperatePhoto.yolo_detect(competed_video_url, frame_count)  # 进行yolo的探测
            # OperatePhoto.yolo_v4_detect(competed_video_url, frame_count)
            # 在获取的时候，假设是两个灯的话，则需要在获取json的时候做一个判断。
            # 然后根据json得来的数据，进行判断是不是两个灯还是一个灯的信号灯。

            cord = SplitLump.get_output_labeljson()  # 获取json文件的信息
            # 获取json数据之后，进行一个判断。
            if len(cord[0]) > 5:
                # 如果是两个信号灯的话。
                ratio = OperatePhoto.get_raito_length_width(cord)  # 这里获取的是最大最小的长宽比。

                predict_hand = OperatePhoto.predict_frame_tobond_doublelump(cord)

                OperatePhoto.store_pos_txt('txt/lumppospredicate.txt', predict_hand)  # 将预测结果进行储存。

                OperatePhoto.yolo_detect_to_smooth_doublelump1120(ratio, competed_video_url, frame_count)  # 对获取的pos进行一个平滑处理。

                OperatePhoto.get_yolo_and_hand_pos_doublelump()  # 手工与yolo进行合并。
                #  这里需要进行一处修改。图片的替换进行控制。红绿黄三种颜色的灯。
                OperatePhoto.replace_video_to_store_doublelump(competed_video_url, frame_count)  # 进行图片替换，并进行将视频保存。
                # 对数据进行释放。
                OperatePhoto.clearData()
            else:

                print(cord)
                #  在标记之后，通过yolo与标记的坐标进行一个合并。

                ratio = OperatePhoto.get_raito_length_width(cord)  # 这里获取的是最大最小的长宽比。
                #  对标记的进行预测。
                predict_hand = OperatePhoto.predict_frame_tobond(cord)  # 进行预测之后，应该进行储存
                OperatePhoto.store_pos_txt('txt/lumppospredicate.txt', predict_hand)  # 将预测结果进行储存。
                #  并且需要将yolo中的探测进行一个寻找中心。然后才能够进行合并。
                OperatePhoto.yolo_detct_to_smooth(ratio, competed_video_url, frame_count)  # 对获取的pos进行一个平滑处理。
                #  对手工的pos与标记出来的pos进行一个合并。
                OperatePhoto.get_yolo_and_hand_pos()
                #  合并完成了之后，对其进行一个替换然后再修改的操作。
                OperatePhoto.replace_video_to_store(competed_video_url, frame_count)  # 进行图片替换，并进行将视频保存。
                # 对数据进行释放。
                OperatePhoto.clearData()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "显示excel"))

    def openfile(self):

        #  获取路径===================================================================
        self.excelName = QFileDialog.getOpenFileName(self, '选择文件', '', '*.csv;;*.xls;;All Files (*)')

        # print(openfile_name)
        global path_openfile_name

        #  获取路径====================================================================

        path_openfile_name = self.excelName[0]

    def creat_table_show1(self):
        if self.excelName == "":
            QMessageBox.information(self, '提示', '请输入excel')
        else:
            all_csv = []
            if len(path_openfile_name) > 0:
                with open(path_openfile_name, mode='r') as f:
                    data = csv.reader(f)
                    for row in data:
                        if row[1] == '进站' or row[1] == '出站' or row[1] == '过信号机':
                            csv_part = [row[1], row[2], row[3], row[5], row[6], row[7], row[8]]
                            all_csv.append(csv_part)

                self.tableWidget.setColumnCount(len(all_csv[0]))

                self.tableWidget.setRowCount(len(all_csv))
                head_csv = ['记录说明', '时间', '里程', '距离', '信号机', '信号', '速度']
                self.tableWidget.setHorizontalHeaderLabels(head_csv)
                print(all_csv)
                for i in range(0, len(all_csv)):
                    print(all_csv[i])
                    j = 0
                    for j in range(len(all_csv[0])):
                        input_table_items = str(all_csv[i][j])
                        print(all_csv[i][j])
                        newItem = QTableWidgetItem(input_table_items)
                        newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                        self.tableWidget.setItem(i, j, newItem)
                        # self.tableWidget.setItem(i, j, QTableWidgetItem())
                    # self.tableWidget.setCellWidget(i, j + 1, self.pushButton3)

    def creat_table_show(self):
        #  ===========读取表格，转换表格，===========================================
        if len(path_openfile_name) > 0:
            # input_table = pd.read_excel(path_openfile_name)
            input_table = pd.read_excel(path_openfile_name)

            # print(input_table)
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            # print(input_table_rows)
            # print(input_table_colunms)
            input_table_header = input_table.columns.values.tolist()
            # print(input_table_header)

            #  ===========读取表格，转换表格，============================================
            #  ======================给tablewidget设置行列表头============================

            self.tableWidget.setColumnCount(input_table_colunms)
            self.tableWidget.setRowCount(input_table_rows)
            self.tableWidget.setHorizontalHeaderLabels(input_table_header)

            #  ======================给tablewidget设置行列表头============================

            #  ================遍历表格每个元素，同时添加到tablewidget中========================
            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                # print(input_table_rows_values)
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                # print(input_table_rows_values_list)
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    # print(input_table_items_list)
                    # print(type(input_table_items_list))

                    #  ==============将遍历的元素添加到tablewidget中并显示=======================

                    input_table_items = str(input_table_items_list)
                    newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    self.tableWidget.setItem(i, j, newItem)

                    #  ================遍历表格每个元素，同时添加到tablewidget中========================
        else:
            self.centralWidget.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
