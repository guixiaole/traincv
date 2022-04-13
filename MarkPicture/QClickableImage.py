from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class QClickableImage(QWidget):
    image_id = ''

    def __init__(self, width=0, height=0, pixmap=None, image_id=''):
        QWidget.__init__(self)

        self.layout = QVBoxLayout(self)
        self.label1 = QLabel()
        self.label1.setObjectName('label1')
        self.lable2 = QLabel()
        self.lable2.setObjectName('label2')
        self.width = width
        self.height = height
        self.pixmap = pixmap

        if self.width and self.height:
            self.resize(self.width, self.height)
        if self.pixmap:
            pixmap = self.pixmap.scaled(QSize(self.width, self.height), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label1.setPixmap(pixmap)
            self.label1.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.label1)

        if image_id:
            self.image_id = image_id
            self.lable2.setText(image_id)
            self.lable2.setAlignment(Qt.AlignCenter)
            #  让文字自适应大小
            self.lable2.adjustSize()
            self.layout.addWidget(self.lable2)
        self.setLayout(self.layout)

    clicked = pyqtSignal(object)
    rightClicked = pyqtSignal(object)

    def mouseReleaseEvent(self, event):
        print('55555555555555555')
        if event.button() == Qt.RightButton:

            # 鼠标右击
            self.rightClicked.emit(self.image_id)

        else:
            self.clicked.emit(self.image_id)

    def imageId(self):
        return self.image_id


