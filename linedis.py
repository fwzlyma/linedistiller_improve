# -*- coding: utf-8 -*-

import os
import cv2
import sys
from PIL import Image
import shutil
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from keras.models import load_model

R = 2 ** 3


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(938, 708)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 0, 2, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 0, 3, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 1, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout.addWidget(self.pushButton_4, 1, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label.setScaledContents(True)
        self.gridLayout.addWidget(self.label, 2, 0, 1, 2)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_2.setScaledContents(True)
        self.gridLayout.addWidget(self.label_2, 2, 2, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 938, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.selectFile)
        self.pushButton_2.clicked.connect(self.selectFolder)
        self.pushButton_3.clicked.connect(self.predict)
        self.pushButton_4.clicked.connect(self.mulPredict)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def selectFile(self):
        # 前处理
        work_dir = os.getcwd()
        self.pic_1 = os.path.join(work_dir, "input/example.png")
        if os.path.exists(self.pic_1):
            os.remove(self.pic_1)
        pic_2 = os.path.join(work_dir, "input/example2.png")
        if os.path.exists(pic_2):
            os.remove(pic_2)
        self.img_path = ""
        self.lineEdit.setText("")
        self.lineEdit_2.setText("")
        self.label.setPixmap(QtGui.QPixmap(""))
        self.label_2.setPixmap(QtGui.QPixmap(""))
        # 读文件
        self.img_path, filetype = QFileDialog.getOpenFileName(None, "选择路径", os.getcwd(), "Image Files(*.jpg *.png)")
        if self.img_path == "":
            print("未选择任何文件！！")
        else:
            try:
                # 处理文件
                self.lineEdit.setText(self.img_path)
                shutil.copy(self.img_path, self.pic_1)
                outfile, get_size = self.compress_image(self.pic_1,
                                                        pic_2)
                self.resize_image(self.pic_1, x_s=200)
                self.label.setPixmap(QtGui.QPixmap(pic_2))
                print(self.img_path)
            except Exception as e:
                print(e)

    def selectFolder(self):
        try:
            self.save_path = ""
            self.save_path = QFileDialog.getExistingDirectory(None, "选择文件夹", os.getcwd())
            if self.save_path == "":
                print("未选择文件夹")
            else:
                # 重命名图片
                self.method1()
                self.lineEdit_2.setText(self.save_path)
        except Exception as e:
            print(e)

    def predict(self):
        try:
            work_dir2 = os.getcwd()
            picR_1 = os.path.join(work_dir2, "output/example.png")
            picR_2 = os.path.join(work_dir2, "output/example2.png")
            if os.path.exists(picR_1):
                os.remove(picR_1)
            if os.path.exists(picR_2):
                os.remove(picR_2)
            model = load_model(os.path.join(work_dir2, "model.h5"))
            img = cv2.imread(self.pic_1)  # 从内存数据读入图片
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # perform brightness correction in tiles
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            img = clahe.apply(img)

            img_predict = cv2.resize(img, (img.shape[1] // R * R, img.shape[0] // R * R), interpolation=cv2.INTER_AREA)
            img_predict = np.reshape(img_predict, (1, img_predict.shape[0], img_predict.shape[1], 1))
            img_predict = img_predict.astype(np.float32) * 0.003383

            result = model.predict(img_predict, batch_size=1)[0]

            img_res = (result - np.mean(result) + 1.) * 255
            img_res = cv2.resize(img_res, (img.shape[1], img.shape[0]))
            out_dir = os.path.join(work_dir2, "output")
            cv2.imencode(".png", img_res)[1].tofile(
                os.path.join(out_dir, self.img_path.split("/")[-1]))
            # 处理文件
            shutil.copy(os.path.join(out_dir, self.img_path.split('/')[-1]),
                        picR_1)
            outfile, get_size = self.compress_image(picR_1,
                                                    picR_2)
            self.resize_image(picR_1, x_s=200)
            self.label_2.setPixmap(QtGui.QPixmap(picR_2))
            print("finish process single picture~")
            # cv2.imwrite(os.path.join('./output', "".join([self.img_path.split("/")[-1].replace(self.img_path.split("/")[-1].split(".")[-1],""), "jpg"])), img_res)
        except Exception as e:
            print(e)

    def mulPredict(self):
        try:
            print("start...")
            self.main()
            self.method2()
            print("finish mulprocess...")
        except Exception as e:
            print(e)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "imgProcessSystem1.0"))
        self.pushButton.setText(_translate("MainWindow", "选择文件"))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "文件路径->用于单张提取、展示"))
        self.pushButton_2.setText(_translate("MainWindow", "选择文件夹"))
        self.lineEdit_2.setPlaceholderText(_translate("MainWindow", "文件夹路径->用于批量提取,暂不展示"))
        self.pushButton_3.setText(_translate("MainWindow", "提取线图"))
        self.pushButton_4.setText(_translate("MainWindow", "批量提取线图"))

    # 将process 文件夹的文件由中文名改为英文名
    def method1(self):
        # os.remove("process")
        # shutil.copytree("input", "process")
        self.dict1 = {}
        work_dir3 = os.getcwd()
        path = os.path.join(work_dir3, "process")
        # print(os.listdir(path))
        i = 1
        for file in os.listdir(path):
            print(file)
            file_path = os.path.join(path, file)
            self.dict1["s" + i.__str__()] = file.split(".")[0]

            file_new = file.replace(
                file.replace(file.split(".")[-1], "")[0:len(file.replace(file.split(".")[-1], "")) - 1],
                "s" + i.__str__())
            file_new_path = os.path.join(path, file_new)
            os.rename(file_path, file_new_path)
            # print(file_new_path)
            i += 1

        filename = os.path.join(work_dir3, "name.txt")
        with open(filename, 'w') as f:
            for item in self.dict1.items():
                f.write(item[0] + ":" + item[1])
                f.write("\n")
                print(item)

    # 将process 文件夹的文件由英文名还原为中文名
    def method2(self):
        work_dir4 = os.getcwd()
        path = os.path.join(work_dir4, "process")
        # print(dict1)
        j = 1
        for file in os.listdir(path):
            # print(file)
            file_path = os.path.join(path, file)
            key = file.split(".")[0]
            ori_path = os.path.join(path, file.replace(key, self.dict1.get(key)))
            os.rename(file_path, ori_path)
            j += 1

    def main(self):
        work_dir5 = os.getcwd()
        model = load_model(os.path.join(work_dir5, "model.h5"))

        r = ""
        n = ""
        try:
            for root, dirs, files in os.walk(os.path.join(work_dir5, "process"), topdown=False):
                for name in files:
                    r = root
                    n = name
                    # name = "e" + name[0:len(name)]
                    print(os.path.join(root, name))

                    im = cv2.imread(os.path.join(root, name))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                    # perform brightness correction in tiles
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
                    im = clahe.apply(im)

                    im_predict = cv2.resize(im, (im.shape[1] // R * R, im.shape[0] // R * R),
                                            interpolation=cv2.INTER_AREA)
                    im_predict = np.reshape(im_predict, (1, im_predict.shape[0], im_predict.shape[1], 1))
                    # im_predict = ((im_predict/255)*220)/255
                    im_predict = im_predict.astype(np.float32) * 0.003383

                    result = model.predict(im_predict, batch_size=1)[0]

                    im_res = (result - np.mean(result) + 1.) * 255
                    im_res = cv2.resize(im_res, (im.shape[1], im.shape[0]))

                    # name = name[1:len(name)]
                    s1 = name.split(".")[-1]
                    s2 = name.replace(s1, "")
                    s3 = s2[0:len(s2) - 1]
                    new_name = self.dict1.get(s3)
                    print(new_name)
                    cv2.imencode(".png", im_res)[1].tofile(
                        os.path.join(os.path.join(work_dir5, "output"), ".".join([new_name, "png"])))
                    # cv2.imwrite(os.path.join('./output', ".".join([new_name,"jpg"])), im_res)
                    # os.remove(os.path.join(r, n))
        except Exception as e:
            print(e)
        # finally:
        #     os.remove(os.path.join(r, n))

    def get_size(self, file):
        # 获取文件大小:KB
        size = os.path.getsize(file)
        return size / 1024

    def get_outfile(self, infile, outfile):
        if outfile:
            return outfile
        dir, suffix = os.path.splitext(infile)
        outfile = '{}2{}'.format(dir, suffix)
        return outfile

    def compress_image(self, infile, outfile='', mb=100, step=10, quality=80):
        """不改变图片尺寸压缩到指定大小
        :param infile: 压缩源文件
        :param outfile: 压缩文件保存地址
        :param mb: 压缩目标，KB
        :param step: 每次调整的压缩比率
        :param quality: 初始压缩比率
        :return: 压缩文件地址，压缩文件大小
        """
        o_size = self.get_size(infile)
        if o_size <= mb:
            return infile
        outfile = self.get_outfile(infile, outfile)
        while o_size > mb:
            im = Image.open(infile)
            im.save(outfile, quality=quality)
            if quality - step < 0:
                break
            quality -= step
            o_size = self.get_size(outfile)
        return outfile, self.get_size(outfile)

    def resize_image(self, infile, outfile='', x_s=200):
        """修改图片尺寸
        :param infile: 图片源文件
        :param outfile: 重设尺寸文件保存地址
        :param x_s: 设置的宽度
        :return:
        """
        im = Image.open(infile)
        x, y = im.size
        y_s = int(y * x_s / x)
        out = im.resize((x_s, y_s), Image.ANTIALIAS)
        outfile = self.get_outfile(infile, outfile)
        out.save(outfile)



# 主方法，程序从此处启动PyQt设计的窗体
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()  # 创建窗体对象
    ui = Ui_MainWindow()  # 创建PyQt设计的窗体对象
    ui.setupUi(MainWindow)  # 调用PyQt窗体的方法对窗体对象进行初始化设置
    MainWindow.show()  # 显示窗体
    sys.exit(app.exec_())  # 程序关闭时退出进程
