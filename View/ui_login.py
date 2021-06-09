# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_login.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(210, 250)
        Dialog.setMinimumSize(QtCore.QSize(210, 250))
        Dialog.setMaximumSize(QtCore.QSize(210, 250))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/mainwindow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_password = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_password.setMinimumSize(QtCore.QSize(140, 30))
        self.lineEdit_password.setMaximumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_password.setFont(font)
        self.lineEdit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_password.setObjectName("lineEdit_password")
        self.gridLayout.addWidget(self.lineEdit_password, 1, 1, 1, 1)
        self.lineEdit_username = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_username.setMinimumSize(QtCore.QSize(140, 30))
        self.lineEdit_username.setMaximumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_username.setFont(font)
        self.lineEdit_username.setObjectName("lineEdit_username")
        self.gridLayout.addWidget(self.lineEdit_username, 0, 1, 1, 1)
        self.label_resource = QtWidgets.QLabel(Dialog)
        self.label_resource.setMinimumSize(QtCore.QSize(40, 40))
        self.label_resource.setMaximumSize(QtCore.QSize(40, 40))
        self.label_resource.setText("")
        self.label_resource.setPixmap(QtGui.QPixmap(":/icon/resource.png"))
        self.label_resource.setScaledContents(True)
        self.label_resource.setObjectName("label_resource")
        self.gridLayout.addWidget(self.label_resource, 2, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.lineEdit_resource = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_resource.setMinimumSize(QtCore.QSize(140, 30))
        self.lineEdit_resource.setMaximumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_resource.setFont(font)
        self.lineEdit_resource.setObjectName("lineEdit_resource")
        self.gridLayout.addWidget(self.lineEdit_resource, 2, 1, 1, 1)
        self.pushButton_login = QtWidgets.QPushButton(Dialog)
        self.pushButton_login.setMinimumSize(QtCore.QSize(190, 40))
        self.pushButton_login.setMaximumSize(QtCore.QSize(190, 40))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(18)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_login.setFont(font)
        self.pushButton_login.setObjectName("pushButton_login")
        self.gridLayout.addWidget(self.pushButton_login, 4, 0, 1, 2)
        self.label_password = QtWidgets.QLabel(Dialog)
        self.label_password.setMinimumSize(QtCore.QSize(32, 40))
        self.label_password.setMaximumSize(QtCore.QSize(32, 40))
        self.label_password.setText("")
        self.label_password.setPixmap(QtGui.QPixmap(":/icon/password.png"))
        self.label_password.setScaledContents(True)
        self.label_password.setAlignment(QtCore.Qt.AlignCenter)
        self.label_password.setObjectName("label_password")
        self.gridLayout.addWidget(self.label_password, 1, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.label_username = QtWidgets.QLabel(Dialog)
        self.label_username.setMinimumSize(QtCore.QSize(40, 40))
        self.label_username.setMaximumSize(QtCore.QSize(40, 40))
        self.label_username.setText("")
        self.label_username.setPixmap(QtGui.QPixmap(":/icon/user.png"))
        self.label_username.setScaledContents(True)
        self.label_username.setObjectName("label_username")
        self.gridLayout.addWidget(self.label_username, 0, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.lineEdit_ip = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_ip.setMinimumSize(QtCore.QSize(140, 30))
        self.lineEdit_ip.setMaximumSize(QtCore.QSize(140, 30))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.lineEdit_ip.setFont(font)
        self.lineEdit_ip.setObjectName("lineEdit_ip")
        self.gridLayout.addWidget(self.lineEdit_ip, 3, 1, 1, 1)
        self.label_ip = QtWidgets.QLabel(Dialog)
        self.label_ip.setMinimumSize(QtCore.QSize(40, 40))
        self.label_ip.setMaximumSize(QtCore.QSize(40, 40))
        self.label_ip.setText("")
        self.label_ip.setPixmap(QtGui.QPixmap(":/icon/ip.png"))
        self.label_ip.setScaledContents(True)
        self.label_ip.setObjectName("label_ip")
        self.gridLayout.addWidget(self.label_ip, 3, 0, 1, 1, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.lineEdit_password.setPlaceholderText(_translate("Dialog", "密码"))
        self.lineEdit_username.setPlaceholderText(_translate("Dialog", "用户名"))
        self.lineEdit_resource.setPlaceholderText(_translate("Dialog", "资源名"))
        self.pushButton_login.setText(_translate("Dialog", "Log in"))
        self.lineEdit_ip.setPlaceholderText(_translate("Dialog", "IP地址"))
import source.source