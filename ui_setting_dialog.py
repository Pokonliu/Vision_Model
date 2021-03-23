# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_setting_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_dialog(object):
    def setupUi(self, dialog):
        dialog.setObjectName("dialog")
        dialog.resize(240, 479)
        dialog.setMinimumSize(QtCore.QSize(240, 479))
        dialog.setMaximumSize(QtCore.QSize(240, 479))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/mainwindow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        dialog.setWindowIcon(icon)
        self.verticalLayout = QtWidgets.QVBoxLayout(dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.input_setting_groupbox = QtWidgets.QGroupBox(dialog)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        self.input_setting_groupbox.setFont(font)
        self.input_setting_groupbox.setObjectName("input_setting_groupbox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.input_setting_groupbox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.source_output_label = QtWidgets.QLabel(self.input_setting_groupbox)
        self.source_output_label.setMinimumSize(QtCore.QSize(0, 20))
        self.source_output_label.setMaximumSize(QtCore.QSize(16777215, 20))
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        self.source_output_label.setFont(font)
        self.source_output_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.source_output_label.setText("")
        self.source_output_label.setAlignment(QtCore.Qt.AlignCenter)
        self.source_output_label.setObjectName("source_output_label")
        self.verticalLayout_2.addWidget(self.source_output_label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.image_radio_button = QtWidgets.QRadioButton(self.input_setting_groupbox)
        self.image_radio_button.setObjectName("image_radio_button")
        self.horizontalLayout.addWidget(self.image_radio_button)
        self.image_tool_button = QtWidgets.QToolButton(self.input_setting_groupbox)
        self.image_tool_button.setEnabled(False)
        self.image_tool_button.setObjectName("image_tool_button")
        self.horizontalLayout.addWidget(self.image_tool_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.video_radio_button = QtWidgets.QRadioButton(self.input_setting_groupbox)
        self.video_radio_button.setObjectName("video_radio_button")
        self.horizontalLayout_4.addWidget(self.video_radio_button)
        self.video_tool_button = QtWidgets.QToolButton(self.input_setting_groupbox)
        self.video_tool_button.setEnabled(False)
        self.video_tool_button.setObjectName("video_tool_button")
        self.horizontalLayout_4.addWidget(self.video_tool_button)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.camera_radio_button = QtWidgets.QRadioButton(self.input_setting_groupbox)
        self.camera_radio_button.setObjectName("camera_radio_button")
        self.horizontalLayout_2.addWidget(self.camera_radio_button)
        self.camera_combo_box = QtWidgets.QComboBox(self.input_setting_groupbox)
        self.camera_combo_box.setEnabled(False)
        self.camera_combo_box.setObjectName("camera_combo_box")
        self.camera_combo_box.addItem("")
        self.camera_combo_box.setItemText(0, "")
        self.horizontalLayout_2.addWidget(self.camera_combo_box)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.search_device_push_button = QtWidgets.QPushButton(self.input_setting_groupbox)
        self.search_device_push_button.setObjectName("search_device_push_button")
        self.horizontalLayout_3.addWidget(self.search_device_push_button)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addWidget(self.input_setting_groupbox)
        self.serial_port_settin_groupbox = QtWidgets.QGroupBox(dialog)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        self.serial_port_settin_groupbox.setFont(font)
        self.serial_port_settin_groupbox.setObjectName("serial_port_settin_groupbox")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.serial_port_settin_groupbox)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.serial_output_label = QtWidgets.QLabel(self.serial_port_settin_groupbox)
        self.serial_output_label.setMinimumSize(QtCore.QSize(0, 20))
        self.serial_output_label.setMaximumSize(QtCore.QSize(16777215, 20))
        self.serial_output_label.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.serial_output_label.setText("")
        self.serial_output_label.setAlignment(QtCore.Qt.AlignCenter)
        self.serial_output_label.setObjectName("serial_output_label")
        self.verticalLayout_3.addWidget(self.serial_output_label)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.search_serial_push_button = QtWidgets.QPushButton(self.serial_port_settin_groupbox)
        self.search_serial_push_button.setObjectName("search_serial_push_button")
        self.horizontalLayout_8.addWidget(self.search_serial_push_button)
        self.serial_port_combo_box = QtWidgets.QComboBox(self.serial_port_settin_groupbox)
        self.serial_port_combo_box.setObjectName("serial_port_combo_box")
        self.horizontalLayout_8.addWidget(self.serial_port_combo_box)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.baud_rate_label = QtWidgets.QLabel(self.serial_port_settin_groupbox)
        self.baud_rate_label.setEnabled(True)
        self.baud_rate_label.setAlignment(QtCore.Qt.AlignCenter)
        self.baud_rate_label.setObjectName("baud_rate_label")
        self.horizontalLayout_7.addWidget(self.baud_rate_label)
        self.baud_rate_combo_box = QtWidgets.QComboBox(self.serial_port_settin_groupbox)
        self.baud_rate_combo_box.setObjectName("baud_rate_combo_box")
        self.baud_rate_combo_box.addItem("")
        self.baud_rate_combo_box.addItem("")
        self.baud_rate_combo_box.addItem("")
        self.baud_rate_combo_box.addItem("")
        self.baud_rate_combo_box.addItem("")
        self.horizontalLayout_7.addWidget(self.baud_rate_combo_box)
        self.verticalLayout_3.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.byte_size_label = QtWidgets.QLabel(self.serial_port_settin_groupbox)
        self.byte_size_label.setEnabled(True)
        self.byte_size_label.setAlignment(QtCore.Qt.AlignCenter)
        self.byte_size_label.setObjectName("byte_size_label")
        self.horizontalLayout_6.addWidget(self.byte_size_label)
        self.byte_size_combo_box = QtWidgets.QComboBox(self.serial_port_settin_groupbox)
        self.byte_size_combo_box.setObjectName("byte_size_combo_box")
        self.byte_size_combo_box.addItem("")
        self.byte_size_combo_box.addItem("")
        self.byte_size_combo_box.addItem("")
        self.byte_size_combo_box.addItem("")
        self.horizontalLayout_6.addWidget(self.byte_size_combo_box)
        self.verticalLayout_3.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.stop_bits_label = QtWidgets.QLabel(self.serial_port_settin_groupbox)
        self.stop_bits_label.setEnabled(True)
        self.stop_bits_label.setAlignment(QtCore.Qt.AlignCenter)
        self.stop_bits_label.setObjectName("stop_bits_label")
        self.horizontalLayout_5.addWidget(self.stop_bits_label)
        self.stop_bits_combo_box = QtWidgets.QComboBox(self.serial_port_settin_groupbox)
        self.stop_bits_combo_box.setObjectName("stop_bits_combo_box")
        self.stop_bits_combo_box.addItem("")
        self.stop_bits_combo_box.addItem("")
        self.stop_bits_combo_box.addItem("")
        self.horizontalLayout_5.addWidget(self.stop_bits_combo_box)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.open_serial_push_button = QtWidgets.QPushButton(self.serial_port_settin_groupbox)
        self.open_serial_push_button.setObjectName("open_serial_push_button")
        self.horizontalLayout_9.addWidget(self.open_serial_push_button)
        self.close_serial_push_button = QtWidgets.QPushButton(self.serial_port_settin_groupbox)
        self.close_serial_push_button.setEnabled(False)
        self.close_serial_push_button.setObjectName("close_serial_push_button")
        self.horizontalLayout_9.addWidget(self.close_serial_push_button)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.verticalLayout.addWidget(self.serial_port_settin_groupbox)
        self.led_control_groupbox = QtWidgets.QGroupBox(dialog)
        self.led_control_groupbox.setEnabled(True)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        self.led_control_groupbox.setFont(font)
        self.led_control_groupbox.setObjectName("led_control_groupbox")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.led_control_groupbox)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.led_brightness_slider = QtWidgets.QSlider(self.led_control_groupbox)
        self.led_brightness_slider.setEnabled(False)
        self.led_brightness_slider.setStyleSheet("QSlider::groove:horizontal {\n"
"border: 1px solid #bbb;\n"
"background: white;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::sub-page:horizontal {\n"
"background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,\n"
"    stop: 0 #66e, stop: 1 #bbf);\n"
"background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,\n"
"    stop: 0 #bbf, stop: 1 #55f);\n"
"border: 1px solid #777;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::add-page:horizontal {\n"
"background: #fff;\n"
"border: 1px solid #777;\n"
"height: 10px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::handle:horizontal {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #eee, stop:1 #ccc);\n"
"border: 1px solid #777;\n"
"width: 13px;\n"
"margin-top: -2px;\n"
"margin-bottom: -2px;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::handle:horizontal:hover {\n"
"background: qlineargradient(x1:0, y1:0, x2:1, y2:1,\n"
"    stop:0 #fff, stop:1 #ddd);\n"
"border: 1px solid #444;\n"
"border-radius: 4px;\n"
"}\n"
" \n"
"QSlider::sub-page:horizontal:disabled {\n"
"background: #bbb;\n"
"border-color: #999;\n"
"}\n"
" \n"
"QSlider::add-page:horizontal:disabled {\n"
"background: #eee;\n"
"border-color: #999;\n"
"}\n"
" \n"
"QSlider::handle:horizontal:disabled {\n"
"background: #eee;\n"
"border: 1px solid #aaa;\n"
"border-radius: 4px;\n"
"}")
        self.led_brightness_slider.setOrientation(QtCore.Qt.Horizontal)
        self.led_brightness_slider.setObjectName("led_brightness_slider")
        self.horizontalLayout_10.addWidget(self.led_brightness_slider)
        self.verticalLayout.addWidget(self.led_control_groupbox)
        self.buttonBox = QtWidgets.QDialogButtonBox(dialog)
        font = QtGui.QFont()
        font.setFamily("等线")
        font.setBold(True)
        font.setWeight(75)
        self.buttonBox.setFont(font)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(True)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.retranslateUi(dialog)
        QtCore.QMetaObject.connectSlotsByName(dialog)

    def retranslateUi(self, dialog):
        _translate = QtCore.QCoreApplication.translate
        dialog.setWindowTitle(_translate("dialog", "源设置"))
        self.input_setting_groupbox.setTitle(_translate("dialog", "Input Source Setting"))
        self.image_radio_button.setText(_translate("dialog", "Image Source"))
        self.image_tool_button.setText(_translate("dialog", "..."))
        self.video_radio_button.setText(_translate("dialog", "Video Source"))
        self.video_tool_button.setText(_translate("dialog", "..."))
        self.camera_radio_button.setText(_translate("dialog", "Camera Sourece"))
        self.search_device_push_button.setText(_translate("dialog", "Searching Device"))
        self.serial_port_settin_groupbox.setTitle(_translate("dialog", "Serial Port Setting"))
        self.search_serial_push_button.setText(_translate("dialog", "Searching Serial"))
        self.baud_rate_label.setText(_translate("dialog", "Baud Rate"))
        self.baud_rate_combo_box.setItemText(0, _translate("dialog", "921600"))
        self.baud_rate_combo_box.setItemText(1, _translate("dialog", "128000"))
        self.baud_rate_combo_box.setItemText(2, _translate("dialog", "115200"))
        self.baud_rate_combo_box.setItemText(3, _translate("dialog", "14400"))
        self.baud_rate_combo_box.setItemText(4, _translate("dialog", "9600"))
        self.byte_size_label.setText(_translate("dialog", "Byte Size"))
        self.byte_size_combo_box.setItemText(0, _translate("dialog", "8"))
        self.byte_size_combo_box.setItemText(1, _translate("dialog", "7"))
        self.byte_size_combo_box.setItemText(2, _translate("dialog", "6"))
        self.byte_size_combo_box.setItemText(3, _translate("dialog", "5"))
        self.stop_bits_label.setText(_translate("dialog", "Stop Bits"))
        self.stop_bits_combo_box.setItemText(0, _translate("dialog", "1"))
        self.stop_bits_combo_box.setItemText(1, _translate("dialog", "1.5"))
        self.stop_bits_combo_box.setItemText(2, _translate("dialog", "2"))
        self.open_serial_push_button.setText(_translate("dialog", "Open"))
        self.close_serial_push_button.setText(_translate("dialog", "Close"))
        self.led_control_groupbox.setTitle(_translate("dialog", "LED Control"))
import source.source
