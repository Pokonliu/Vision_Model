import time
import serial
import serial.tools.list_ports
from PyQt5.QtCore import *
from const import const


class Communication:
    def __init__(self, com, bps, bts, stb, timeout=5, parity="N"):
        self.port = com
        self.bps = bps
        self.bts = bts
        self.stb = stb
        self.timeout = timeout
        try:
            self.com = serial.Serial(port=self.port, baudrate=self.bps, bytesize=self.bts, stopbits=self.stb, timeout=self.timeout, parity=parity)
            self.com.set_buffer_size(rx_size=500000, tx_size=500000)
        except Exception as error:
            print("Serial Port occurred {}".format(error))

    # 打印设备基本信息
    def collect_info(self):
        print("设备名字:{}".format(self.com.name))
        print("读或者写端口:{}".format(self.com.port))
        print("波特率:{}".format(self.com.baudrate))
        print("字节大小:{}".format(self.com.bytesize))
        print("校验位:{}".format(self.com.parity))
        print("停止位:{}".format(self.com.stopbits))
        print("读超时设置:{}".format(self.com.timeout))
        print("写超时:{}".format(self.com.writeTimeout))
        print("软件流控XON/XOFF:{}".format(self.com.xonxoff))
        print("硬件流控RTS/CTS:{}".format(self.com.rtscts))
        print("硬件流控DSR/DTR:{}".format(self.com.dsrdtr))
        print("字符间隔超时:{}".format(self.com.interCharTimeout))

    def name(self):
        if self.com:
            return self.com.name
        else:
            return ""

    # 打开串口
    def open(self):
        self.com.open()

    # 关闭串口
    def close(self):
        self.com.close()

    def read(self, size=1):
        if self.com.in_waiting:
            # print("cache:", self.com.in_waiting)
            return True, self.com.read(size)
        return False, None

    def write(self, data):
        self.com.write(data)

    # 打印可用串口列表
    @staticmethod
    def com_check():
        com_list = []
        port_list = list(serial.tools.list_ports.comports())
        for port in port_list:
            com_list.append(port.name)
        return com_list


class SerialPortThread(QThread):
    # TODO 设置串口后，关闭，再打开奔溃(必现)
    serial_port_receive_signal = pyqtSignal(list)

    def __init__(self, UI):
        super().__init__()
        self.super_UI = UI
        self.serial_port = None
        self.serial_port_flag = const.SERIAL_PORT_IDLE
        self.communication_flag = const.SEARCHING_HEADER

        self.super_UI.serial_port_send_signal.connect(self.send_data)

        self.data_length = 0
        self.data_command = 0
        self.data_list = []

        self.counts = 0

    def run(self) -> None:
        serial_port_function_point = {const.SERIAL_PORT_IDLE: self.serial_port_idle,
                                      const.SERIAL_PORT_CHANGE: self.serial_port_change,
                                      const.SERIAL_PORT_RUN: self.serial_port_run}
        # TODO: 业务逻辑，while循环中添加循环读写操作，以及上传信息解析处理
        while self.serial_port_flag:
            serial_port_function_point.get(self.serial_port_flag)()

    @staticmethod
    def serial_port_idle():
        time.sleep(0.01)

    # 注意Python引用计数带来的BUG
    def serial_port_change(self):
        # TODO: 第一次设置USB但不设置串口，第二次设置串口奔溃(必现BUG)
        if self.super_UI.serial_port:
            # 删除之前的引用计数
            if self.serial_port != self.super_UI.serial_port:
                del self.serial_port
                self.serial_port = self.super_UI.serial_port
            self.serial_port_flag = const.SERIAL_PORT_RUN
        else:
            self.serial_port = None
            self.serial_port_flag = const.SERIAL_PORT_IDLE

    def serial_port_run(self):
        try:
            # TODO: 按照串口协议解析数据
            self.receive_data_process()
        except Exception as error:
            print("Serial port error occurred {}".format(error))

    def receive_data_process(self):
        communication_process_function_point = {const.SEARCHING_HEADER: self.searching_header,
                                                const.SEARCHING_LENGTH: self.searching_length,
                                                const.SEARCHING_COMMAND: self.searching_command,
                                                const.SEARCHING_DATA: self.searching_data}
        communication_process_function_point.get(self.communication_flag)()

    def send_data(self, data):
        self.serial_port.write(data)

    def searching_header(self):
        res, data = self.serial_port.read()
        if res:
            if int.from_bytes(data, byteorder='big', signed=False) == 170:
                self.communication_flag = const.SEARCHING_LENGTH

    def searching_length(self):
        res, data = self.serial_port.read()
        if res:
            self.data_length = int.from_bytes(data, byteorder='big', signed=False)
            self.data_length -= 1
            self.communication_flag = const.SEARCHING_COMMAND

    def searching_command(self):
        res, data = self.serial_port.read()
        if res:
            self.data_command = int.from_bytes(data, byteorder='big', signed=False)
            self.data_length -= 1
            self.communication_flag = const.SEARCHING_DATA

    def searching_data(self):
        res, data = self.serial_port.read(self.data_length)
        if res:
            # TODO: 数据处理
            # self.counts += 1

            row = int.from_bytes(data[: 4], byteorder='big', signed=False)
            col = int.from_bytes(data[4: 6], byteorder='big', signed=True)
            direction = "S"
            if self.data_command == 160:
                direction = "L"
            elif self.data_command == 161:
                direction = "R"
            elif self.data_command == 162:
                direction = "TR"
            elif self.data_command == 163:
                direction = "TL"
            analyze = "行:{} 列:{}, 方向:{}".format(row, col, direction)
            self.super_UI.data_flow_thread.needle_direction = direction
            self.super_UI.data_flow_thread.needle_row = row
            self.super_UI.data_flow_thread.needle_col = col
            self.serial_port_receive_signal.emit([data.hex(), analyze])

            self.data_length = 0
            self.data_command = 0
            self.communication_flag = const.SEARCHING_HEADER
