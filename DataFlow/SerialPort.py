# Third-party library
import serial
import serial.tools.list_ports

# User-defined library
from Common.Singleton import SingletonType


# 串口基类
class Communication(metaclass=SingletonType):
    def __init__(self, com, bps, bts, stb, timeout=5, parity="N"):
        # 串口参数
        self.port = None
        self.bps = None
        self.bts = None
        self.stb = None
        self.timeout = None
        self.parity = None
        # 串口实例
        self.com = None
        # 串口初始化
        self.switch_serial_port(com, bps, bts, stb, timeout, parity)

    # 打印设备基本信息
    def collect_info(self):
        print("设备信息:")
        print("  设备名字:{}".format(self.com.name))
        print("  读或者写端口:{}".format(self.com.port))
        print("  波特率:{}".format(self.com.baudrate))
        print("  字节大小:{}".format(self.com.bytesize))
        print("  校验位:{}".format(self.com.parity))
        print("  停止位:{}".format(self.com.stopbits))
        print("  读超时设置:{}".format(self.com.timeout))
        print("  写超时:{}".format(self.com.writeTimeout))
        print("  软件流控XON/XOFF:{}".format(self.com.xonxoff))
        print("  硬件流控RTS/CTS:{}".format(self.com.rtscts))
        print("  硬件流控DSR/DTR:{}".format(self.com.dsrdtr))
        print("  字符间隔超时:{}".format(self.com.interCharTimeout))

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
            return True, self.com.read(size)
        return False, None

    def write(self, data):
        self.com.write(data)

    def is_open(self):
        return self.com.isOpen()

    # 打印可用串口列表
    @classmethod
    def com_check(cls):
        com_list = []
        port_list = list(serial.tools.list_ports.comports())
        for port in port_list:
            com_list.append(port.name)
        return com_list

    def switch_serial_port(self, com, bps, bts, stb, timeout=5, parity="N"):
        # 先设置串口实例，之后再修改参数，方便回滚
        self.com = serial.Serial(port=com, baudrate=bps, bytesize=bts, stopbits=stb, timeout=timeout, parity=parity)
        self.com.set_buffer_size(rx_size=500000, tx_size=500000)
        self.port = com
        self.bps = bps
        self.bts = bts
        self.stb = stb
        self.timeout = timeout
        self.parity = parity
