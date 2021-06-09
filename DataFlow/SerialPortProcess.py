# Third-party library
import multiprocessing
import time
import threading
from multiprocessing import Process

# User-defined library
from Common.const import const
from DataFlow.SerialPort import Communication


# 多线程处理全双工通信
class SerialPortSend(threading.Thread):
    def __init__(self, device, v2s_queue, serial_port_changed_event: threading.Event):
        super().__init__()
        self.serial_port = device
        self.serial_queue = v2s_queue
        self.serial_port_changed_event = serial_port_changed_event

    def run(self) -> None:
        while True:
            if self.serial_port_changed_event.is_set() and not self.serial_queue.empty():
                self.serial_port.write(self.serial_queue.get())


class SerialPortReceive(threading.Thread):
    def __init__(self, device, s2v_queue, s2p_queue, serial_port_changed_event: threading.Event):
        super().__init__()
        self.serial_port = device
        self.serial_upload_queue = s2v_queue
        self.serial_queue = s2p_queue
        self.serial_port_changed_event = serial_port_changed_event
        self.communication_flag = const.SEARCHING_HEADER
        self.data_length = 0
        self.data_command = 0

        # TODO：测试参数，后期删除
        self.serial_receive_cnt = 0

    def run(self) -> None:
        communication_process_function_point = {const.SEARCHING_HEADER: self.searching_header,
                                                const.SEARCHING_LENGTH: self.searching_length,
                                                const.SEARCHING_COMMAND: self.searching_command,
                                                const.SEARCHING_DATA: self.searching_data}
        while True:
            if self.serial_port_changed_event.is_set():
                communication_process_function_point.get(self.communication_flag)()

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
            # TODO: 数据通过队列传输
            self.serial_queue.put([direction, row, col])
            self.serial_upload_queue.put([data.hex(), analyze])
            # TODO：测试代码，后期删除
            self.serial_receive_cnt += 1
            # print("Serial receive count: {}, data:{}".format(self.serial_receive_cnt, analyze))

            self.data_length = 0
            self.data_command = 0
            self.communication_flag = const.SEARCHING_HEADER


class SerialPortProcess:
    def __init__(self, v2s_queue, s2v_queue, s2p_queue, v2s_params_dict, v2s_control_flag, sv_control_pipe):
        self.serial_port_process = Process(target=self.serial_port_handle, args=(v2s_queue, s2v_queue, s2p_queue, v2s_params_dict, v2s_control_flag, sv_control_pipe, ))

    # 封装进程启动API
    def open(self):
        if not self.serial_port_process.is_alive():
            self.serial_port_process.start()

    # 封装进程结束API
    def close(self):
        if self.serial_port_process.is_alive():
            self.serial_port_process.terminate()

    @staticmethod
    def serial_port_handle(v2s_Q, s2v_Q, s2p_Q, param_D, ctrl_F, ctrl_P):
        def display_img_data():
            print('TEst')

        v2s_queue = v2s_Q
        s2v_queue = s2v_Q
        s2p_queue = s2p_Q
        serial_port_flag = ctrl_F
        serial_init_params = param_D
        sv_control_pipe = ctrl_P

        serial_port_change_event = threading.Event()
        serial_port = None
        serial_port_send_thread = None
        serial_port_receive_thread = None

        while True:
            # 空状态
            if serial_port_flag.value == const.SERIAL_PORT_IDLE:
                time.sleep(0.01)

            # 切换状态
            elif serial_port_flag.value == const.SERIAL_PORT_CHANGE:
                # 第一次初始化
                if not serial_port_send_thread and not serial_port_receive_thread:
                    # 初始化串口类
                    try:
                        serial_port = Communication(serial_init_params["com"], serial_init_params["baud_rate"], serial_init_params["byte_size"], serial_init_params["stop_bits"])
                    except Exception as error:
                        sv_control_pipe.send(["S2V", const.OPEN_SERIAL_PORT, False, error])
                    else:
                        # 创建串口读写线程
                        serial_port_send_thread = SerialPortSend(device=serial_port, v2s_queue=v2s_queue, serial_port_changed_event=serial_port_change_event)
                        serial_port_receive_thread = SerialPortReceive(device=serial_port, s2v_queue=s2v_queue, s2p_queue=s2p_queue, serial_port_changed_event=serial_port_change_event)
                        # 启动读写线程
                        serial_port_change_event.set()
                        serial_port_send_thread.start()
                        serial_port_receive_thread.start()
                        sv_control_pipe.send(["S2V", const.OPEN_SERIAL_PORT, True, "Serial Port init successfully"])

                else:
                    serial_port_change_event.clear()
                    serial_port.close()
                    try:
                        serial_port.switch_serial_port(serial_init_params["com"], serial_init_params["baud_rate"], serial_init_params["byte_size"], serial_init_params["stop_bits"])
                    except Exception as error:
                        sv_control_pipe.send(["S2V", const.OPEN_SERIAL_PORT, False, error])
                        # TODO:当切换串口失败时，串口状态回复到切换之前,此处逻辑需要继续优化
                        serial_port.open()
                    else:
                        serial_port_change_event.set()
                        sv_control_pipe.send(["S2V", const.OPEN_SERIAL_PORT, True, "Serial Port init successfully"])
                serial_port_flag.value = const.SERIAL_PORT_RUN

            # 执行状态
            elif serial_port_flag.value == const.SERIAL_PORT_RUN:
                # TODO:在这里处理控制队列中的信息,不需要单独启动线程来处理
                direction, info_type, res, command = sv_control_pipe.recv()
                if direction == "V2S" and info_type == const.CLOSE_SERIAL_PORT:
                    serial_port_change_event.clear()
                    serial_port.close()
                    sv_control_pipe.send(["S2V", const.CLOSE_SERIAL_PORT, True, "Serial port close successfully"])
                    serial_port_flag.value = const.SERIAL_PORT_IDLE
                # TODO:后续可以扩展其他控制信息



# if __name__ == '__main__':
#     v2s_queue_T = multiprocessing.Queue()
#     s2p_queue_T = multiprocessing.Queue()
#     serial_port_flag_T = multiprocessing.Value('i', 1)
#     serial_init_params_T = multiprocessing.Manager().dict({"com": "COM2", "baud_rate": 256000, "byte_size": 8, "stop_bits": 1})
#
#     SP = SerialPortProcess(v2s_queue_T, s2p_queue_T, serial_port_flag_T, serial_init_params_T)
#     SP.open()
#     time.sleep(2)
#     serial_port_flag_T.value = const.SERIAL_PORT_CHANGE
#
#     while True:
#         for i in range(10):
#             time.sleep(0.5)
#             v2s_queue_T.put(bytes.fromhex("ABCDEF1216487ACB"))
#         else:
#             serial_init_params_T["baud_rate"] = 9600
#             serial_port_flag_T.value = const.SERIAL_PORT_CHANGE
#         # 模拟串口更新
#         for i in range(1000):
#             time.sleep(0.5)
#             v2s_queue_T.put(bytes.fromhex("ABCDEF1216487ACB"))