# Third-party library
from multiprocessing import Process

# User-defined library
from Common import utils


# TODO：IO Process现阶段只用来采集数据
class IOProcess:
    def __init__(self, s2p_queue, d2p_queue, ):
        self.IO_process = Process(target=self.io_handle, args=(s2p_queue, d2p_queue, ))

    # 封装进程启动API
    def open(self):
        if not self.IO_process.is_alive():
            self.IO_process.start()

    # 封装进程结束API
    def close(self):
        if self.IO_process.is_alive():
            self.IO_process.terminate()

    @staticmethod
    def io_handle(s2p_queue, d2p_queue):
        s2p_queue = s2p_queue
        d2p_queue = d2p_queue
        while True:
            if not s2p_queue.empty() and not d2p_queue.empty():
                print("I queue:", d2p_queue.qsize())
                print("S queue:", s2p_queue.qsize())
                image_data = d2p_queue.get()
                direction, row, col = s2p_queue.get()
                # print("D:{}, R:{}, C:{}".format(direction, row, col))
                utils.save_image_by_needle(image_data, direction, col, row)
