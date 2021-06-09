# Third-party library
import os
import time
import cv2.cv2 as cv2
from multiprocessing import Process
from apscheduler.schedulers.background import BackgroundScheduler

# User-defined library
from Common.const import const
from DataFlow.HikCameraSource import HikCamera
from DataFlow.USBCameraSource import USBCamera
from DataFlow.VideoSource import Video


class DataFlowProcess:
    def __init__(self, d2p_queue, d2v_queue, v2d_params_str, v2d_control_flag, dv_control_pipe):
        self.data_flow_process = Process(target=self.data_flow_handle, args=(d2p_queue, d2v_queue, v2d_params_str, v2d_control_flag, dv_control_pipe, ))

    # 封装进程启动API
    def open(self):
        if not self.data_flow_process.is_alive():
            self.data_flow_process.start()

    # 封装进程结束API
    def close(self):
        if self.data_flow_process.is_alive():
            self.data_flow_process.terminate()

    @staticmethod
    def data_flow_handle(d2p_Q, d2v_Q, param_S, ctrl_F, ctrl_P):
        def data_flow_idle():
            time.sleep(0.01)

        def data_flow_change():
            nonlocal source_name
            nonlocal source_type
            nonlocal source_control_flag
            nonlocal dv_control_pipe
            if source_name.value == "":
                dv_control_pipe.send(["D2V", const.SOURCE_CHANGE, "None", "You didn't choose anything"])
                source_type = const.SOURCE_TYPE_NONE
                source_control_flag.value = const.DATA_FLOW_IDLE

            elif os.path.exists(source_name.value) and source_name.value.endswith((".jpg", ".gif", ".png", ".bmp")):
                dv_control_pipe.send(["D2V", const.STATUS_BAR_SHOW, "IMAGE", "Input source changed to Image"])
                source_type = const.SOURCE_TYPE_IMAGE
                source_control_flag.value = const.DATA_FLOW_INIT

            elif os.path.exists(source_name.value) and source_name.value.endswith((".avi", ".mp4", ".wmv", ".rmvb")):
                dv_control_pipe.send(["D2V", const.STATUS_BAR_SHOW, "VIDEO", "Input source changed to Video"])
                source_type = const.SOURCE_TYPE_VIDEO
                source_control_flag.value = const.DATA_FLOW_INIT

            elif source_name.value.split()[0] == "USB":
                dv_control_pipe.send(["D2V", const.STATUS_BAR_SHOW, "USB", "Input source changed to USB {}".format(source_name.value.split()[1])])
                source_type = const.SOURCE_TYPE_USB
                source_control_flag.value = const.DATA_FLOW_INIT

            elif source_name.value.split()[0] == "GigE":
                dv_control_pipe.send(["D2V", const.STATUS_BAR_SHOW, "GigE", "Input source changed to GigE {}".format(source_name.value.split()[1])])
                source_type = const.SOURCE_TYPE_GIGE
                source_control_flag.value = const.DATA_FLOW_INIT

        def data_flow_init():
            nonlocal source_type
            init_function_point = {const.SOURCE_TYPE_IMAGE: image_source_init,
                                   const.SOURCE_TYPE_VIDEO: video_source_init,
                                   const.SOURCE_TYPE_USB: USB_source_init,
                                   const.SOURCE_TYPE_GIGE: GigE_source_init}
            init_function_point.get(source_type)()

        def data_flow_run():
            nonlocal image_data
            nonlocal data_instance
            nonlocal d2p_queue
            assert isinstance(data_instance, HikCamera) or isinstance(data_instance, USBCamera) or isinstance(data_instance, Video)
            res, image_data = data_instance.run()
            if res:
                d2p_queue.put(image_data)

        def data_flow_release():
            nonlocal data_instance
            nonlocal source_control_flag
            assert isinstance(data_instance, HikCamera) or isinstance(data_instance, USBCamera) or isinstance(data_instance, Video)
            data_instance.release()
            data_instance = None
            source_control_flag.value = const.DATA_FLOW_CHANGE

        '''source init function'''
        def image_source_init():
            nonlocal image_data
            nonlocal source_name
            nonlocal d2v_queue
            nonlocal source_control_flag
            image_data = cv2.imread(source_name.value, cv2.IMREAD_COLOR)
            d2v_queue.put(image_data)
            # 初始化完成进入空闲状态
            source_control_flag.value = const.DATA_FLOW_IDLE

        def video_source_init():
            nonlocal data_instance
            nonlocal source_name
            nonlocal source_control_flag
            nonlocal d2v_queue
            nonlocal d2p_queue
            nonlocal dv_control_pipe
            data_instance = Video(video_path=source_name.value, d2v_queue=d2v_queue, d2p_queue=d2p_queue, source_control_flag=source_control_flag, dv_control_pipe=dv_control_pipe)
            source_control_flag.value = const.DATA_FLOW_RUN

        def USB_source_init():
            nonlocal data_instance
            nonlocal source_name
            nonlocal source_control_flag
            data_instance = USBCamera(device_name=int(source_name.value.split()[1]))
            source_control_flag.value = const.DATA_FLOW_RUN

        def GigE_source_init():
            nonlocal data_instance
            nonlocal source_name
            nonlocal source_control_flag
            data_instance = HikCamera(device_name=int(source_name.value.split()[1]),
                                      trigger_mode=const.TRIGGER_MODE_OFF,
                                      frame_rate_control=const.FRAME_RATE_CONTROL_OFF,
                                      trigger_source=const.TRIGGER_SOURCE_LINE0,
                                      trigger_polarity=const.TRIGGER_POLARITY_RISING_EDGE,
                                      anti_shake_time=const.ANTI_SHAKE_TIME,
                                      GEV=const.GEV,
                                      exposure_time=const.EXPOSURE_TIME,
                                      cache_capacity=const.CACHE_CAPACITY)
            source_control_flag.value = const.DATA_FLOW_RUN

        def display_img_data():
            nonlocal image_data
            nonlocal source_control_flag
            nonlocal d2v_queue
            nonlocal data_instance
            if isinstance(data_instance, Video):
                data_instance.schedule_trigger = True
            elif image_data is not None and source_control_flag.value == const.DATA_FLOW_RUN:
                d2v_queue.put(image_data)

        d2p_queue = d2p_Q
        d2v_queue = d2v_Q
        source_name = param_S
        source_control_flag = ctrl_F
        dv_control_pipe = ctrl_P

        image_data = None
        source_type = None
        data_instance = None

        # 定时调度任务，保证界面现实满足设定帧率
        time_scheduler = BackgroundScheduler()
        time_scheduler.add_job(display_img_data, 'interval', seconds=1 / const.DISPLAY_FRAME_RATE)
        time_scheduler.start()

        data_flow_function_point = {const.DATA_FLOW_IDLE: data_flow_idle,
                                    const.DATA_FLOW_CHANGE: data_flow_change,
                                    const.DATA_FLOW_INIT: data_flow_init,
                                    const.DATA_FLOW_RUN: data_flow_run,
                                    const.DATA_FLOW_RELEASE: data_flow_release}
        while True:
            data_flow_function_point.get(source_control_flag.value)()
