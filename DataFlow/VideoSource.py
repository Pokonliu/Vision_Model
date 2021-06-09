# Third-party library
import cv2.cv2 as cv2

# User-defined library
from Common.const import const


class Video:
    def __init__(self, video_path, d2v_queue, d2p_queue, source_control_flag, dv_control_pipe):
        # 当前source是视频流路径，初始化视频流对象
        self.video_src = cv2.VideoCapture(video_path)
        # 视频参数初始化
        self.play_flag = False
        self.schedule_trigger = False
        self.frame_single_step = const.VIDEO_FRAME_SINGLE_STEP

        self.total_frame_num = int(self.video_src.get(7))
        self.cur_frame_num = 0
        self.d2v_queue = d2v_queue
        self.d2p_queue = d2p_queue
        self.source_control_flag = source_control_flag
        self.dv_control_pipe = dv_control_pipe
        self.video_source_init()

    def video_source_init(self):
        # 主窗口的界面显示
        self.dv_control_pipe.send(["D2V", const.CONTROL_BAR_VISIBLE, True, None])
        # 进度条初始化设置
        self.dv_control_pipe.send(["D2V", const.SLIDER_INIT, self.total_frame_num, None])
        # 播放按键初始化
        self.dv_control_pipe.send(["D2V", const.PLAY_BUTTON_TRIGGERED, True, None])
        # input label显示第一帧图像
        try:
            ret, frame = self.video_src.read()
            if not ret:
                raise Exception("读取第一帧失败")
            else:
                self.d2v_queue.put(frame)
                self.d2p_queue.put(frame)
        except Exception as error:
            print("读取视频流出现错误：{}".format(error))
        # 当前已经读取一张图片到流中
        self.cur_frame_num += 1
        # 进度条初始化设置
        self.dv_control_pipe.send(["D2V", const.SLIDER_VALUE, 1, None])
        # 帧数label初始化设置
        self.dv_control_pipe.send(["D2V", const.TOTAL_FRAME, str(self.total_frame_num), None])
        self.dv_control_pipe.send(["D2V", const.CURRENT_FRAME, str(self.cur_frame_num), None])

    def run(self):
        ret = False
        frame = None
        if self.dv_control_pipe.poll():
            direction, info_type, data, detail = self.dv_control_pipe.recv()
            if info_type == const.PLAY_STATUS:
                self.play_flag = True if data else False
            elif info_type == const.FASTER_TRIGGER:
                self.cur_frame_num += self.frame_single_step
                self.cur_frame_num = min(self.cur_frame_num, self.total_frame_num)
            elif info_type == const.SLOWER_TRIGGER:
                self.cur_frame_num -= self.frame_single_step
                self.cur_frame_num = max(self.cur_frame_num, 0)
            elif info_type == const.SLIDER_TRIGGER:
                self.cur_frame_num = data
            self.video_src.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_num)
        if self.play_flag and self.schedule_trigger:
            ret, frame = self.video_src.read()
            if ret:
                self.schedule_trigger = False
                self.cur_frame_num += 1
                self.dv_control_pipe.send(["D2V", const.CURRENT_FRAME, str(self.cur_frame_num), None])
                self.dv_control_pipe.send(["D2V", const.SLIDER_VALUE, self.cur_frame_num, None])
                self.d2v_queue.put(frame)
            else:
                self.source_control_flag.value = const.DATA_FLOW_RELEASE
        return ret, frame

    '''source release function'''
    def release(self):
        # 释放Video实例
        self.video_src.release()
        self.video_src = None
        # 隐藏Control Bar
        self.dv_control_pipe.send(["D2V", const.CONTROL_BAR_VISIBLE, False, None])
