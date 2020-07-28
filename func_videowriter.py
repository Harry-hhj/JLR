import cv2
from datetime import datetime
import time


class Writer:
    def __init__(self, cam_num):
        self.num = cam_num
        self.size = (1920, 1080)  # 保存视频分辨率的大小
        if self.num == 2:
            self.writer1 = cv2.VideoWriter('recording/{}_orig1.mov'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')), cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 45.0, self.size)
            self.writer2 = cv2.VideoWriter('recording/{}_orig2.mov'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')), cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 45.0, self.size)
            self.writer_1 = cv2.VideoWriter('recording/{}_proc1.mov'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')), cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 45.0, self.size)
            self.writer_2 = cv2.VideoWriter('recording/{}_proc2.mov'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')), cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 45.0, self.size)
        elif self.num == 1:
            self.writer1 = cv2.VideoWriter('recording/{}_orig1.mov'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')), cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 45.0, self.size)
            self.writer_1 = cv2.VideoWriter('recording/{}_porc1.mov'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M')), cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 45.0, self.size)

    def write_orig(self, frame1, frame2=None):
        if self.num == 2:
            self.writer1.write(frame1)
            self.writer2.write(frame2)
        else:
            self.writer1.write(frame1)

    def write_pros(self, frame1, frame2=None):
        if self.num == 2:
            self.writer_1.write(frame1)
            self.writer_2.write(frame2)
        else:
            self.writer_1.write(frame1)

    def clean(self):
        print("CLASS WRITER FUNC CLEAN NOT IMPLEMENTED!")
        pass

    def __del__(self):
        if self.num == 2:
            self.writer1.release()
            self.writer2.release()
            self.writer_1.release()
            self.writer_2.release()
        else:
            self.writer1.release()
            self.writer_1.release()
