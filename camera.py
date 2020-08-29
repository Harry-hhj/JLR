import numpy as np
from _sdk import mvsdk
import pyzed.sl as sl

ZED_camera_matrix1 = np.array([[1114.1804893712708, 0.0, 1074.2415297217708],
                              [0.0, 1113.4568392254073, 608.6477877664104],
                              [0.0, 0.0, 1.0]])


class HT_Camera:
    def __init__(self):
        # 枚举相机
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        i = 0 if nDev == 1 else int(input("Select camera: "))
        DevInfo = DevList[i]
        print(DevInfo)

        # 打开相机
        self.hCamera = 0
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(self.hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, 6 * 1000)
        mvsdk.CameraSetGain(self.hCamera, 128, 128, 128)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    def read(self):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape(
                (FrameHead.iHeight, FrameHead.iWidth,
                 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            return True, frame
        except mvsdk.CameraException as e:
            return False, None

    def release(self):
        # 关闭相机
        mvsdk.CameraUnInit(self.hCamera)

        # 释放帧缓存
        mvsdk.CameraAlignFree(self.pFrameBuffer)


class ZED_Camera:
    def __init__(self, serial_number=0, resolution='2K', depth_minimum=-1, depth_max=30000, accuracy='milimeter', record=False,
                 save_path='myVideoFile'):
        """

        """
        # --- Initialize a Camera object and open the ZED
        # Create a ZED camera object
        self.zed = sl.Camera()
        # Set configuration parameters
        self.init_params = sl.InitParameters()
        # self.init_params.camera_linux_id = cam_id  # TODO
        self.init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        self.init_params.set_from_serial_number(serial_number)
        if resolution == '2K':
            self.init_params.camera_resolution = sl.RESOLUTION.HD2K
        elif resolution == '720P':
            self.init_params.camera_resolution = sl.RESOLUTION.HD720
        else:
            self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.depth_maximum_distance = depth_max
        self.init_params.depth_minimum_distance = depth_minimum
        if accuracy == 'meter':
            self.init_params.coordinate_units = sl.UNIT.METER
        elif accuracy == 'millimeter':
            self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        else:
            self.init_params.coordinate_units = sl.UNIT.CENTIMETER  # Use centimeter units (for depth measurements)

        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            RuntimeError('ZED_Camera_{} failed to open.'.format(cam_id))

        # TODO: jiemian
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 8)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, 3)
        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 1)
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
        # self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 0)

        self.serial = self.zed.get_camera_information().serial_number

        self.runtime_param = sl.RuntimeParameters()
        self.runtime_param.sensing_mode = sl.SENSING_MODE.FILL

        if record:
            record_params = sl.RecordingParameters("{}.svo".format(save_path), sl.SVO_COMPRESSION_MODE.H265)
            err = self.zed.enable_recording(record_params)
            if err != sl.ERROR_CODE.SUCCESS:
                print(repr(err))
                exit(-1)

    def set_param(self, var, num):
        if var == 'EXPOSURE':
            num = num if num <= 8 else 8
            num = num if num >= 0 else 0
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, num)
        elif var == 'CONTRAST':
            num = num if num <= 8 else 8
            num = num if num >= 0 else 0
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, num)
        elif var == 'GAIN':
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, num)
        elif var == 'WHITEBALANCE_AUTO':
            num: bool = num
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, num)

    def set_runtime_param(self, mode='standard'):
        if mode == 'fill':
            self.runtime_param.sensing_mode = sl.SENSING_MODE.FILL
        else:
            self.runtime_param.sensing_mode = sl.SENSING_MODE.STANDARD

    def read(self, return_depth=True):
        image = sl.Mat()
        depth = sl.Mat()
        if self.zed.grab(
                self.runtime_param) == sl.ERROR_CODE.SUCCESS:  # A new image is available if grab() returns SUCCESS
            # Display a pixel color
            self.zed.retrieve_image(image, sl.VIEW.LEFT)  # Get the left image
            # Display a pixel depth
            self.zed.retrieve_measure(depth, sl.MEASURE.DEPTH)  # Get the depth map
            depth_map = depth.get_data()
            frame = image.get_data()
            frame = frame[:, :, 0:3]
            if return_depth:
                return True, frame, depth_map.T
            else:
                return True, frame
        else:
            if return_depth:
                return False, None, None
            else:
                return False, None

    def pause_recording(self, pause: bool = True):
        self.zed.pause_recording(pause)

    def get_shape(self):
        resolution = self.zed.get_camera_information().camera_resolution
        shape = (resolution.width, resolution.height, 3)
        return shape

    def reboot(self):
        self.zed.reboot(self.serial)

    def release(self):
        self.zed.close()

    def __del__(self):
        self.release()


if __name__ == '__main__':
    '''
    # test for HT
    cap = HT_Camera()
    while True:
        _, frame = cap.read()
        while frame is None:
            _, frame = cap.read()
        cv2.imshow('1', frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    '''
    import cv2

    cap = ZED_Camera(record=True)
    while True:
        _, frame, _ = cap.read()
        cv2.imshow('1', frame)
        if cv2.waitKey(10) & 0xff == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
