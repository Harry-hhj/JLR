import time
import sys

from pydarknet import Detector, Image
import cv2
import re
import numpy as np
import dlib
from PyQt5 import QtWidgets
from camera.camera import Camera  # incorrect!!!!!!!!!!!!!!!!!

from mainEntry import mywindow

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
|                       参数调整区域                            |
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
enermy: int = 0  # 0:red, 1:blue
cam: int = 0  # 0:two input videos, 1:one camera plugin, 2:two cameras plugin
third_cam = "antimissile"  # "":no extra cam, "antimissile":反导, "lobshot":吊射
f_show: bool = True  # True: frame1, False: frame2
loc = {"base_b": [], "base_r": [], "watcher-b": [], "watcher-r": []}
exit_signal = False

battle_mode: bool = False  # automatically set some value, ready for battle #not implement yet


def init(frame1, frame2=None):
    """
    do some procedures before start detecting

    :param frame1: input frame of any size
    :param frame2: None if only one camera source

    :return:
    cahce: numpy array for calculating
    shape: the shape of the input, used to ensure the constant size of frame during process
    """
    type_regex = r'^[0-9][0-9]?$'
    cv2.namedWindow('ROI_init', cv2.WINDOW_NORMAL)
    tmp = list(loc.keys())
    message = ""
    mi = []
    ma = []
    rec = []
    for i in range(len(tmp)):
        message = message + str(i) + ": " + str(tmp[i]) + " || "
    message = message + "c: cancel\n"
    while True:
        boxtmp = cv2.selectROI('ROI_init', frame1, False)
        if boxtmp == (0, 0, 0, 0):
            break
        type = input(message)  # keyouhua
        if type in ['c', 'C']:
            break
        if re.search(type_regex, type):
            box = ((boxtmp[0] + boxtmp[2]) / 2 / frame1.shape[0], (boxtmp[1] + boxtmp[2]) / 2 / frame1.shape[1],
                   (boxtmp[2] - boxtmp[0]) / frame1.shape[0], (boxtmp[3] - boxtmp[1]) / frame1.shape[1])
            loc[tmp[int(type)]].append(box)
            cv2.rectangle(frame1, (boxtmp[0], boxtmp[1]), (boxtmp[0] + boxtmp[2], boxtmp[1] + boxtmp[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame1, tmp[int(type)] + str(len(loc[tmp[int(type)]])), (boxtmp[0], boxtmp[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
            mi.append([int(boxtmp[0] + boxtmp[2]), int(boxtmp[1] + boxtmp[3])])
            ma.append([int(boxtmp[0]), int(boxtmp[1])])
            rec.append(tmp[int(type)] + str(len(loc[tmp[int(type)]])))
        else:
            print("Invalid input.")
    mi = np.array(mi).T
    ma = np.array(ma).T
    cache = {"mi1": mi, "ma1": ma, "rec1": rec}
    if frame2 is None:
        cv2.destroyWindow('ROI_init')
        return cache, frame1.shape  # not necessary . something needs preprocessing.
    else:
        mi = []
        ma = []
        rec = []
        while True:
            boxtmp = cv2.selectROI('ROI_init', frame2, False)
            if boxtmp == (0, 0, 0, 0):
                break
            type = input(message)
            if type in ['c', 'C']:
                break
            if re.search(type_regex, type):
                box = ((boxtmp[0] + boxtmp[2]) / 2 / frame1.shape[0], (boxtmp[1] + boxtmp[2]) / 2 / frame1.shape[1],
                       (boxtmp[2] - boxtmp[0]) / frame1.shape[0], (boxtmp[3] - boxtmp[1]) / frame1.shape[1])
                loc[tmp[int(type)]].append(box)
                cv2.rectangle(frame2, (boxtmp[0], boxtmp[1]), (boxtmp[0] + boxtmp[2], boxtmp[1] + boxtmp[3]),
                              (0, 255, 0), 2)
                cv2.putText(frame2, tmp[int(type)] + str(len(loc[tmp[int(type)]])), (boxtmp[0], boxtmp[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
                mi.append([int(boxtmp[0] + boxtmp[2]), int(boxtmp[1] + boxtmp[3])])
                ma.append([int(boxtmp[0]), int(boxtmp[1])])
                rec.append(tmp[int(type)] + str(len(loc[tmp[int(type)]])))
            else:
                print("Invalid input.")
        mi = np.array(mi).T
        ma = np.array(ma).T
        cache["mi2"] = mi
        cache["ma2"] = ma
        cache["rec2"] = rec
        cv2.destroyWindow('ROI_init')
        return cache, frame1.shape, frame2.shape


def match_box(candidate, matcher, all=False, con=0.7):
    """
    match_box(candidate, matcher[, all[, con]]) -> index(es), score
    searching for matching box using vetorizing

    :param candidate: in form of [[[center_x+w/2 ... ], [center_y+h/2 ...]][[x-w/2 ...][y-h/2 ...]]] list of two array
    :param matcher: the rectangle which is ready to match in the form of (center_x,center_y,w,h)
    :param all: select all boxes that reaches con or return the max convincing box
    :param con: concidence

    :return:
    index: index of group having concidence > con
    nu: concidence of every group in the form of array (n,)
    """
    assert (len(candidate) == 2)
    assert (isinstance(matcher, (tuple, list)))
    mi = candidate[0]
    ma = candidate[1]
    assert (isinstance(mi, np.ndarray))
    assert (isinstance(ma, np.ndarray))
    assert (mi.ndim == 2)
    assert (ma.ndim == 2)
    assert (mi.shape[0] == 2)
    assert (ma.shape[0] == 2)
    assert (len(matcher) == 4)

    mi_ = np.array([[matcher[0] + matcher[2] / 2], [matcher[1] + matcher[3] / 2]])
    ma_ = np.array([[matcher[0] - matcher[2] / 2], [matcher[1] - matcher[3] / 2]])

    mi_use = np.minimum(mi_, mi)
    ma_use = np.maximum(ma_, ma)
    nu = mi_use - ma_use
    nu[nu < 0] = 0
    nu = np.product(nu, axis=0, keepdims=True)
    score = (nu / (matcher[2] * matcher[3])).squeeze()
    if all:
        index = np.argwhere(score > con).squeeze(axis=1)
        return index, score
    else:
        index = np.argmax(nu)
        return index, score


def car_armor_classify(results):
    """
    match armor with car and classify enermy and friend
    :param results: raw output of Detector
    :return:
    car: a list of car, return [] when the number of car equals to zero
    """
    car = []
    car_pos_mi = []
    car_pos_ma = []
    for cat, score, bounds in results:
        if 'car' in str(cat.decode("utf-8")) and score > 0.5:
            car_pos_mi.append([bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2])
            car_pos_ma.append([bounds[0] - bounds[2] / 2, bounds[1] - bounds[3] / 2])
            car.append([bounds[0], bounds[1], bounds[2], bounds[3], ""])
    if len(car) == 0:
        return car
    car_pos_mi = np.array(car_pos_mi).T
    car_pos_ma = np.array(car_pos_ma).T
    # print("car_pos_mi.ndim:", car_pos_mi.ndim)
    assert (car_pos_mi.ndim == 2)
    assert (car_pos_ma.ndim == 2)
    for cat, score, bounds in results:
        if 'armor' in str(cat.decode("utf-8")):
            idx, _ = match_box([car_pos_mi, car_pos_ma], bounds)
            s = str(cat.decode("utf-8"))
            if "red" in s:
                s = "red"
            elif "blue" in s:
                s = "blue"
            elif "grey" in s:
                s = "grey"
            car[idx][4] = s
            continue
        if 'watcher' in str(cat.decode("utf-8")):
            continue
        if 'base' in str(cat.decode("utf-8")):
            pass
    del car_pos_mi
    del car_pos_ma
    return car


def check_car(frame_current, frame_previous, cars):
    """
    use frame difference to reduce misidentification based on the fact that cars are always in motion(actually it's not
    always True)

    :param frame_current: current frame
    :param frame_previous: previous frame
    :param cars: produced by car_armor_classify(...)
    :return:
    cars: after selection
    """
    assert (frame_current is not None and frame_previous is not None)
    if len(cars == 0):
        return []
    current_frame_gray = cv2.cvtColor(frame_current, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(frame_previous, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    _, frame_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_diff = cv2.erode(frame_diff, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    frame_diff = cv2.dilate(frame_diff, kernel)

    diff = frame_diff.copy()
    for i in range(len(cars)):
        car = cars[i]
        score = diff[int(car[0] - car[2] / 2):int(car[0] + car[2] / 2),
                     int(car[1] - car[3] / 2):int(car[1] + car[3] / 2)].sum()
        if score < car[2] * car[3] / 10:
            if car[4] is None:
                cars.pop(i)
    return cars


def set_value(value_index, value):
    """
    API for operator's control.
    :param value_name: global variable in main.py
    :param value: notuse2 value
    :return:
    flag: True indicates sucess while False means failure
    """
    global f_show
    if value_index == 0:
        f_show = value
        return True
    else:
        return False


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    # 初始化UI界面并展示
    myshow.show()
    print("[INFO] loading model...")
    net = Detector(bytes("model/1/yolov3.cfg", encoding="utf-8"),
                   bytes("model/1/yolov3_35000.weights", encoding="utf-8"), 0,
                   bytes("model/1/coco.data", encoding="utf-8"))
    #net = Detector(bytes("model/notuse2/yolov3.cfg", encoding="utf-8"),
    #               bytes("model/notuse2/yolov3_40000.weights", encoding="utf-8"), 0,
    #               bytes("model/notuse2/voc.data", encoding="utf-8"))
    #net = Detector(bytes("model/tmp/yolov3-voc.cfg", encoding="utf-8"),
    #               bytes("model/tmp/yolov3-voc_50000.weights", encoding="utf-8"), 0,
    #               bytes("model/tmp/voc.data", encoding="utf-8"))

    # 一会要追踪多个目标
    trackers1 = []
    labels1 = []
    trackers2 = []
    labels2 = []

    if cam == 0 or cam == 2:
        if cam == 0:
            cap1 = cv2.VideoCapture("testdata/test.mov")
            cap2 = cv2.VideoCapture("testdata/test.MOV")
            if third_cam == "antimissile":
                cap3 = cv2.VideoCapture("testdata/feibiao.MOV")
        else:
            cap1 = Camera()
            cap2 = Camera()  # how to distinguish two cameras hasn't been tested!!*!!!!!!!
            if third_cam == "antimissile":
                cap3 = Camera()
        r1, frame1 = cap1.read()
        r2, frame2 = cap2.read()
        if third_cam == "antimissile":
            r3_size = (960, 540)
            r3, current_frame = cap3.read()
            current_frame = cv2.resize(current_frame, r3_size)
            previous_frame = current_frame
            cv2.namedWindow('missile', cv2.WINDOW_NORMAL)
            f = current_frame.copy()
            missile_launcher = cv2.selectROI('missile', f, False)
            cv2.rectangle(f, (int(missile_launcher[0]), int(missile_launcher[1])),
                          (int(missile_launcher[0] + missile_launcher[2]),
                           int(missile_launcher[1] + missile_launcher[3])), (0, 255, 0), 2)
            verify = cv2.selectROI('missile', f, False)
            if verify != (0, 0, 0, 0):
                missile_launcher = verify
                del verify
            cv2.destroyWindow('missile')


        #thread = threading.Thread(target=antimissile, args=(cap3,))
        #thread.start()
        # intialize loc
        cache, size1, size2 = init(frame1, frame2)  # assert(size1==frame1.shape)
        cv2.namedWindow('show', cv2.WINDOW_NORMAL)

        tic = 0
        while True:
            t1 = time.time()
            r1, frame1 = cap1.read()
            r2, frame2 = cap2.read()
            if frame1 is None or frame2 is None:
                break
            # 预处理操作
            rgb1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            rgb2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            if time.time() - tic > 5 or len(trackers1) == 0 or len(trackers2) == 0:
                #print("detect")
                trackers1 = []
                labels1 = []
                trackers2 = []
                labels2 = []
                dark_frame = Image(frame1)
                results1 = net.detect(dark_frame)
                dark_frame = Image(frame2)
                results2 = net.detect(dark_frame)
                del dark_frame
                if battle_mode:
                    pass
                else:
                    pass

                assert (size1 == frame1.shape)
                assert (size2 == frame2.shape)
                '''
                for cat, score, bounds in results1:  # 暂时没有把报警位置画出来
                    if enermy == 0 and 'blue' in str(cat.decode("utf-8")) or \
                            enermy == 1 and 'red' in str(cat.decode("utf-8")):
                        print(str(cat.decode("utf-8")) + "skip")
                        continue
                    if score >= 0.5:
                        x, y, w, h = bounds
                        mi_ = [[int(x + w / 2)], [int(y + h / 2)]]
                        ma_ = [[int(x - w / 2)], [int(y - h / 2)]]
                        cv2.rectangle(frame1, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (255, 0, 0))
                        cv2.putText(frame1, str(cat.decode("utf-8")) + ",score: " + str(score), (int(x), int(y)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                        mi_ = np.array(mi_)
                        ma_ = np.array(ma_)
                        assert (mi_.shape == (2, 1))
                        assert (ma_.shape == (2, 1))
                        mi_use = np.minimum(mi_, cache["mi1"])
                        ma_use = np.maximum(ma_, cache["ma1"])
                        assert (mi_use.shape == (2, len(cache["rec1"])))
                        assert (ma_use.shape == (2, len(cache["rec1"])))
                        nu = mi_use - ma_use
                        nu[nu < 0] = 0
                        nu = np.product(nu, axis=0, keepdims=True)
                        nu = (nu / (w * h)).squeeze()
                        index = np.argwhere(nu > 0.7).squeeze(axis=1)
                        for i in index:
                            print("frame1 " + str(cache["rec1"][i]) + " detected enermy!")
                for cat, score, bounds in results2:
                    if enermy == 0 and 'blue' in str(cat.decode("utf-8")) or \
                            enermy == 1 and 'red' in str(cat.decode("utf-8")):
                        continue
                    if score >= 0.5:
                        x, y, w, h = bounds
                        mi_ = [[int(x + w / 2)], [int(y + h / 2)]]
                        ma_ = [[int(x - w / 2)], [int(y - h / 2)]]
                        cv2.rectangle(frame2, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (255, 0, 0))
                        cv2.putText(frame2, str(cat.decode("utf-8")) + ",score: " + str(score), (int(x), int(y)),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))
                        mi_ = np.array(mi_)
                        ma_ = np.array(ma_)
                        assert (mi_.shape == (2, 1))
                        assert (ma_.shape == (2, 1))
                        mi_use = np.minimum(mi_, cache["mi2"])
                        ma_use = np.maximum(ma_, cache["ma2"])
                        assert (mi_use.shape == (2, len(cache["rec2"])))
                        assert (ma_use.shape == (2, len(cache["rec2"])))
                        nu = mi_use - ma_use
                        nu[nu < 0] = 0
                        nu = np.product(nu, axis=0, keepdims=True)
                        nu = (nu / (w * h)).squeeze()
                        index = np.argwhere(nu > 0.7).squeeze(axis=1)
                        for i in index:
                            print("frame2 " + str(cache["rec2"][i]) + " detected enermy!")
                '''
                cars = car_armor_classify(results1)
                for car in cars:
                    x, y, w, h, cat = car
                    (startX, startY, endX, endY) = (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2))
                    # 使用dlib来进行目标追踪
                    # http://dlib.net/python/index.html#dlib.correlation_tracker
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    t.start_track(rgb1, rect)

                    # 保存结果
                    labels1.append(car[4])
                    trackers1.append(t)
                    if enermy == 0 and 'blue' in car[4] or enermy == 1 and 'red' in car[4]:
                        continue
                    #x, y, w, h, cat = car
                    mi_ = [[int(x + w / 2)], [int(y + h / 2)]]
                    ma_ = [[int(x - w / 2)], [int(y - h / 2)]]
                    mi_ = np.array(mi_)
                    ma_ = np.array(ma_)
                    assert (mi_.shape == (2, 1))
                    assert (ma_.shape == (2, 1))
                    mi_use = np.minimum(mi_, cache["mi1"])
                    ma_use = np.maximum(ma_, cache["ma1"])
                    assert (mi_use.shape == (2, len(cache["rec1"])))
                    assert (ma_use.shape == (2, len(cache["rec1"])))
                    nu = mi_use - ma_use
                    nu[nu < 0] = 0
                    nu = np.product(nu, axis=0, keepdims=True)
                    nu = (nu / (w * h)).squeeze()
                    index = np.argwhere(nu > 0.7).squeeze(axis=1)
                    for i in index:
                        print("frame1 " + str(cache["rec1"][i]) + " detected enermy!")
                    if car[4] == "":
                        cv2.rectangle(frame1, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (255, 0, 0))
                        cv2.putText(frame1, "Car of unkown type", (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                                    1, (255, 255, 0))
                    else:
                        cv2.rectangle(frame1, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (255, 0, 0))
                        cv2.putText(frame1, "Car with " + cat, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                                    1, (255, 255, 0))
                cars = car_armor_classify(results2)
                for car in cars:
                    x, y, w, h, cat = car
                    (startX, startY, endX, endY) = (int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2))
                    # 使用dlib来进行目标追踪
                    # http://dlib.net/python/index.html#dlib.correlation_tracker
                    t = dlib.correlation_tracker()
                    rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                    t.start_track(rgb2, rect)

                    # 保存结果
                    labels2.append(car[4])
                    trackers2.append(t)
                    if enermy == 0 and 'blue' in car[4] or enermy == 1 and 'red' in car[4]:
                        continue
                    #x, y, w, h, cat = car
                    mi_ = [[int(x + w / 2)], [int(y + h / 2)]]
                    ma_ = [[int(x - w / 2)], [int(y - h / 2)]]
                    mi_ = np.array(mi_)
                    ma_ = np.array(ma_)
                    assert (mi_.shape == (2, 1))
                    assert (ma_.shape == (2, 1))
                    mi_use = np.minimum(mi_, cache["mi2"])
                    ma_use = np.maximum(ma_, cache["ma2"])
                    assert (mi_use.shape == (2, len(cache["rec2"])))
                    assert (ma_use.shape == (2, len(cache["rec2"])))
                    nu = mi_use - ma_use
                    nu[nu < 0] = 0
                    nu = np.product(nu, axis=0, keepdims=True)
                    nu = (nu / (w * h)).squeeze()
                    index = np.argwhere(nu > 0.7).squeeze(axis=1)
                    for i in index:
                        print("frame2 " + str(cache["rec2"][i]) + " detected enermy!")
                    if car[4] == "":
                        cv2.rectangle(frame2, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (255, 0, 0))
                        cv2.putText(frame2, "Car of unkown type", (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                                    1, (255, 255, 0))
                    else:
                        cv2.rectangle(frame2, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                      (255, 0, 0))
                        cv2.putText(frame2, "Car with " + cat, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                                    1, (255, 255, 0))
                tic = time.time()
            else:
                #print("genzong")
                # 每一个追踪器都要进行更新
                # toc = time.time()
                for (t, l) in zip(trackers1, labels1):
                    t.update(rgb1)
                    pos = t.get_position()

                    # 得到位置
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # 画出来
                    cv2.rectangle(frame1, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    cv2.putText(frame1, l, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                for (t, l) in zip(trackers2, labels2):
                    t.update(rgb2)
                    pos = t.get_position()

                    # 得到位置
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # 画出来
                    cv2.rectangle(frame2, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    cv2.putText(frame2, l, (startX, startY - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            #print(time.time()-t1)
            if third_cam == "antimissile" and current_frame is not None:
                current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                current_frame_gray = cv2.GaussianBlur(current_frame_gray, (7, 7), 0)
                previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (7, 7), 0)

                frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
                _, frame_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                frame_diff = cv2.erode(frame_diff, kernel)
                frame_diff = cv2.dilate(frame_diff, kernel)
                contours, _ = cv2.findContours(frame_diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                current_frame_copy = current_frame.copy()
                cv2.rectangle(current_frame_copy, (int(missile_launcher[0]), int(missile_launcher[1])),
                              (int(missile_launcher[0] + missile_launcher[2]),
                               int(missile_launcher[1] + missile_launcher[3])), (0, 255, 0), 2)
                for c in contours:
                    if 100 < cv2.contourArea(c) < 40000:
                        x, y, w, h = cv2.boundingRect(c)
                        #cv2.rectangle(current_frame_copy, (x, y), (x + w, y + h), (0, 0, 255))
                        if int(missile_launcher[0]) < x+w/2 < int(missile_launcher[0]+missile_launcher[2]) and int(missile_launcher[1]) < y+h/2 < int(missile_launcher[1]+missile_launcher[3]):
                            cv2.putText(current_frame_copy, "detected missile!!!!!!!!!!!!!!!!!!!", (25, 25), cv2.FONT_HERSHEY_COMPLEX,
                                        1, (255, 255, 0))
                            cv2.rectangle(current_frame_copy, (x, y), (x + w, y + h), (0, 0, 255))
                            myshow.set_text("alarm_location", "Missile detected!")
                cv2.imshow('fgmask', current_frame_copy)

                #cv2.imshow('frame diff ', frame_diff)

                # cv2.moveWindow(cap, 40,30)

                previous_frame = current_frame.copy()
                ret, current_frame = cap3.read()
                current_frame = cv2.resize(current_frame, (960, 540))

            if f_show:
                cv2.imshow('show', frame1)
                myshow.set_image(frame1, "main_demo")
                myshow.set_image(frame2, "sub_demo1")
            else:
                cv2.imshow('show', frame2)
                myshow.set_image(frame2, "main_demo")
                myshow.set_image(frame1, "sub_demo1")
            if third_cam is not None:
                myshow.set_image(current_frame_copy, "sub_demo2")
            k = cv2.waitKey(2)
            if k == 0xFF & ord("q"):
                exit_signal = True
                break
            elif k == 0xFF & ord("a"):
                set_value(0, not f_show)
                #f_show = not f_show
            elif k == 0xFF & ord("p"):
                cv2.waitKey(0)
        if cam == 0:
            cap1.release()
            cap2.release()
        else:
            del cap1, cap2  # need to be changed!!!!!!!!!!!!
    elif cam == 1:
        cap = cv2.VideoCapture(0)
    else:
        cam = int(input("Incorrect num of camera! Please try again:"))

    print("=" * 30)
    print("[INFO] Finished.")
    cv2.destroyAllWindows()
    sys.exit(app.exec_())
