import time

from pydarknet import Detector, Image
import cv2
import re
import numpy as np
from camera.camera import Camera

'''
参数整整
'''
enermy: int = 0  # 0:red, 1:blue
cam: int = 0  # 0:two input videos, 1:one camera plugin, 2:two cameras plugin
f_show: bool = True  # True: frame1, False: frame2
loc = {"base_b": [], "base_r": [], "watcher-b": [], "watcher-r": []}

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


def match_box(candidate, matcher, con=0.7, all=False):
    """
    searching for matching box using vetorizing

    :param candidate: in form of [[[center_x+w/2 ... ], [center_y+h/2 ...]][[x-w/2 ...][y-h/2 ...]]] list of two array
    :param matcher: the rectangle which is ready to match in the form of (center_x,center_y,w,h)

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
    nu = (nu / (matcher[2] * matcher[3])).squeeze()
    if all:
        index = np.argwhere(nu > con).squeeze(axis=1)
        return index, nu
    else:
        index = np.argmax(nu)
        return index, nu


def classify(results):
    """
    match armor with car and classify enermy and friend
    :param results: raw output of Detector
    :return:

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
            car[idx][4] = str(cat.decode("utf-8"))
            continue
        if 'watcher' in str(cat.decode("utf-8")):
            continue
        if 'base' in str(cat.decode("utf-8")):
            pass
    del car_pos_mi
    del car_pos_ma
    return car


if __name__ == "__main__":
    net = Detector(bytes("cfg/yolov3-voc.cfg", encoding="utf-8"),
                   bytes("weights/yolov3-voc_10000.weights", encoding="utf-8"), 0,
                   bytes("cfg/voc.data", encoding="utf-8"))

    if cam == 0 or cam == 2:
        if cam == 0:
            cap1 = cv2.VideoCapture("testdata/test.mov")
            cap2 = cv2.VideoCapture("testdata/test.MOV")
        else:
            cap1 = Camera()
            cap2 = Camera()  # how to distinguish two cameras hasn't been tested!!!!!!!!!
        r1, frame1 = cap1.read()
        r2, frame2 = cap2.read()
        # intialize loc
        cache, size1, size2 = init(frame1, frame2)  # assert(size1==frame1.shape)
        cv2.namedWindow('show', cv2.WINDOW_NORMAL)

        while r1 and r2:
            tic = time.time()
            r1, frame1 = cap1.read()
            r2, frame2 = cap2.read()
            dark_frame = Image(frame1)
            results1 = net.detect(dark_frame)
            dark_frame = Image(frame2)
            results2 = net.detect(dark_frame)
            del dark_frame
            toc = time.time()
            if battle_mode:
                pass
            else:
                fps = 1 / (toc - tic)
                print("FPS:", fps)

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
            cars = classify(results1)
            for car in cars:
                if enermy == 0 and 'blue' in car[4] or enermy == 1 and 'red' in car[4]:
                    continue
                x, y, w, h, cat = car
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
                    cv2.putText(frame1, "car with " + cat, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                                1, (255, 255, 0))
            cars = classify(results2)
            for car in cars:
                if enermy == 0 and 'blue' in car[4] or enermy == 1 and 'red' in car[4]:
                    continue
                x, y, w, h, cat = car

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
                    cv2.putText(frame2, "car with " + cat, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                                1, (255, 255, 0))
            if f_show:
                cv2.imshow('show', frame1)
            else:
                cv2.imshow('show', frame2)
            k = cv2.waitKey(1)
            if k == 0xFF & ord("q"):
                break
            elif k == 0xFF & ord("a"):
                f_show = not f_show
                print(f_show, not f_show)
            elif k == 0xFF & ord("p"):
                cv2.waitKey(0)
        cap1.release()
        cap2.release()

    elif cam == 1:
        cap = cv2.VideoCapture(0)
    else:
        cam = int(input("Incorrect num of camera! Please try again:"))

    print("=" * 30)
    print("Finished.")
