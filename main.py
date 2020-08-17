import sys

from pydarknet import Detector, Image
import cv2
import re
import numpy as np
import dlib
from PyQt5 import QtWidgets
from camera.camera import HT_Camera
import threading

from mainEntry import mywindow

import serial
import _thread
import time
from queue import Queue
import pexpect

from chuankou import offical_Judge_Handler, Game_data_define

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
|                       参数调整区域                            |
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
enermy: int = 0  # 0:red, 1:blue
cam: int = 0  # 0:two input videos, 1:one camera plugin, 2:two cameras plugin
third_cam = "antimissile"  # "":no extra cam, "antimissile":反导, "lobshot":吊射
third_cam_type = ""
f_show: int = 0  # 0: frame1, 1: frame2, 2: extra_frame
loc = {"base_b": [], "base_r": [], "watcher-b": [], "watcher-r": []}
communication_queue = Queue()

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
                car[idx][4] = s
            elif "blue" in s:
                s = "blue"
                car[idx][4] = s
            elif "grey" in s:
                s = "grey"
                if car[idx][4] == "":
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


def missile_detection(cap, size, missile_launcher, myshow):
    while True:
        _, current_frame = cap.read()
        if current_frame is None:
            break
        current_frame = cv2.resize(current_frame, size)
        current_frame_copy = current_frame.copy()
        # TODO: color threshold liangdu
        # TODO: genzong
        if 1:
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
            cv2.rectangle(current_frame_copy, (int(missile_launcher[0]), int(missile_launcher[1])),
                          (int(missile_launcher[0] + missile_launcher[2]),
                           int(missile_launcher[1] + missile_launcher[3])), (0, 255, 0), 2)
            for c in contours:
                if 100 < cv2.contourArea(c) < 40000:
                    x, y, w, h = cv2.boundingRect(c)
                    # cv2.rectangle(current_frame_copy, (x, y), (x + w, y + h), (0, 0, 255))
                    if int(missile_launcher[0]) < x + w / 2 < int(
                            missile_launcher[0] + missile_launcher[2]) and int(
                        missile_launcher[1]) < y + h / 2 < int(missile_launcher[1] + missile_launcher[3]):
                        cv2.putText(current_frame_copy, "detected missile!!", (25, 25),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    1, (255, 255, 0))
                        cv2.rectangle(current_frame_copy, (x, y), (x + w, y + h), (0, 0, 255))
                        myshow.set_text("alarm_location", "Missile detected!")
        if not battle_mode:
            cv2.imshow('fgmask', current_frame_copy)
            cv2.imshow('frame diff ', frame_diff)
        if f_show != 2:
            myshow.set_image(current_frame_copy, "sub_demo2")
        else:
            if not battle_mode:
                cv2.imshow('show', current_frame_copy)
            myshow.set_image(current_frame_copy, "main_demo")

    '''
    串口数据传输
    '''
buffercnt = 0
buffer = [int(1).to_bytes(1, 'big')]
buffer *= 1000
cmdID = 0
indecode = 0

def read():
    global buffercnt
    buffercnt = 0
    global buffer
    global cmdID
    global indecode

    while True:
        s = ser.read(1)
        s = int().from_bytes(s, 'big')
        #doc.write('s: '+str(s)+'        ')

        if buffercnt > 50:
            buffercnt = 0

        #print(buffercnt)
        buffer[buffercnt] = s
        #doc.write('buffercnt: '+str(buffercnt)+'        ')
        #doc.write('buffer: '+str(buffer[buffercnt])+'\n')
        #print(hex(buffer[buffercnt]))

        if buffercnt == 0:
            if buffer[buffercnt] != 0xa5:
                buffercnt = 0
                continue

        if buffercnt == 5:
            if offical_Judge_Handler.myVerify_CRC8_Check_Sum(id(buffer), 5) == 0:
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 7:
            cmdID = (0x0000 | buffer[5]) | (buffer[6] << 8)
            #print("cmdID")
            #print(cmdID)

        if buffercnt == 10 and cmdID == 0x0002:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Referee_Game_Result()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 12 and cmdID == 0x0001:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 12):
                Referee_Update_GameData()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 41 and cmdID == 0x0003:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 41):
                Referee_Robot_HP()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 12 and cmdID == 0x0004:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 12):
                Referee_dart_status()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt  == 13 and cmdID == 0x0101:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Referee_event_data()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 13 and cmdID == 0x0102:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Refree_supply_projectile_action()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 11 and cmdID == 0x0104:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 11):
                Refree_Warning()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 10 and cmdID == 0x0105:
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Refree_dart_remaining_time()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue

        if buffercnt == 17 and cmdID == 0x301: #2bite数据
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 17):
                Receive_Robot_Data()
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        if buffercnt == 25 and cmdID == 0x202: # 雷达没有 工程实验屏蔽掉
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 25):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        if buffercnt == 25 and cmdID == 0x203: # 雷达没有 工程实验屏蔽掉
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 25):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        if buffercnt == 27 and cmdID == 0x201: # 雷达没有 工程实验屏蔽掉
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 27):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        if buffercnt == 10 and cmdID == 0x204: # 雷达没有 工程实验屏蔽掉
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        if buffercnt == 10 and cmdID == 0x206: # 雷达没有 工程实验屏蔽掉
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        if buffercnt == 13 and cmdID == 0x209: # 雷达没有 工程实验屏蔽掉
            if offical_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                buffercnt = 0
                if buffer[buffercnt] == 0xa5:
                    buffercnt = 1
                continue


        buffercnt += 1


Game_state = Game_data_define.game_state()
Game_result = Game_data_define.game_result()
Game_robot_HP = Game_data_define.game_robot_HP()
Game_dart_status = Game_data_define.dart_status()
Game_event_data = Game_data_define.event_data()
Game_supply_projectile_action = Game_data_define.supply_projectile_action()
Game_refree_warning = Game_data_define.refree_warning()
Game_dart_remaining_time = Game_data_define.dart_remaining_time()


def Judge_Refresh_Result():
    print("Judge_Refresh_Result")


def Referee_Game_Result():
    print("Referee_Game_Result")
    Game_result.winner = buffer[7]
    print(Game_result.winner)


def  Referee_Update_GameData():
    print("Referee_Update_GameData")
    Game_state.stage_remain_time[0] = buffer[8]
    Game_state.stage_remain_time[1] = buffer[9]
    print((0x0000 | buffer[8]) | (buffer[9] << 8))

    #doc.write('Referee_Update_GameData'+'\n')
    #doc.write('remaining time: '+str((0x0000 | buffer[8]) | (buffer[9] << 8))+'\n')


def Referee_Robot_HP():
    print("Referee_Robot_HP")
    Game_robot_HP.red_1_robot_HP = [buffer[7], buffer[8]]
    Game_robot_HP.red_2_robot_HP = [buffer[9], buffer[10]]
    Game_robot_HP.red_3_robot_HP = [buffer[11], buffer[12]]
    Game_robot_HP.red_4_robot_HP = [buffer[13], buffer[14]]
    Game_robot_HP.red_5_robot_HP = [buffer[15], buffer[16]]
    Game_robot_HP.red_7_robot_HP = [buffer[17], buffer[18]]
    Game_robot_HP.red_outpost_HP = [buffer[19], buffer[20]]
    Game_robot_HP.red_base_HP = [buffer[21], buffer[22]]
    Game_robot_HP.blue_1_robot_HP = [buffer[23], buffer[24]]
    Game_robot_HP.blue_2_robot_HP = [buffer[25], buffer[26]]
    Game_robot_HP.blue_3_robot_HP = [buffer[27], buffer[28]]
    Game_robot_HP.blue_4_robot_HP = [buffer[29], buffer[30]]
    Game_robot_HP.blue_5_robot_HP = [buffer[31], buffer[32]]
    Game_robot_HP.blue_7_robot_HP = [buffer[33], buffer[34]]
    Game_robot_HP.blue_outpost_HP = [buffer[35], buffer[36]]
    Game_robot_HP.blue_base_HP = [buffer[37], buffer[38]]
    Game_robot_HP_show = '红方:'+'\n'+\
                         '英雄:'+'(0x0000 | buffer[7]) | (buffer[8] << 8)  '+\
                         '工程:'+'(0x0000 | buffer[9]) | (buffer[10] << 8)  '+\
                         '步兵3:'+'(0x0000 | buffer[11]) | (buffer[12] << 8)  '+\
                         '步兵4:'+'(0x0000 | buffer[13]) | (buffer[14] << 8)  '+'\n'+\
                         '步兵5:'+'(0x0000 | buffer[15]) | (buffer[16] << 8)  '+\
                         '哨兵:'+'(0x0000 | buffer[17]) | (buffer[18] << 8)  ' +\
                         '前哨站:' + '(0x0000 | buffer[19]) | (buffer[20] << 8)  ' +\
                         '基地:' + '(0x0000 | buffer[21]) | (buffer[22] << 8)  ' +'\n'+\
                         '蓝方:' + '\n' +\
                         '英雄:' + '(0x0000 | buffer[23]) | (buffer[24] << 8)  ' + \
                         '工程:' + '(0x0000 | buffer[25]) | (buffer[26] << 8)  ' + \
                         '步兵3:' + '(0x0000 | buffer[27]) | (buffer[28] << 8)  ' + \
                         '步兵4:' + '(0x0000 | buffer[29]) | (buffer[30] << 8)  ' + '\n' + \
                         '步兵5:' + '(0x0000 | buffer[31]) | (buffer[32] << 8)  ' + \
                         '哨兵:' + '(0x0000 | buffer[33]) | (buffer[34] << 8)  ' + \
                         '前哨站:' + '(0x0000 | buffer[35]) | (buffer[36] << 8)  ' + \
                         '基地:' + '(0x0000 | buffer[37]) | (buffer[38] << 8)  '
        #mywindow.set_text('message', Game_robot_HP_show)

def Referee_dart_status():
    print("Referee_dart_status")
    Game_dart_status.dart_belong = buffer[7]
    Game_dart_status.stage_remaining_time = [buffer[8], buffer[9]]


def Referee_event_data():
    print("Referee_event_data")
    Game_event_data.event_type = [buffer[7], buffer[8], buffer[9], buffer[10]]

    #doc.write('Referee_event_data' + '\n')


def Refree_supply_projectile_action():
    print("Refree_supply_projectile_action")
    Game_supply_projectile_action.supply_projectile_id = buffer[7]
    Game_supply_projectile_action.supply_robot_id = buffer[8]
    Game_supply_projectile_action.supply_projectile_step = buffer[9]
    Game_supply_projectile_action.supply_projectile_num = buffer[10]


def Refree_Warning():
    print("Refree_Warning")
    Game_refree_warning.level = buffer[7]
    Game_refree_warning.foul_robot_id = buffer[8]


def Refree_dart_remaining_time():
    print("Refree_dart_remaining_time")
    Game_dart_remaining_time.time = buffer[8]

    #doc.write('Refree_dart_remaining_time' + '\n')


def Receive_Robot_Data():
    print("Receive_Robot_Data()")
    if (0x0000 | buffer[7]) | (buffer[8] << 8) == 0x0200:
        print('change')


def Referee_Transmit_UserData_model( cmdID, datalength, dataID, receiverID, data):

    buffer = [0]
    buffer = buffer*200

    buffer[0] = 0xA5#数据帧起始字节，固定值为 0xA5
    buffer[1] = (datalength+6) & 0x00ff#数据帧中 data 的长度,占两个字节
    buffer[2] = ((datalength+6) & 0xff00) >> 8
    buffer[3] = 1#包序号
    buffer[4] = offical_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff);#帧头 CRC8 校验
    buffer[5] = cmdID & 0x00ff
    buffer[6] = (cmdID & 0xff00) >> 8

    buffer[7] = dataID & 0x00ff#数据的内容 ID,占两个字节
    buffer[8] = (dataID & 0xff00) >> 8
    buffer[9] = 0x09 #发送者的 ID, 占两个字节 （雷达9） #测试时候用工程2
    buffer[10] = 0x00
    buffer[11] = receiverID & 0x00ff #接收者ID
    buffer[12] = (receiverID & 0xff00) >> 8

    for i in range(datalength):
        buffer[13+i] = data[i]

    '''
    CRC16 = offical_Judge_Handler.myGet_CRC16_Check_Sum(id(buffer), 13+datalength,  0xffff)
    buffer[13+datalength] = CRC16 & 0x00ff #0xff
    buffer[14+datalength] = (CRC16 >> 8) & 0xff
    '''
    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), 13 + datalength + 2) #等价的
    #'''


    buffer_tmp_array = [0]
    buffer_tmp_array *= 15 + datalength

    for i in range(15+datalength):
        buffer_tmp_array[i] = buffer[i]
    print(buffer_tmp_array)
    print(bytearray(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))


def Robot_Data_Transmit(data, datalength, dataID, receiverID):
    Referee_Transmit_UserData_model(0x0301, datalength, dataID, receiverID, data)


''' #装作工程机器人进行测试
class client_graph_for_test():
    def __init__(self):
        client_graph_for_test.operate_type= 3
        client_graph_for_test.graphic_type= 3
        client_graph_for_test.layer= 4
        client_graph_for_test.color= 4
        client_graph_for_test.start_angle= 9
        client_graph_for_test.end_angle= 9
        client_graph_for_test.width= 10
        client_graph_for_test.start_x= 11
        client_graph_for_test.start_y= 11
        client_graph_for_test.radius= 10
        client_graph_for_test.end_x= 11
        client_graph_for_test.end_y= 11


def draw_graphics_for_test():
    client_graph = client_graph_for_test()

    client_graph.layer = 0
    client_graph.operate_type = 2
    client_graph.graphic_type = 1
    client_graph.color = 0
    client_graph.width = 5
    client_graph.start_x = 1450
    client_graph.start_y = 380
    client_graph.radius = 0
    client_graph.end_x = 1860
    client_graph.end_y = 620
    client_graph.start_angle = 0
    client_graph.end_angle = 0

    buffer[16] = (client_graph.operate_type | (client_graph.graphic_type << 3) |
                  (client_graph.layer << 6))
    buffer[17] = ((client_graph.layer >> 2) | (client_graph.color << 2) | (client_graph.start_angle << 6))
    buffer[18] = ((client_graph.start_angle >> 2) | (client_graph.end_angle << 7))
    buffer[19] = ((client_graph.end_angle >> 1))

    buffer[20] = ((client_graph.width))
    buffer[21] = 0b11111111 & ((client_graph.width >> 8) | (client_graph.start_x << 2))
    buffer[22] = 0b11111111 & ((client_graph.start_x >> 6) | (client_graph.start_y << 5))
    buffer[23] = ((client_graph.start_y >> 3))
    buffer[24] = ((client_graph.radius))
    buffer[25] = 0b11111111 & ((client_graph.radius >> 8) | (client_graph.end_x << 2))
    buffer[26] = 0b11111111 & ((client_graph.end_x >> 6) | (client_graph.end_y << 5))
    buffer[27] = ((client_graph.end_y >> 3))


    data_test = [122, 101, 114,
                 10, 8, 0, 0,
                 5, 168, 150, 47,
                 0, 16, 157, 77]


    data = [0]
    data *= 15
    data[0] = 1
    data[1] = 0
    data[2] = 0
    for i in range(12):
        data[i+3] = buffer[i+16]
    Referee_Transmit_UserData_model(0x0301, 15, 0x0101, 0x0102, data_test)
'''

robot_location = Game_data_define.robot_location()

def Robot_Data_Transmit_test():
    x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 = robot_location.get()
    data = [x1 & 0x00ff,
            (x1 & 0xff00) >> 8,
            y1 & 0x00ff,
            (y1 & 0xff00) >> 8,
            x2 & 0x00ff,
            (x2 & 0xff00) >> 8,
            y2 & 0x00ff,
            (y2 & 0xff00) >> 8,
            x3 & 0x00ff,
            (x3 & 0xff00) >> 8,
            y3 & 0x00ff,
            (y3 & 0xff00) >> 8,
            x4 & 0x00ff,
            (x4 & 0xff00) >> 8,
            y4 & 0x00ff,
            (y4 & 0xff00) >> 8,
            x5 & 0x00ff,
            (x5 & 0xff00) >> 8,
            y5 & 0x00ff,
            (y5 & 0xff00) >> 8]

    Robot_Data_Transmit(data, 20, 0x0200, 0x0002)


q = Queue()
def write():
    while True:
        #tmp = q.get()
        #print(tmp)
        #draw_graphics_for_test()
        Robot_Data_Transmit_test()
        print('send')
        a = time.time()
        while time.time() - a <= 1:
            pass
        a = 0


def time_count():
    time0 = 0
    while True:
        print(time0)
        #doc.write('time: '+str(time0)+'\n')
        time0 = time0+1
        time.sleep(1)
        robot_location.robot_1_x = time0 * 0.02
        robot_location.robot_1_y = time0 * 0.02
        robot_location.robot_2_x = 0
        robot_location.robot_2_y = 0
        robot_location.robot_3_x = 0
        robot_location.robot_3_y = 0
        robot_location.robot_4_x = 0
        robot_location.robot_4_y = 0
        robot_location.robot_5_x = 0
        robot_location.robot_5_y = 0


if __name__ == "__main__":

    password = 'zyt19991017'
    ch = pexpect.spawn('sudo chmod 777 /dev/ttyUSB0')
    ch.sendline(password)
    print('set password ok')

    ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.2)
    if ser.is_open:
        print("open ok")
        ser.flushInput()
    else:
        ser.open()
    _thread.start_new_thread(read, ())
    _thread.start_new_thread(write, ())
    _thread.start_new_thread(time_count, ())

    app = QtWidgets.QApplication(sys.argv)
    myshow = mywindow()
    # 初始化UI界面并展示
    myshow.show()

    print("[INFO] loading model...")
    net = Detector(bytes("model/1/yolov3.cfg", encoding="utf-8"),
                   bytes("model/1/yolov3_35000.weights", encoding="utf-8"), 0,
                   bytes("model/1/coco.data", encoding="utf-8"))
    # net = Detector(bytes("model/notuse2/yolov3.cfg", encoding="utf-8"),
    #               bytes("model/notuse2/yolov3_40000.weights", encoding="utf-8"), 0,
    #               bytes("model/notuse2/voc.data", encoding="utf-8"))
    # net = Detector(bytes("model/tmp/yolov3-voc.cfg", encoding="utf-8"),
    #               bytes("model/tmp/yolov3-voc_50000.weights", encoding="utf-8"), 0,
    #               bytes("model/tmp/voc.data", encoding="utf-8"))

    # 一会要追踪多个目标
    trackers1 = []
    labels1 = []
    trackers2 = []
    labels2 = []

    if cam == 0 or cam == 2:
        if cam == 0:
            cap1 = cv2.VideoCapture("testdata/red.MOV")
            cap2 = cv2.VideoCapture("testdata/b.MOV")
            if third_cam == "antimissile":
                cap3 = cv2.VideoCapture("testdata/feibiao.MOV")
        else:
            cap1 = HT_Camera()
            cap2 = HT_Camera()  # TODO: how to distinguish two cameras hasn't been tested!!*!!!!!!!
            if third_cam == "antimissile":
                cap3 = HT_Camera()
        r1, frame1 = cap1.read()
        r2, frame2 = cap2.read()

        if third_cam == "antimissile":
            cap3 = cv2.VideoCapture("testdata/feibiao.MOV")
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

            missile = threading.Thread(target=missile_detection, args=(cap3, r3_size, missile_launcher, myshow))
            missile.start()

        # intialize loc
        cache, size1, size2 = init(frame1, frame2)  # assert(size1==frame1.shape)
        cv2.namedWindow('show', cv2.WINDOW_NORMAL)


        print("="*30)
        print("[INFO] Starting.")
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
                # print("detect")
                trackers1 = []
                labels1 = []
                trackers2 = []
                labels2 = []
                dark_frame = Image(frame1)
                results1 = net.detect(dark_frame)
                dark_frame = Image(frame2)
                results2 = net.detect(dark_frame)
                del dark_frame

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
                    # x, y, w, h, cat = car
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
                    # x, y, w, h, cat = car
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
                # print("genzong")
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
            if not battle_mode:
                fps = 1/(time.time()-t1)
                print(fps)
                # cv2.moveWindow(cap, 40,30)

            if f_show == 0:
                if not battle_mode:
                    cv2.imshow('show', frame1)
                myshow.set_image(frame1, "main_demo")
                myshow.set_image(frame2, "sub_demo1")
            elif f_show == 1:
                if not battle_mode:
                    cv2.imshow('show', frame2)
                myshow.set_image(frame2, "main_demo")
                myshow.set_image(frame1, "sub_demo1")
            elif f_show == 2:
                myshow.set_image(frame1, "sub_demo1")
                myshow.set_image(frame2, "sub_demo2")
            k = cv2.waitKey(10)
            if k == 0xFF & ord("q"):
                break
            elif k == 0xFF & ord("a"):
                set_value(0, (f_show + 1) % 3)
            elif k == 0xFF & ord("p"):
                cv2.waitKey(0)
        if cam == 0:
            cap1.release()
            cap2.release()
        else:
            del cap1, cap2  # TODO: need to be changed!!!!!!!!!!!!
    elif cam == 1:
        cap = cv2.VideoCapture(0)
    else:
        cam = int(input("Incorrect num of camera! Please try again:"))

    print("=" * 30)
    print("[INFO] Finished.")
    cv2.destroyAllWindows()
    sys.exit(app.exec_())
