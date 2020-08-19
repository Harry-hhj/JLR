import time
from serial_package import offical_Judge_Handler, Game_data_define

buffercnt = 0
buffer = [0]
buffer *= 1000
cmdID = 0
indecode = 0


def read(ser, myshow):
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
                Referee_Robot_HP(myshow)
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
    #print(buffer[8])
    print('time:'+str((0x0000 | buffer[8]) | (buffer[9] << 8)))

    #doc.write('Referee_Update_GameData'+'\n')
    #doc.write('remaining time: '+str((0x0000 | buffer[8]) | (buffer[9] << 8))+'\n')


def Referee_Robot_HP(myshow):
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
                         '英雄:'+str((0x0000 | buffer[7]) | (buffer[8] << 8))+'\t'+\
                         '工程:'+str((0x0000 | buffer[9]) | (buffer[10] << 8))+'\t'+\
                         '步兵3:'+str((0x0000 | buffer[11]) | (buffer[12] << 8)) +'\t'+\
                         '步兵4:'+str((0x0000 | buffer[13]) | (buffer[14] << 8)) +'\n'+\
                         '步兵5:'+str((0x0000 | buffer[15]) | (buffer[16] << 8)) +'\t'+\
                         '哨兵:'+str((0x0000 | buffer[17]) | (buffer[18] << 8))  +'\t'+\
                         '前哨站:' + str((0x0000 | buffer[19]) | (buffer[20] << 8))  +'\t'+\
                         '基地:' + str((0x0000 | buffer[21]) | (buffer[22] << 8)) +'\n'+\
                         '蓝方:' + '\n' +\
                         '英雄:' + str((0x0000 | buffer[23]) | (buffer[24] << 8)) +'\t'+ \
                         '工程:' + str((0x0000 | buffer[25]) | (buffer[26] << 8)) + '\t'+\
                         '步兵3:' + str((0x0000 | buffer[27]) | (buffer[28] << 8)) + '\t'+\
                         '步兵4:' + str((0x0000 | buffer[29]) | (buffer[30] << 8)) + '\n' + \
                         '步兵5:' + str((0x0000 | buffer[31]) | (buffer[32] << 8)) + '\t'+\
                         '哨兵:' + str((0x0000 | buffer[33]) | (buffer[34] << 8)) + '\t'+\
                         '前哨站:' + str((0x0000 | buffer[35]) | (buffer[36] << 8)) + '\t'+\
                         '基地:' + str((0x0000 | buffer[37]) | (buffer[38] << 8))
    
    print(Game_robot_HP_show)
    myshow.set_text('message_box', Game_robot_HP_show)


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


def Referee_Transmit_UserData_model( cmdID, datalength, dataID, receiverID, data, ser):

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
    offical_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), 13 + datalength + 2)  # 等价的

    buffer_tmp_array = [0]
    buffer_tmp_array *= 15 + datalength

    for i in range(15+datalength):
        buffer_tmp_array[i] = buffer[i]
    #print(buffer_tmp_array)
    #print(bytearray(buffer_tmp_array))
    ser.write(bytearray(buffer_tmp_array))


def Robot_Data_Transmit(data, datalength, dataID, receiverID, ser):
    Referee_Transmit_UserData_model(0x0301, datalength, dataID, receiverID, data, ser)


def Robot_Data_Transmit_test(robot_loc, ser):
    x1,y1,x2,y2,x3,y3,x4,y4,x5,y5 = robot_loc.get()
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

    Robot_Data_Transmit(data, 20, 0x0200, 0x0002, ser)


def write(robot_loc, ser):
    while True:
        Robot_Data_Transmit_test(robot_loc, ser)
        #print('send')
        time.sleep(0.2)
