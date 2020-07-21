#导入工具包
import numpy as np
import dlib
import cv2, time
from pydarknet import Detector, Image

print("[INFO] starting video stream...")
vs = cv2.VideoCapture('testdata/test.mov')
# 一会要追踪多个目标
trackers = []
labels = []

print("[INFO] loading model...")
net = Detector(bytes("model/1/yolov3.cfg", encoding="utf-8"),
                   bytes("model/1/yolov3_35000.weights", encoding="utf-8"), 0,
                   bytes("model/1/coco.data", encoding="utf-8"))

#size = (1920, 1080)  # 保存视频分辨率的大小

#videoWriter = cv2.VideoWriter('results/track_output.mov', cv2.VideoWriter_fourcc('a', 'v', 'c', '1'), 45.0, size)
tic = 0
#fps = 0

while True:
    # 读取一帧
    grabbed, frame = vs.read()

    # 是否是最后了
    if frame is None:
        break

    # 预处理操作
    (h, w) = frame.shape[:2]
    width = 600
    r = width / float(w)
    dim = (width, int(h * r))
    #frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 先检测 再追踪
    if time.time()-tic > 5 or len(trackers) == 0:
        trackers = []
        labels = []
        # 获取blob数据
        dark_frame = Image(frame)
        results = net.detect(dark_frame)
        #print(results)
        del dark_frame

        # 遍历得到的检测结果
        for cat, score, bounds in results:
            # 过滤
            if score > 0.2:
                label = cat.decode('utf-8')

                # 只保留人的
                if "car" not in label:
                    continue

                # 得到BBOX
                x,y,w,h = bounds
                (startX, startY, endX, endY) = (int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2))
                #print((startX, startY, endX, endY))

                # 使用dlib来进行目标追踪
                # http://dlib.net/python/index.html#dlib.correlation_tracker
                t = dlib.correlation_tracker()
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                t.start_track(rgb, rect)

                # 保存结果
                labels.append(label)
                trackers.append(t)

                # 绘图
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                tic = time.time()

    # 如果已经有了框，就可以直接追踪了
    else:
        # 每一个追踪器都要进行更新
        #toc = time.time()
        for (t, l) in zip(trackers, labels):
            t.update(rgb)
            pos = t.get_position()

            # 得到位置
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # 画出来
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(frame, l, (startX, startY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        #fps = 0.98*fps + 0.02*1/(time.time()-toc)

    # 显示
    cv2.imshow("Frame", frame)
    #frame = cv2.resize(frame, size)
    #videoWriter.write(frame)
    key = cv2.waitKey(1) & 0xFF

    # 退出
    if key == 27:
        break
#print(fps)
cv2.destroyAllWindows()
#videoWriter.release()
vs.release()

