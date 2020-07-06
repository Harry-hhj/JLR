import time

from pydarknet import Detector, Image
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/radar/Desktop/Radar/testVideo.mp4")

    average_time = 0

    net = Detector(bytes("cfg/yolov3-voc.cfg", encoding="utf-8"), bytes("weights/.weights", encoding="utf-8"), 0,
                   bytes("cfg/voc.data", encoding="utf-8"))

    while True:
        r, frame = cap.read()
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            average_time = average_time * 0.8 + (end_time - start_time) * 0.2
            # Frames per second can be calculated as 1 frame divided by time required to process 1 frame
            fps = 1 / (end_time - start_time)

            print("FPS: ", fps)
            print("Total Time:", end_time - start_time, ":", average_time)

            for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0))
                cv2.putText(frame, str(cat.decode("utf-8")), (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (255, 255, 0))

            cv2.imshow("preview", frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break

    cap.release()
