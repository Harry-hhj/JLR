import cv2

cap = cv2.VideoCapture("testdata/feibiao.MOV")
ret, current_frame = cap.read()
previous_frame = current_frame

while (cap.isOpened()):
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)
    _, frame_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_diff = cv2.erode(frame_diff, kernel)
    frame_diff = cv2.dilate(frame_diff, kernel)
    cv2.imshow('fgmask', current_frame)

    cv2.imshow('frame diff ', frame_diff)

    # cv2.moveWindow(cap, 40,30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    previous_frame = current_frame.copy()
    ret, current_frame = cap.read()