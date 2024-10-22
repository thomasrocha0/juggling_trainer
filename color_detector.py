import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import argparse

#red masking bounds
lower1=np.array([0, 1750, 100])
upper1=np.array([15,255,255])
lower2=np.array([160, 175, 100])
upper2=np.array([180, 255, 255])

#video feed
cap = cv2.VideoCapture(r"C:/Users/thoma/OneDrive/Desktop/CV_project/myvenv/Lib/site-packages/Programs/test.mp4")
kernel = np.ones((3,3), np.uint8)
while True:
    ret, frame = cap.read()

    if not ret:
        break

    #get the dimensions of the frame
    h, w, c = frame.shape

    #isolate region of interest

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #isolate red pixels
    lower = cv2.inRange(hsvImage, lower1, upper1)
    upper = cv2.inRange(hsvImage, lower2, upper2)
    fullMask = lower + upper

    #remove noise
    eroded = cv2.erode(fullMask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    #convert Matlike to Image
    mask_ = Image.fromarray(dilated)

    bbox = mask_.getbbox()

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    #cv2.imshow("fullMask", fullMask)
    cv2.imshow('fullMask', fullMask)
    cv2.imshow('dilated', dilated)
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()