import cv2
import numpy as np
from PIL import Image
import argparse

def update_positions(array, values):
    for x in values:
        # determine which ball (row) has the closest position to the given location
        min_index = min_distance_index(array, x)

        #make room for the new position
        array[min_index] = np.roll(array[min_index], 1)

        #update the position of the ball in the array
        array[min_index,0] = x
    return array

#returns the index of the row who's 0-indexed element is the closest to the given location
def min_distance_index(array, location):
    min_dist = np.inf
    min_index = 0
    print(array)
    for i in range(len(array)):
        dist = np.linalg.norm(np.array(array[i, 0]) - np.array(location))
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return min_index



#red masking bounds
lower1=np.array([0, 1750, 100])
upper1=np.array([15,255,255])
lower2=np.array([160, 175, 100])
upper2=np.array([180, 255, 255])

#video feed
cap = cv2.VideoCapture('test.mp4')
kernel = np.ones((3,3), np.uint8)



ball_count = int(input("Number of balls: "))

#2D array of tuples - 1st dimension is which ball is being tracked - second dimension is the 10 most recent positions of the ball
# data is tuples containinng the xy coordinates of the middle of the bounding box of that ball at that time
ball_positions_ot = np.empty((ball_count, 10), dtype=tuple)
for i in range(ball_count):
    for j in range(10):
        ball_positions_ot[i, j] = ()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    #get the dimensions of the frame
    #h, w, c = frame.shape

    #isolate region of interest

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #isolate red pixels
    lower = cv2.inRange(hsvImage, lower1, upper1)
    upper = cv2.inRange(hsvImage, lower2, upper2)
    fullMask = lower + upper

    #remove noise
    eroded = cv2.erode(fullMask, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    #code below creates a bounding box around all balls in an image
    #convert Matlike to Image
    #mask_ = Image.fromarray(dilated)
    #bbox = mask_.getbbox()

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #store the ball positions for this frame
    cur_ball_positions = np.zeros(ball_count, dtype=tuple)
    curBall = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 20:
            x, y, w, h = cv2.boundingRect(cnt)
            cur_ball_positions[curBall] = (x + w // 2, y + h // 2)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
            curBall += 1
            if curBall == ball_count: break

    ball_positions_ot = update_positions(ball_positions_ot, cur_ball_positions) 
    #cv2.imshow("fullMask", fullMask)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    cv2.imshow('frame', frame)
    print(ball_positions_ot)

    
cap.release()
cv2.destroyAllWindows()


