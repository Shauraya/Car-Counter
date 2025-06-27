import numpy as np
from ultralytics import YOLO
import cv2 as cv
import math
import cvzone
from sort import *

#for videos
cap = cv.VideoCapture("cars.mp4")



model =  YOLO("../Yolo Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv.imread("mask.png")


#tracking

tracker = Sort(max_age=20)

#400 = distance from left corner, 1250 is length in x and same with y
limits = [400,500,1250,500]

totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img, mask)
    #results = model(img, stream=True)
    results = model(img, stream=True) #-for sliced masked video

    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding Box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            #cv.rectangle(img,(x1,y1),(x2,y2), (255,0,255), 3)
            w,h = x2 - x1,y2 - y1
            #confidence
            conf = math.ceil(box.conf[0]*100)/100
            #print(conf)

            #class name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            #For detetcing cars and bikes of only a specific corner or portion of the video, go to
            # canva and create a mask using black rectangles around the area that is needed and save that mask,
            # make sure the canva frame is the same size as the video

            #to detetct only cars  and bikes
            if currentClass == "car" or currentClass == "motorbike" and conf > 0.3:

                #cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0, x1),max(40, y1)))

                #cvzone.cornerRect(img, (x1, y1, w, h), l=9)

                #for tracking id
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),3)

    for result in resultsTracker:
        x1,y1,x2,y2,ID = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        #print(result)
        w, h = x2 - x1,y2 - y1
        cvzone.cornerRect(img,(x1,y1,w,h), l=9, rt=2, colorR=(255,0,0))
        cvzone.putTextRect(img,f'{int(ID)}',(max(0, x1),max(40, y1)),scale = 2, thickness = 3, offset =10)

        cx, cy = x1+w//2, y1+h//2
        cv.circle(img,(cx,cy),5,(255,0,255),cv.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[1]+20:

            if totalCount.count(ID) == 0:
                totalCount.append(ID)
                # cv.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

    cvzone.putTextRect(img,f'Count : {len(totalCount)}',(50,50))

    cv.imshow('Video', img)
    #cv.imshow('Video Region', imgRegion)
    key = cv.waitKey(1)

    if key == ord('q'):
        break
