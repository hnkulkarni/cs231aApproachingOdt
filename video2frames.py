import numpy as np
import cv2
import os

foldername = "/media/hnkulkarni/SSD-1TB/datasets/approach_footpath"
cap = cv2.VideoCapture(os.path.join(foldername, 'IMG_8502.MOV'))

ctr = 0
while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imwrite_path = os.path.join(foldername,  f"frames/{ctr}.png")
    if ctr % 5 == 0:
        cv2.imwrite( imwrite_path, frame)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ctr += 1



cap.release()
cv2.destroyAllWindows()