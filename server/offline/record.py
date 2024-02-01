import numpy as np
import cv2 as cv
from sys import argv
import time


stream_url='tcp://192.168.0.31:8123'

cap = cv.VideoCapture(stream_url)
if not cap.isOpened():
    print("Cannot open camera")
    exit()


t0 = time.time()
ret, frame = cap.read()
prev = frame


buffer = np.array([frame])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #live cap
    cv.imshow('frame', frame)
    
    t = time.time()

    if t - t0 >= 0.1:
        print(t-t0)
        #take computation time into account
        buffer = np.append(buffer,[frame],axis=0)
        t0 = time.time()

    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

for idx, frame in enumerate(buffer):
    cv.imwrite(f'./recordings/{argv[1]}/frame_{idx:06d}.png' , frame )

