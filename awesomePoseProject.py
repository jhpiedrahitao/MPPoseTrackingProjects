import cv2
import poseModule as pm
import time
import math

cap=cv2.VideoCapture("PoseVideos/7.mp4")
pTime=0
detector=pm.poseDetector()
while True:
    success,img=cap.read()    
    img=cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
    img=detector.findPose(img)
    lmList=detector.findPosition(img)
    if len(lmList)!=0:
        print(lmList)
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"fps: "+str(fps),(240,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    cv2.imshow("Image",img)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break  