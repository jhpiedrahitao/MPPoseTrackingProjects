import cv2
import numpy as np 
import poseModule as pm
import time

cap=cv2.VideoCapture("personalAITrainer/2.mp4")
pTime=0
detector=pm.poseDetector()
count=0
dir=0 # 0 -> going up

while True:
    success,img=cap.read()    
    img=cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
    img=detector.findPose(img,False)
    lmList=detector.findPosition(img,False)
    if len(lmList)!=0:
        #right
        #detector.findAngle(img,12,14,16)
        #left
        angle=detector.findAngle(img,11,13,15)
        rangeArm=(210,320)
        per=int(np.interp(angle,rangeArm,(0,100)))
        bar=int(np.interp(angle,rangeArm,(500,200)))
        #check for the dumbbel culs
        tic=4
        if per==100:
            tic=30
            if dir==0:
                count+=0.5
                dir=1
        if per==0:
            if dir==1:
                count+=0.5
                dir=0
        cv2.rectangle(img,(53,200),(90,500),(255,255,255),tic)
        cv2.rectangle(img,(53,bar),(90,500),(0,0,255),cv2.FILLED)
        cv2.putText(img,str(per)+"%",(45,170),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        cv2.circle(img,(70,80),45,(0,0,255),cv2.FILLED)
        cv2.circle(img,(70,80),47,(255,255,255),2)
        cv2.putText(img, f'{int(count)}',(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),4)
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"fps: "+str(fps),(240,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    cv2.imshow("Image",img)
    if (cv2.waitKey(10)  & 0xFF == ord('q')):
        break  