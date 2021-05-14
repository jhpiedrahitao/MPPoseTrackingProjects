import cv2
import mediapipe as mp 
import time

mpPose=mp.solutions.pose
pose=mpPose.Pose()
mpDraw=mp.solutions.drawing_utils

cap=cv2.VideoCapture("PoseVideos/5.mp4")
pTime=0
while True:
    success,img=cap.read()  
    img=cv2.resize(img,(img.shape[1]//3,img.shape[0]//3))
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,landmark in enumerate(results.pose_landmarks.landmark):
            h,w,c=img.shape
            cx,cy=int(landmark.x*w),int(landmark.y*h)
            cv2.circle(img,(cx,cy),5,(211,46,134), cv2.FILLED)


    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv2.putText(img,"fps: "+str(fps),(240,40),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
    cv2.imshow("Image",img)
    if (cv2.waitKey(1)  & 0xFF == ord('q')):
        break