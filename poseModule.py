import cv2
import mediapipe as mp 
import time
import math

class poseDetector():
    def __init__(self,mode=False,upper_body=False,smooth=True, detectionCon=0.5, trackingConf=0.5):
        self.mode=mode
        self.upper_body=upper_body
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackingConf=trackingConf
        
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose(self.mode,self.upper_body,self.smooth,self.detectionCon,self.trackingConf)
        

    def findPose(self, img, draw=True):
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img
 
    def findPosition(self, img, draw=True):
        self.lmList=[]
        if self.results.pose_landmarks:
            for id,landmark in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                cx,cy=int(landmark.x*w),int(landmark.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),5,(211,46,134), cv2.FILLED)
        return self.lmList

    def findAngle(self,img,p1,p2,p3,draw=True):
        #get landmarks
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        x3,y3=self.lmList[p3][1:]
        #calculate the angle
        angle=math.atan2((y3-y2),(x3-x2))-math.atan2((y1-y2),(x1-x2))
        angle=math.degrees(angle)
        if angle < 0:
            angle+=360
        # draw
        if draw:
            cv2.line(img, (x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img, (x2,y2),(x3,y3),(255,255,255),3)
            cv2.circle(img, (x1,y1),10, (25,0,255),cv2.FILLED)
            cv2.circle(img, (x1,y1),15, (25,0,255),2)
            cv2.circle(img, (x2,y2),10, (100,0,255),cv2.FILLED)
            cv2.circle(img, (x2,y2),15, (25,0,255),2)
            cv2.circle(img, (x3,y3),10, (25,0,255),cv2.FILLED)
            cv2.circle(img, (x3,y3),15, (25,0,255),2)
            cv2.putText(img,str(int(angle)),(x2+20,y2+50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        return angle

def main():
    cap=cv2.VideoCapture("PoseVideos/5.mp4")
    pTime=0
    detector=poseDetector()
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

if __name__ == "__main__":
    main()