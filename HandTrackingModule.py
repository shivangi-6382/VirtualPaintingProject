import cv2
import mediapipe as mp
import time
import math






class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
         self.mode=mode
         self.maxHands=maxHands
         self.detectionCon=detectionCon
         self.trackCon=trackCon

         self.mpHands = mp.solutions.hands
         self.hands = self.mpHands.Hands(static_image_mode=self.mode,
    max_num_hands=self.maxHands,
    min_detection_confidence=self.detectionCon,
    min_tracking_confidence=self.trackCon)
         self.mpDraw = mp.solutions.drawing_utils
         self.tipsIds=[4,8,12,16,20]

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img



    def findPosition(self,img,handNo=0,draw=True):

        self.lmList=[]
        if self.results.multi_hand_landmarks:
            self.mpHand=self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(self.mpHand.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id,cx,cy])
                # if id==4:
                if draw:
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
        return self.lmList



    def fingersUp(self):
        fingers = []

        # THUMB
        if self.lmList[self.tipsIds[0]][1] < self.lmList[self.tipsIds[0] - 1][1]:
            fingers.append(1)

        else:
            fingers.append(0)

        # FOR FINGERS
        for id in range(1, 5):

            if self.lmList[self.tipsIds[id]][2] < self.lmList[self.tipsIds[id] - 2][2]:
                # print("Index finger open")
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    def findDistance(self,x,y,img):
        length = math.hypot(self.lmList[x][1]- self.lmList[y][1], self.lmList[x][2] - self.lmList[y][2])
        x1, y1 = self.lmList[x][1], self.lmList[x][2]
        x2, y2 = self.lmList[y][1], self.lmList[y][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        #cx,cy = (self.lmList[x][1]-self.lmList[y][1])//2,(self.lmList[x][2]-self.lmList[y][2])//2
        cv2.line(img,(self.lmList[x][1],self.lmList[x][2]),(self.lmList[y][1],self.lmList[y][2]),(255,0,0),3)
        cv2.circle(img,(cx,cy),12,(255,0,255),cv2.FILLED)
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

        return length,img,cx,cy






def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success, img = cap.read()
        img=detector.findHands(img)

        lmList=detector.findPosition(img)
        #if len(lmList)>0:
            #print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__=="__main__":
    main()