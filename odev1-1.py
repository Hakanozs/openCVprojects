import time
import os
import mediapipe as mp
import cv2

 
 

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands= mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tipIds = [3,4,11,12]
img_counter = 0
while True:
    success, img = cap.read()
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sonuc= hands.process(imgrgb)

    
    if sonuc.multi_hand_landmarks:
        for handLms in sonuc.multi_hand_landmarks:
            lmList = []
            for id , lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                lmList.append([id, cx, cy])
            if lmList[8][2]<=lmList[7][2] and lmList[4][2]<= lmList[3][2] and lmList[12][2]<= lmList[11][2] and lmList[16][2]<= lmList[15][2] and lmList[20][2]<= lmList[19][2]:
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, img)
                print("{} written!".format(img_name))
                img_counter += 1
            
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    
    cv2.imshow("Image", img)

    key =cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
