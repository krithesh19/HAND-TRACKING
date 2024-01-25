import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw1 = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    
    success1, img1 = cap.read()
    imgRGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    results1 = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)


    if results1.multi_hand_landmarks:
        for handLms in results1.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img1.shape
                # h = h*2
                # w = w * 2
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                if id == 0:
                    cv2.circle(img1, (cx,cy), 5, (255,0,255), cv2.FILLED)


            mpDraw1.draw_landmarks(img1, handLms, mpHands.HAND_CONNECTIONS)



    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime


    cv2.putText(img1,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,255),3)

    cv2.imshow("Image", img1)
    cv2.waitKey(1)

