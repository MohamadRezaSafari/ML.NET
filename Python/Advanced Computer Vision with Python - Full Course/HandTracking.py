import cv2
import mediapipe as mp
import time


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    imRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


# vid = cv2.VideoCapture(0) 
  
# while(True): 
      
#     ret, frame = vid.read() 
  
#     cv2.imshow('frame', frame) 
      
#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         break
  

# vid.release() 
# cv2.destroyAllWindows() 
