import os
import cv2
import mediapipe as mp
import time


curerntPath = os.path.dirname(os.path.abspath(__file__))
videoFile = os.path.join(curerntPath, 'videos/4.mp4')
cap = cv2.VideoCapture(0)
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1)
draqSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_TESSELATION, draqSpec, draqSpec)
            for lm in faceLms.landmark:
                print(lm)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 , 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

