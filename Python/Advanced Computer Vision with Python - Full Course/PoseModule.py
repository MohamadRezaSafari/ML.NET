import os
import time
import cv2
import mediapipe as mp


class poseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True, complexity=1, detectionCon=0.5, trackCon=0.5):
        # self.mode = mode
        # self.upBody = upBody
        # self.smooth = smooth
        # self.complexity = complexity
        # self.detectionCon = detectionCon
        # self.trackCon = trackCon

        self.mode=mode,
        self.complexity=complexity,
        self.smooth=smooth,
        self.upBody=upBody,
        # self.smooth=smooth,
        self.detectionCon=detectionCon,
        self.trackCon=trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        # self.pose = self.mpPose.Pose(self.mode, 1, self.smooth, self.upBody, self.smooth,
        #                              self.detectionCon, self.trackCon)
        self.pose = self.mpPose.Pose(self.mode, 1, self.smooth, upBody, self.smooth, 
                                     detectionCon, self.trackCon)
        
    def findPos(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)
        return lmList

def main():
    curerntPath = os.path.dirname(os.path.abspath(__file__))
    videoFile = os.path.join(curerntPath, 'videos/1.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPos(img)
        lmList = detector.findPosition(img, draw=False)


        try:
            print(lmList[5])
            cv2.circle(img, (lmList[5][1], lmList[5][2]), 15, (0,0,255), cv2.FILLED)
        except IndexError:
            gotdata = 'null'
            

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0 , 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()