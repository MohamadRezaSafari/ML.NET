import os
import time
import cv2
import mediapipe as mp

class FaceDetector():
    def __init__(self, minDetectionCon=0.5):        
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxc = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih), int(bboxc.width * iw), int(bboxc.height * ih)
                bboxs.append([id, bbox, detection.score])

                cv2.rectangle(img, bbox, (225,0,255), 2)
                cv2.putText(img, 
                            f'{int(detection.score[0] * 100)}%', 
                            (bbox[0], bbox[1] - 20), 
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0 , 255), 2)
                
        return img, bboxs

def main():
    curerntPath = os.path.dirname(os.path.abspath(__file__))
    videoFile = os.path.join(curerntPath, 'videos/4.mp4')
    cap = cv2.VideoCapture(videoFile)
    pTime = 0
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bbox = detector.findFaces(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()