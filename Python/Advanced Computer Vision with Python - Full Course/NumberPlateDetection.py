import os
import cv2
import numpy as np
from imutils import contours

curerntPath = os.path.dirname(os.path.abspath(__file__))
imgFile = os.path.join(curerntPath, 'images')


# Importing class module
from lpce import PlateExtractor
# Generating our istance
extractor = PlateExtractor()
# Apply extraction on a given path (image or an entire folder containing ONLY images)
extractor.apply_extraction_onpath(input_path=imgFile)


# image = cv2.imread(imgFile)
# mask = np.zeros(image.shape, dtype=np.uint8)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# (cnts, _) = contours.sort_contours(cnts, method="left-to-right")
# ROI_number = 0
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area < 800 and area > 200:
#         x,y,w,h = cv2.boundingRect(c)
#         ROI = 255 - thresh[y:y+h, x:x+w]
#         cv2.drawContours(mask, [c], -1, (255,255,255), -1)
#         cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
#         ROI_number += 1

# cv2.imshow('mask', mask)
# cv2.imshow('thresh', thresh)
# cv2.waitKey()

# img = cv2.imread(imgFile)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # read haarcascade for number plate detection
# cascade = cv2.CascadeClassifier('haarcascades\haarcascade_russian_plate_number.xml')

# # Detect license number plates
# plates = cascade.detectMultiScale(gray, 1.2, 5)
# print('Number of detected license plates:', len(plates))

# # loop over all plates
# for (x,y,w,h) in plates:
   
#    # draw bounding rectangle around the license number plate
#    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
#    gray_plates = gray[y:y+h, x:x+w]
#    color_plates = img[y:y+h, x:x+w]
   
#    # save number plate detected
#    cv2.imwrite('Numberplate.jpg', gray_plates)
#    cv2.imshow('Number Plate', gray_plates)
#    cv2.imshow('Number Plate Image', img)
#    cv2.waitKey(0)
# cv2.destroyAllWindows()
