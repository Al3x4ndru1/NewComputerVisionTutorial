from PIL import Image
import cv2 as cv
import numpy as np
import os


lbph_face_classifier = cv.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('lbpg_classifier.yml')

test_image = 'Face_Recognition/1/yalefaces/subject10.sad'
image = Image.open(test_image).convert('L')
image_np = np.array(image)

prediction = lbph_face_classifier.predict(image_np)
print (prediction)

expected_output = int(os.path.split(test_image)[1].split('.')[0].replace('subject',''))
print(expected_output)

cv.putText(image_np, 'Pred:' + str(prediction[0]), (10,30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv.putText(image_np, 'Conf:' + str(prediction[1]), (10,50), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv.putText(image_np, 'Exp:' + str(expected_output), (10,70), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0))
cv.imshow('a',image_np)
cv.waitKey()
