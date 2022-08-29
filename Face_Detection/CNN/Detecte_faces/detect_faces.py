import cv2 as cv
import dlib

frame = cv.imread('./1/CNN/figsCNN/people1.jpg')

cnn_detector = dlib.cnn_face_detection_model_v1('/home/al3x4ndru1/ComputerVision/1/CNN/Weights/mmod_human_face_detector.dat')

detection = cnn_detector(frame,1) # the image and teh ScaleFactor, which is required
#print (len(detection))

for face in detection:
    l, r, t, b, c = face.rect.left(), face.rect.right(), face.rect.top(), face.rect.bottom(), face.confidence
    cv.rectangle(frame,(l,t), (r,b),(0,255,0),2)

cv.imshow('a',frame)
cv.waitKey() 

