import dlib
import cv2 as cv

face_detector_hog = dlib.get_frontal_face_detector()
#body_detector = dlib.full_object_detection()

frame = cv.imread('./HOG/figsHog/people1.jpg')

detector = face_detector_hog(frame, 2)
#detector = body_detector(frame, 1)
print(len(detector))

for face in detector:
    cv.rectangle(frame, (face.left(),face.top()),(face.right(),face.bottom()),(0,255,0),2)


#a=cv.resize(frame, (1200, 720))
cv.imshow('a',frame)
cv.waitKey()