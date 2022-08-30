from PIL import Image
import cv2 as cv
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn
import matplotlib.pyplot as plt
import pandas as pd


lbph_face_classifier = cv.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read('lbpg_classifier.yml')

paths = [os.path.join('Face_Recognition/1/yalefaces', f) for f in os.listdir('./Face_Recognition/1/yalefaces')]
predictions = []
expected_outputs = [] 

for path in paths:
    image  = Image.open(path).convert('L')
    image_np = np.array(image)
    prediction, _ = lbph_face_classifier.predict(image_np)
    expected_output = int(os.path.split(path)[1].split('.')[0].replace('subject',''))
    expected_outputs.append(expected_output)
    predictions.append(prediction)

print(predictions)
print(expected_outputs)
CM = accuracy_score(expected_outputs, predictions)
print(CM) 

cn = confusion_matrix(expected_outputs, predictions)
print(cn)

heat = seaborn.heatmap(cn,annot= True)
print(heat)
