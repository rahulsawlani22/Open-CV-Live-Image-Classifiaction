import numpy as np
import cv2
import os
import pickle

labels={}
with open("label-ids.pickle",'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade=cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
eye_cascade=cv2.CascadeClassifier('data/haarcascade_eye.xml')
recognizer.read("trainer.yml")

cap=cv2.VideoCapture(0)

while(1):
    res,frame=cap.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=5)
    for (x,y,z,w) in faces:
        #print(x,y,z,w)
        roi_frame=frame[y:y+w,x:x+z]
        roi_gray=gray[y:y+w,x:x+z]

        id_,conf=recognizer.predict(roi_gray)
        if conf>45:#B and conf<=85:
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=2
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        img_item="my_img.png"
        cv2.imwrite(img_item,roi_frame)
        color=(255, 0, 0)
        stroke=2
        height=y+w
        width=x+z
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)
    eyes=eye_cascade.detectMultiScale(frame,scaleFactor=1.5,minNeighbors=5)
    for (x,y,z,w) in eyes:
        #print(x,y,z,w)
        roi_frame=frame[y:y+w,x:x+z]
        color=(255, 0, 0)
        stroke=2
        height=y+w
        width=x+z
        cv2.rectangle(frame,(x,y),(width,height),color,stroke)

    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
