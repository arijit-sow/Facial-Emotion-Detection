from keras.models import load_model
from time import sleep
from keras.utils import img_to_array
import warnings
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'E:/git file/Emotion_Detection_CNN/haarcascades/haarcascade_frontalface_default.xml')
classifier =load_model(r'E:/git file/Emotion_Detection_CNN/model/model-1.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
img_size=2304
cap = cv2.VideoCapture(0)

while True:
        success, frame = cap.read()

        if not success:
             break
        
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(img_size,img_size))



            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                # roi = np.expand_dims(roi,axis=0)
                roi = np.squeeze(roi,axis=-1)


                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                label_position = (x,y-10)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            cap.release()
            cv2.destroyAllWindows()