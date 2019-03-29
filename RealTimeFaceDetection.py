import cv2
import numpy as np
from keras.preprocessing import image


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

# eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

from keras.models import model_from_json

# This is a CNN model is trained on FER2013 dataset.
model = model_from_json(open("facial_expression_model_structure.json", "r").read())

# Loading model weights
model.load_weights('facial_expression_model_weights.h5')

# These 7 emotions are classified 
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

while True:

    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + h]
        
        #Resizing the facial image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        img_pixels = image.img_to_array(roi_gray)
        
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        
        pred = model.predict(img_pixels)
        
        max_idx = np.argmax(pred[0])
        
        emotion = emotions[max_idx]
        
        cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

     #   eyes = eye_cascade.detectMultiScale(roi_gray)

     #   for(ex, ey, ew, eh) in eyes:

     #       cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    
    
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
