import cv2
import numpy as np
from keras.preprocessing import image
import os

def get_emojis():
    emojis_folder = 'emojis/'
    emojis = []
    for emoji in range(len(os.listdir(emojis_folder))):
        emojis.append(cv2.imread(emojis_folder+str(emoji)+'.png', -1))
    return emojis


def overlay(image, emoji, x,y,w,h):
    emoji = cv2.resize(emoji, (w, h))
    try:
         image[y:y+h, x:x+w] = blend_transparent(image[y:y+h, x:x+w], emoji)
    except:
        pass
    return image


def blend_transparent(face_img, overlay_t_img):
    # Split out the transparency mask from the colour info
    overlay_img = overlay_t_img[:,:,:3] # Grab the BRG planes
    overlay_mask = overlay_t_img[:,:,3:]  # And the alpha plane

    # Again calculate the inverse mask
    background_mask = 255 - overlay_mask

    # Turn the masks into three channel, so we can use them as weights
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    # Create a masked out face image, and masked out overlay
    # We convert the images to floating point in range 0.0 - 1.0
    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))

    # And finally just add them together, and rescale it back to an 8bit integer image
    return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))


def main():

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    from keras.models import model_from_json

    # This is a CNN model is trained on FER2013 dataset.
    model = model_from_json(open("modelStructure.json", "r").read())

    # Loading model weights
    model.load_weights('modelWeights.h5')

    # These 7 emotions are classified 
    #emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    emotions = get_emojis()

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

            img = overlay(img, emotion, 400, 250, 90, 90)

        cv2.imshow('img', img)

        k = cv2.waitKey(30) & 0xff

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

