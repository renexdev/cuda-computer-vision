import sys
import cv2
import datetime as dt
import numpy

cascade = '/Users/Senthil/Documents/elec-301/opencv-object-detect/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade)

image1 = '/Users/Senthil/Documents/elec-301/images/birthday1.jpg'
daftPunk = cv2.imread('/Users/Senthil/Downloads/DaftPunk.png', cv2.IMREAD_UNCHANGED)

def faceDetectWebcam():

    videoFeed = cv2.VideoCapture(0)

    while True:
        ret, frame = videoFeed.read()
        grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = faceCascade.detectMultiScale(

            grayImg,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE

        )

        for (x, y, w, h) in face:

            recog = cv2.resize(daftPunk, (w, h), interpolation=cv2.INTER_AREA)
            #print(recog.shape, recog.shape[0], recog.shape[1], w, h)
            for c in range(0,3):
                frame[y:y+h, x:x+w, c] = recog[:,:,c] * (recog[:,:,3]/255.0) +  frame[y:y+h, x:x+w, c] * (1.0 - recog[:,:,3]/255.0)
            #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoFeed.release()
    cv2.destroyAllWindows()


#train classifiers
#train to recognize rich b's face, and apply a daft punk mask

def faceDetectTiming(images):
    image = cv2.imread(images, cv2.IMREAD_UNCHANGED)
    start_time = dt.datetime.now()

    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(
            grayImg,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

    end_time = dt.datetime.now()
    time = (end_time-start_time).microseconds
    print("detected in:" + " " + str(time) + "ms")

    for (x, y, w, h) in face:
        recog = cv2.resize(daftPunk, (w, h), interpolation=cv2.INTER_AREA)
        #print(recog.shape, recog.shape[0], recog.shape[1], w, h)
        #for c in range(0,3):
        #    image[y:y+h, x:x+w, c] = recog[:,:,c] * (recog[:,:,3]/255.0) +  image[y:y+h, x:x+w, c] * (1.0 - recog[:,:,3]/255.0)
        #image[y:y+h, x:x+w] = recog
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)

    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


#faceDetectTiming(image1)
faceDetectWebcam()
