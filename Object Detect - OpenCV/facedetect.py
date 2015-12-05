import sys
import cv2

cascade = '/Users/Senthil/Documents/elec-301/Object Detect - OpenCV/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascade)

image1 = '/Users/Senthil/Documents/elec-301/images/senthil1.jpg'

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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoFeed.release()
    cv2.destroyAllWindows()


def faceDetectTiming(images, display = False):
    image = cv2.imread(images)
    grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    face = faceCascade.detectMultiScale(
            grayImg,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (30,30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

    for (x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)

    cv2.waitKey(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


faceDetectTiming(image1)

