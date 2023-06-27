import cv2

from random import randrange

cap = cv2.VideoCapture(0)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    success, img = cap.read()

    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    (face_coordinates)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(img, (x, y), (x + w, y + h), ((256),(256),(256)), 2)


    cv2.imshow('Image',img)
    key = cv2.waitKey(1)
    if key == ord("b"):
        break

cap.release()









