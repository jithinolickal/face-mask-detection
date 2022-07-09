import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():

    # ------------------------ test -----------------------------

    # img = cv2.imread('./test_image.jpg')
    # img.shape
    # print("img.shape -- ", img.shape)
    # plt.imshow(img)

    # Show image in popup
    # while True:
    #     cv2.imshow('Image', img)
    #     if cv2.waitKey(2) == 27: # press escape to exit image popup
    #         break
    # cv2.destroyAllWindows()

    # haar_data = cv2.CascadeClassifier('./haarcascade_face_data/haarcascade_frontalface_default.xml') # Loading Haar data

    # print("Faces -- ", haar_data.detectMultiScale(img)) # Detects the dimensions of a face if present

    # Draw Rectangle
    # while True:
    #     faces = haar_data.detectMultiScale(img)
    #     for x,y,w,h in faces:
    #         cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
    #     cv2.imshow('Image', img)
    #     if cv2.waitKey(2) == 27: # press escape to exit image popup
    #         break
    # cv2.destroyAllWindows()

    # ------------------------ use this to create with and without mask data ------------------------

    haar_data = cv2.CascadeClassifier('./haarcascade_face_data/haarcascade_frontalface_default.xml') # Loading Haar data

    # capture = cv2.VideoCapture('videofile.mp4')
    capture = cv2.VideoCapture(0)
    IMAGE_FRAME_COUNT = 400
    data = []
    while True:
        flag, image = capture.read()
        if flag:
            faces = haar_data.detectMultiScale(image)
            for x,y,w,h in faces:
                cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 4)
                face = image[y:y+h, x:x+w, :] # Slice the face from live photo
                face = cv2.resize(face, (50,50))
                print(len(data))
                if len(data) < IMAGE_FRAME_COUNT:
                    data.append(face)
            cv2.imshow('Image', image)
            if cv2.waitKey(2) == 27 or len(data) > IMAGE_FRAME_COUNT:
                break

    capture.release()
    cv2.destroyAllWindows()
    np.save('./training_data/without_mask.npy', data) # Saving as .npy file

main()