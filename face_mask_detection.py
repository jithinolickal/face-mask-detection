import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

MASK_PREDICTION_VALUE = {0: 'Mask', 1: 'No Mask'} # Define the mask prediction value

def loadMaskData(filepath):
    return np.load(filepath)

def convertTo2DAndCombineData(with_mask, without_mask):
    # converting to 2 dimensional
    with_mask = with_mask.reshape(201, 50*50*3)
    without_mask = without_mask.reshape(201, 50*50*3)

    # Combine data
    return np.r_[with_mask, without_mask] 


def labelMaskData(X):
    # labeling with and without mask inside combined data
    # Total data is ~400 frames
    # 0 defines with mask (first 201 data from input)
    # 1 defines with mask (last 201 data from input)
    labels = np.zeros(X.shape[0])
    labels[200:] = 1.0
    return labels

def trainModel(X, labels):
    x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25) # uses 25% data for testing, and rest 75% for training model
    print("x_train.shape -- ", x_train.shape)

    # Reduce dimensions for faster training (3 dimension)
    pca = PCA(n_components=3)
    x_train = pca.fit_transform(x_train)
    print("x_train -- ", x_train[0])
    print("x_train.shape -- ", x_train.shape)


    # apply machine learning
    svm = SVC()
    svm.fit(x_train, y_train)

    x_test = pca.transform(x_test) # Converts the test data too to 3 dimension
    y_pred = svm.predict(x_test)

    print("Model Accuracy == ", accuracy_score(y_test, y_pred))

    return [pca, svm]


def live(pca, svm):
    haar_data = cv2.CascadeClassifier('./haarcascade_face_data/haarcascade_frontalface_default.xml') # Loading Haar data
    capture = cv2.VideoCapture(0)
    data = []
    font = cv2.FONT_HERSHEY_COMPLEX
    while True:
        flag, image = capture.read()
        if flag:
            faces = haar_data.detectMultiScale(image)
            for x,y,w,h in faces:
                cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,255), 4)
                face = image[y:y+h, x:x+w, :] # Slice the face from live photo
                face = cv2.resize(face, (50,50))
                face = face.reshape(1,-1)
                face = pca.transform(face)
                pred = svm.predict(face)
                n = MASK_PREDICTION_VALUE[int(pred)]
                cv2.putText(image, n, (x,y), font, 1, (244,250,250),2)
                print(n)
            cv2.imshow('Image', image)
            if cv2.waitKey(2) == 27:
                break

    capture.release()
    cv2.destroyAllWindows()


def main():
    with_mask = loadMaskData('./training_data/with_mask.npy')
    without_mask = loadMaskData('./training_data/without_mask.npy')
    
    print(with_mask.shape)
    print(without_mask.shape)

    # converting to 2 dimensional data from w,h,c
    X = convertTo2DAndCombineData(with_mask, without_mask)

    print("X.shape -- ", X.shape)

    labels = labelMaskData(X)

    # -------------------- use machine learning using sklearn --------------------
    [pca, svm] = trainModel(X, labels)

    # -------------------- Testing live --------------------
    live(pca,svm)


main()
