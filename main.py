import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


with_mask = np.load('./training_data/with_mask.npy')
without_mask = np.load('./training_data/without_mask.npy')

print(with_mask.shape)
print(without_mask.shape)

# converting to 2 dimensional
with_mask = with_mask.reshape(201, 50*50*3)
without_mask = without_mask.reshape(201, 50*50*3)

X = np.r_[with_mask, without_mask] # Combine data

print("X.shape -- ", X.shape)
# labeling with and without mask inside combined data
# 0 defines with mask (first 101 data from input)
# 1 defines with mask (last 101 data from input)
labels = np.zeros(X.shape[0])
labels[200:] = 1.0

names = {0: 'Mask', 1: 'No Mask'} # Define the mask prediction value

# -------------------- use machine learning using sklearn ------------

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.25) # uses 25% data for testing, and rest 75% for training model
# x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.20) # uses 25% data for testing, and rest 75% for training model

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

print("Find the trained model accuracy == ", accuracy_score(y_test, y_pred))


# ---------------------- Testing live --------------------

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
            n = names[int(pred)]
            cv2.putText(image, n, (x,y), font, 1, (244,250,250),2)
            print(n)
        cv2.imshow('Image', image)
        if cv2.waitKey(2) == 27:
            break

capture.release()
cv2.destroyAllWindows()
# # img.shape
# # print("img.shape -- ", img.shape)
# plt.imshow(data[0])

# # Show image in popup
# while True:
#     cv2.imshow('Image', data[0])
#     if cv2.waitKey(2) == 27: # press escape to exit image popup
#         break
# cv2.destroyAllWindows()