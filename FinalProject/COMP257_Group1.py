#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Kugavathanan  Yohanathan
         Sayanthana  Yoganathan
         
         COMP-257
"""
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras 
from sklearn.metrics import classification_report, confusion_matrix

file_path = r'C:\Users\ysaya\OneDrive\Desktop\Semester-5\Unsupervised\groupProject\umist_cropped.mat'

# Load the mat file
face_data = scipy.io.loadmat(file_path)
print(dir(face_data))
#print(face_data.keys())
data=face_data['facedat']

print(type(data))

#print(data)

print(data[0].shape)
print(data[0][0].shape)

# Restructuring the data
dataX=np.empty((0,))
dataY=[]
total=0

for i,v in enumerate(data[0]):
    print(f'v.transpose((2, 0, 1)).shape: {v.transpose((2, 0, 1)).shape}')
    total+=v.transpose((2, 0, 1)).shape[0]
    dataX=np.append(dataX, v.transpose((2, 0, 1)).ravel(), axis=0)
    for j in range(v.transpose((2, 0, 1)).shape[0]):
        dataY.append(i)

print(total)    #575

dataX=dataX.reshape(total,112,92)
dataY=np.array(dataY)
print(dataX.shape)
print(dataY.shape)



# Show all images
'''
for i in dataX:
    plt.imshow(i,cmap='Greys')
    plt.show()
'''
# splitting the data train:validation:test= 70:15:15
sss = StratifiedShuffleSplit(test_size=0.3, random_state=10)

for train_index, test_index in sss.split(dataX, dataY):
    X_train, X_temp = dataX[train_index], dataX[test_index]
    y_train, y_temp = dataY[train_index], dataY[test_index]

sss_validation = StratifiedShuffleSplit (test_size=0.5, random_state=10)

for train_index, validation_index in sss_validation.split(X_temp, y_temp):
    X_validation, X_test = X_temp[train_index], X_temp[validation_index]
    y_validation, y_test = y_temp[train_index], y_temp[validation_index]
    

print(np.unique(y_train,return_counts=True))
print(np.unique(y_validation,return_counts=True))
print(np.unique(y_test,return_counts=True))

print(X_train.shape,X_validation.shape,X_test.shape)
print(y_train.shape,y_validation.shape,y_test.shape)
# reshaping the data as per the input for our models
X_train=X_train.reshape(X_train.shape[0],112*92)
X_validation=X_validation.reshape(X_validation.shape[0],112*92)
X_test=X_test.reshape(X_test.shape[0],112*92)
print(X_train.shape,X_validation.shape,X_test.shape)

# transformed the data using LinearDiscriminant method
projector = LinearDiscriminantAnalysis(n_components=5)
faces_points = projector.fit_transform(X=X_train,y=y_train)

faces_points_validation=projector.transform(X_validation)
faces_points_test=projector.transform(X_test)

print(faces_points)
print(faces_points.shape)

#plotting the distribution of actual data
plt.figure(figsize=(15,15))
plt.scatter(faces_points[:,0],faces_points[:,1],c=y_train, cmap='viridis')
plt.title('Actual Data Before Clustering')
plt.colorbar()
plt.show()


# DBSCAN to cluster the DATA with various hyper parameter
score={}
for distance in np.arange(2,10,1): # 2 to 10
    for samples in range(2,5):    #2 to 5
        model = DBSCAN(eps=distance,min_samples=samples)
        faces_y_pred = model.fit_predict(faces_points)
        print(distance,samples,silhouette_score(faces_points,y_train),silhouette_score(faces_points,faces_y_pred))
        plt.figure(figsize=(15,15))
        plt.scatter(faces_points[:, 0], faces_points[:, 1], c=faces_y_pred, cmap='viridis')
        plt.title('DBSCAN Clustering Result EPS'+str(distance)+" min samples "+str(samples))
        plt.colorbar()
        plt.show()
        score["EPS "+str(distance)+" min samples "+str(samples)]=silhouette_score(faces_points,faces_y_pred)

plt.figure(figsize=(15,15))
plt.bar(score.keys(), score.values())
plt.xticks(
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'  
)
plt.show()

# best DBSCAN model for clustering 
model = DBSCAN(eps=5,min_samples=2)
faces_y_pred = model.fit_predict(faces_points)
print(i,silhouette_score(faces_points,y_train),silhouette_score(faces_points,faces_y_pred))
plt.figure(figsize=(15,15))
plt.scatter(faces_points[:, 0], faces_points[:, 1], c=faces_y_pred, cmap='viridis')
plt.title('DBSCAN Clustering Result EPS'+str(i)+" min samples "+str(j))
plt.colorbar()
plt.show()



print(np.unique(faces_y_pred))

for cluster in np.unique(faces_y_pred):
    if input("next Cluster is " +str(cluster)):
        unique_face=X_train[np.where(faces_y_pred==cluster)]
        for i in range(len(unique_face)):
            face_img = unique_face[i].reshape(112,92)
            plt.figure(figsize=(10,10))
            plt.imshow(face_img, cmap='gray')
            plt.show()

faces_y_pred_validation=model.fit_predict(faces_points_validation)
faces_y_pred_test=model.fit_predict(faces_points_test)



# Dimensionality reduction
X_train=X_train/255
X_validation=X_validation/255
X_test=X_test/255
''' ######################EXPERIMENTAL CODE######################
y_train=faces_y_pred
y_validation=faces_y_pred_validation
y_test=faces_y_pred_test
'''
model_pca=PCA(n_components=0.99)

X_train_pca=model_pca.fit_transform(X_train)
X_validation_pca=model_pca.transform(X_validation)
X_test_pca=model_pca.transform(X_test)
print(X_train_pca.shape,X_validation_pca.shape,X_test.shape)
print(y_train.shape,y_validation.shape,y_test.shape)

'''
'''


model=keras.models.Sequential()

model.add(keras.layers.InputLayer(input_shape=X_train_pca[0].shape[0]))
model.add(keras.layers.Dense(200, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(20, activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy','accuracy'])

model.summary()

keras_callbacks   = [
      EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, min_delta=0.1), 
      ModelCheckpoint("", monitor='val_loss', save_best_only=True, mode='min')
]

history=model.fit(X_train_pca,y_train,epochs=300,validation_data=(X_validation_pca,y_validation),callbacks=keras_callbacks)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()



predict=model.predict(model_pca.transform(X_test))

prediction=[np.argmax(x) for x in predict]
print(prediction,y_test)

print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))



def createmodel(hidden=[200,100],activ="relu"):
    model=keras.models.Sequential()

    model.add(keras.layers.InputLayer(input_shape=X_train_pca[0].shape[0]))
    for i in hidden:
        model.add(keras.layers.Dense(i, activation=activ))
    model.add(keras.layers.Dense(20, activation='softmax'))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy','accuracy'])

    return model

modeldict={}
modeldict[createmodel([100,100],"sigmoid")]=-1
modeldict[createmodel([200,200],"sigmoid")]=-1
modeldict[createmodel([100,100],"relu")]=-1
modeldict[createmodel([200,200],"relu")]=-1
modeldict[createmodel([200,100],"relu")]=-1



keras_callbacks   = [
      EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10, min_delta=0.01), 
      ModelCheckpoint("", monitor='val_loss', save_best_only=True, mode='min')
]
for model in modeldict.keys():
    history=model.fit(X_train_pca,y_train,epochs=300,validation_data=(X_validation_pca,y_validation),callbacks=keras_callbacks)
    modeldict[model]=history.history['val_accuracy'][-1]


print(modeldict)

for model in modeldict.keys():
    predict=model.predict(model_pca.transform(X_test))

    prediction=[np.argmax(x) for x in predict]
    #print(prediction,y_test)

    print(classification_report(y_test, prediction))

    print(confusion_matrix(y_test, prediction))

print("---------------------FINAL RESULTS----------------------")

final_model=keras.models.load_model("BestANNModel.keras")

predict_final=model.predict(model_pca.transform(X_test))

prediction_final=[np.argmax(x) for x in predict_final]
#print(prediction_final,y_test)
print(classification_report(y_test, prediction_final))

print(confusion_matrix(y_test, prediction_final))







