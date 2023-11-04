# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:31:41 2023

@author: Kugavathanan Yohanathan
301315801
"""


from sklearn import datasets

import matplotlib.pyplot as plt
import numpy as np

#Retrieve and load the Olivetti faces dataset [0 points]

faces = datasets.fetch_olivetti_faces(shuffle=True, random_state=1)

X = faces.data
y = faces.target

print(X[:5])

print(y[:5])


unique_y = np.unique(y, return_index=True)
unique_faces = X[unique_y[1]]

for i in range(40):
    face = unique_faces[i].reshape(64, 64)
    plt.figure(figsize=(3, 3))
    plt.imshow(face, cmap='gray')
    plt.show()




from sklearn.model_selection import train_test_split


# Split the data into training (80%) and temporary data (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Split the temporary data into validation (10%) and test (10%)
#validation set - hyperparmater tuning 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)


print(X_train)

# Normalize 
X_train = X_train
X_test = X_test
X_train.shape 




from sklearn.decomposition import PCA
import numpy as np

# Initialize PCA with 99% variance explained
pca= PCA(n_components=0.99, whiten=True, svd_solver='full', random_state=1)
pca.fit(X_train)

# Fit PCA to the data and transform the data
X_train_pca = pca.transform(X_train)
X_valid_pca  = pca.transform(X_val)
X_test_pca = pca.transform(X_test)
print(X_train_pca.shape)

# Print the number of components selected and the total variance explained
print("Number of components selected:", pca.n_components_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))



#-------------------------------------------------------------------------------------------

from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from sklearn.model_selection import KFold


input_size = X_train_pca.shape[1]
hidden_values = [64, 128, 256]  # Hidden layers
central_units = [16,32]  # Central layer

best_model = None
best_score = float('inf')
best_hyperparameters = {}

learning_rates = [0.001, 0.01, 0.1]

kfold = KFold(n_splits=5, shuffle=True, random_state=32)

for units in hidden_values:
    for units_2 in central_units:
        for train_idx, val_idx in kfold.split(X_train_pca):
            X_train_fold, X_val_fold = X_train_pca[train_idx], X_train_pca[val_idx]


            input_img = Input(shape=(input_size,))
            
            hidden1 = Dense(units, activation='relu', kernel_regularizer='l1')(input_img)
            
            central_code = Dense(units_2, activation='relu', kernel_regularizer='l1')(hidden1)
            
            hidden3 = Dense(units, activation='relu', kernel_regularizer='l1')(central_code)
            
            output_img = Dense(input_size, activation='linear')(hidden3)

            autoencoder = Model(input_img, output_img)
            
            autoencoder.compile(optimizer=Adam(learning_rate=0.1), loss='mean_squared_error')
            

            autoencoder.fit(X_train_fold, X_train_fold, epochs=500, batch_size=32, verbose=0)

            val_loss = autoencoder.evaluate(X_val_fold, X_val_fold, verbose=1)
            
            
            print("model hidden and central units", units, units_2)

            if val_loss < best_score:
                best_score = val_loss
                best_model = autoencoder
                best_hyperparameters = {
                    'hidden_values': units,
                    'central_units': units_2
                }

print("Best hyperparameters:", best_hyperparameters)

input_size = X_train_pca.shape[1]
hidden_size = best_hyperparameters['hidden_values']
code_size = best_hyperparameters['central_units']

"""
input_img = Input(shape = (input_size,))
# Encoder layers
hidden1 = Dense(hidden_size,activation='relu')(input_img)
code = Dense(code_size,activation='relu')(hidden1)
# Decoder layers
hidden2 = Dense(hidden_size,activation='relu')(code)
output_img=Dense(input_size,activation='sigmoid')(hidden2)
"""


#----------------------------------------------------------
decoded_imgs = best_model.predict(X_test_pca)
decoded_imgs_original_space = pca.inverse_transform(decoded_imgs)

print(X_test_pca)
print(decoded_imgs)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(64, 64), cmap='gray')

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs_original_space[i].reshape(64, 64), cmap='gray')

plt.show()




