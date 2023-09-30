# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 09:11:25 2023

@author: Kugavathanan
"""

from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

import numpy as np

# Retrieve and load the mnist_784 dataset of 70,000 instances.
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist.data , mnist.target

#print(y)
#print(type(X))


#Display first 10 digit. 

for  i in range(10):
    digit = X.iloc[i].values.reshape(28,28)
    plt.imshow(digit, cmap='gray')
    plt.show()


#Use PCA to retrieve th 1st and 2nd  principal component and output their explained variance ratio.

pca =PCA(n_components=2)
X_pca = pca.fit_transform(X.to_numpy())

explained_variance_ratio = pca.explained_variance_ratio_

print(f'explained variance ratio 1st and 2nd PCA : {explained_variance_ratio}')




#Plot the projections of the 1st and 2nd principal component onto a 1D hyperplane.
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.astype(int), cmap=plt.cm.get_cmap('jet', 10), marker='o', edgecolor='green')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.colorbar(label='Digit')
plt.title('PCA of MNIST Dataset')
plt.show()



#Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. 

ipca = IncrementalPCA(n_components=154)
X_ipca = ipca.fit_transform(X.to_numpy())


#Display the original and compressed digits from (5).

random_samples = np.random.randint(0, X.shape[0],5)

print(random_samples)

plt.figure(figsize=(12,6))

for i, sample_idx in enumerate(random_samples):
    
    plt.subplot(2,4, i+1)
    plt.imshow(X.iloc[sample_idx].values.reshape(28, 28), cmap='gray')
    plt.title(f"Original{i}\nDigit {y[sample_idx]}")
    plt.axis('off')
    
plt.show()

for i, sample_idx in enumerate(random_samples):    
    plt.subplot(2, 4, i + 1)
    compressed_image = ipca.inverse_transform(X_ipca[sample_idx]).reshape(28, 28)
    plt.imshow(compressed_image, cmap='gray')
    plt.title(f"Compressed {i}")
    plt.axis('off')
plt.show()



#





"""
 [5 points]

Display each digit. [5 points]
Use PCA to retrieve the 
 and 
principal component and output their explained variance ratio. [5 points]
Plot the projections of the 
 and 
 principal component onto a 1D hyperplane. [5 points]
Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. [10 points]
Display the original and compressed digits from (5). [5 points]
Create a video discussing the code and result for each question. Discuss challenges you confronted and solutions to overcoming them, if applicable [15 points]


"""