# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 04:04:54 2023

@author: Kugavathanan
301315801
Assignment 3 
Heriachial Clustering
"""

"""
This assignment will be similar to Assignment 2 but we will use hierarchical clustering in place of K-Means.

Retrieve and load the Olivetti faces dataset [0 points]
Split the training set, a validation set, and a test set using stratified sampling to ensure that there are the same number of images per person in each set. [0 points]
Using k-fold cross validation, train a classifier to predict which person is represented in each picture, and evaluate it on the validation set. [0 points]


Using either Agglomerative Hierarchical Clustering (AHC) or Divisive Hierarchical Clustering (DHC) and using the centroid-based clustering rule, reduce the dimensionality of the set by using the following similarity measures:
a) Euclidean Distance [20 points]
b) Minkowski Distance [20 points]
c) Cosine Similarity [20 points]
Discuss any discrepancies observed between 4(a), 4(b), or 4(c).
Use the silhouette score approach to choose the number of clusters for 4(a), 4(b), and 4(c). [10 points]
Use the set from (4(a), 4(b), or 4(c)) to train a classifier as in (3) using k-fold cross validation. [30 points]



"""

from sklearn import datasets



#Retrieve and load the Olivetti faces dataset [0 points]

faces = datasets.fetch_olivetti_faces(shuffle=True, random_state=1)

X = faces.data
y = faces.target

print(X[:5])

print(y[:5])

from sklearn.model_selection import train_test_split


# Split the data into training (80%) and temporary data (20%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# Split the temporary data into validation (10%) and test (10%)
#validation set - hyperparmater tuning 
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)

#

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
"""
 for classification tasks, it's often a good practice to use StratifiedKFold when performing k-fold cross-validation, especially when you have
 imbalanced classes or when you want to ensure that the class distribution in each fold closely matches the original dataset. 
 This helps prevent issues where a particular class is underrepresented in some folds, which could lead to biased results.
"""


#from sklearn.neighbors import KNeighborsClassifier  #this can be replaced

# Create a StratifiedKFold object
stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)


#create SVC -Support Vector Classification , Type of SVM
svm_classifier = SVC()

#Perform k-fold cross validation to train & evaluate the classifier
cross_val_scores = cross_val_score(svm_classifier, X_train, y_train, cv=stratified_kfold)



print('\n\n\n')

#Cross validation scores
print(f'Cross validation scores: {cross_val_scores}')
print(f'Mean Accuracy: {cross_val_scores.mean()}')


#train the classifier  on training set
svm_classifier.fit(X_train, y_train)


#evaluate the validation set on the trained classifier
validation_accuracy = svm_classifier.score(X_val, y_val)

print(f'Validation Set Accuracy: {validation_accuracy}')

print('\n\n\n')


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the number of images to visualize
num_images_to_visualize = 5


# Step 4: Hierarchical Clustering with different similarity measures
num_clusters = 5 # Number of clusters

#---------------------------------- Euclidean Distance--------------------------------------------------------------------
agg_clustering_euclidean = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='ward')
agg_labels_euclidean = agg_clustering_euclidean.fit_predict(X_train)

#Reduce dimensionality using PCA for Euclidean Distance
pca_euclidean = PCA(n_components=2)
X_pca_euclidean = pca_euclidean.fit_transform(X_train)

# Plot PCA reduction for Euclidean Distance
plt.figure(figsize=(6, 5))
plt.scatter(X_pca_euclidean[:, 0], X_pca_euclidean[:, 1], c=agg_labels_euclidean, cmap='viridis', marker='o')
plt.title('PCA Reduction - Euclidean Distance')

plt.show()

"""
# Calculate cluster centroids for Euclidean Distance
cluster_centroids_euclidean =[]
for label in range(num_clusters):
    cluster_data = X[agg_labels_euclidean == label]
    cluster_centroid = np.mean(cluster_data, axis=0)
    cluster_centroids_euclidean.append(cluster_centroid)
"""

#cluster_centroids_euclidean = np.array([X[agg_labels_euclidean == label].mean(axis=0) for label in range(num_clusters)])


# ----------------------------------Minkowski Distance------------------------------------------------------------------------
agg_clustering_minkowski = AgglomerativeClustering(n_clusters=num_clusters, metric='manhattan', linkage='complete')
agg_labels_minkowski = agg_clustering_minkowski.fit_predict(X_train)

#Reduce dimensionality using PCA for Euclidean Distance
pca_minkowski = PCA(n_components=2)
X_pca_minkowski = pca_minkowski.fit_transform(X_train)



# Plot PCA reduction for Minkowski Distance
plt.figure(figsize=(6, 5))
plt.title('PCA Reduction - Manhattan Distance')
plt.scatter(X_pca_minkowski[:, 0], X_pca_minkowski[:, 1], c=agg_labels_minkowski, cmap='viridis', marker='o')


plt.show()



"""
# Calculate cluster centroids for Minkowski Distance
cluster_centroids_minkowski = []
for label in range(num_clusters):
    cluster_data = X[agg_labels_minkowski == label]
    cluster_centroid = np.mean(cluster_data, axis=0)
    cluster_centroids_minkowski.append(cluster_centroid)

#cluster_centroids_minkowski = np.array(cluster_centroids_minkowski)

"""

#---------------------------------Cosine Similarity-----------------------------------------------------------------------------
cosine_sim_matrix = cosine_similarity(X_train)
agg_clustering_cosine = AgglomerativeClustering(n_clusters=num_clusters, metric='precomputed', linkage='complete')
agg_labels_cosine = agg_clustering_cosine.fit_predict(1 - cosine_sim_matrix)


#Reduce dimensionality using PCA for Euclidean Distance
pca_cosine = PCA(n_components=2)
X_pca_cosine = pca_minkowski.fit_transform(X_train)



# Plot PCA reduction for Cosine Similarity
plt.figure(figsize=(6, 5))
plt.title('PCA Reduction - Cosine Distance')
plt.scatter(X_pca_cosine[:, 0], X_pca_cosine[:, 1], c=agg_labels_cosine, cmap='viridis', marker='o')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


"""

import matplotlib.pyplot as plt

# Assuming you have already performed PCA for each measure as shown in the previous responses:
# X_train_pca_euclidean, X_train_pca_minkowski, X_train_pca_cosine

# Create subplots for the three measures
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot PCA visualization for Euclidean Distance
axs[0].scatter(X_pca_euclidean[:, 0], X_pca_euclidean[:, 1], c=agg_labels_euclidean, cmap='viridis', marker='o')
axs[0].set_xlabel('Principal Component 1')
axs[0].set_ylabel('Principal Component 2')
axs[0].set_title('PCA - Euclidean Distance')

# Plot PCA visualization for Minkowski Distance
axs[1].scatter(X_pca_minkowski[:, 0], X_pca_minkowski[:, 1], c=agg_labels_minkowski, cmap='viridis', marker='o')
axs[1].set_xlabel('Principal Component 1')
axs[1].set_ylabel('Principal Component 2')
axs[1].set_title('PCA - Minkowski Distance')

# Plot PCA visualization for Cosine Similarity
axs[2].scatter(X_pca_cosine[:, 0], X_pca_cosine[:, 1], c=agg_labels_cosine, cmap='viridis', marker='o')
axs[2].set_xlabel('Principal Component 1')
axs[2].set_ylabel('Principal Component 2')
axs[2].set_title('PCA - Cosine Similarity')

# Add a colorbar to the third subplot
colorbar = plt.colorbar(axs[2].scatter(X_pca_cosine[:, 0], X_pca_cosine[:, 1], c=agg_labels_cosine, cmap='viridis', marker='o'))
colorbar.set_label('Class Label', rotation=270, labelpad=20)

# Show the plots
plt.tight_layout()
plt.show()

"""
cluster_range = range(2,20)

# Calculate silhouette scores for each similarity measure
silhouette_scores_euclidean = silhouette_score(X_train, agg_labels_euclidean)
silhouette_scores_minkowski = silhouette_score(X_train, agg_labels_minkowski)
silhouette_scores_cosine = silhouette_score(X_train, agg_labels_cosine)


print(f'silhouette_scores_euclidean : {silhouette_scores_euclidean}\n')
print(f'silhouette_scores_minkowski : {silhouette_scores_minkowski}\n')
print(f'silhouette_scores_cosine : {silhouette_scores_cosine}\n\n')


# Find the number of clusters with the highest silhouette score
best_n_clusters_euclidean = cluster_range[np.argmax(silhouette_scores_euclidean)]
best_n_clusters_minkowski = cluster_range[np.argmax(silhouette_scores_minkowski)]
best_n_clusters_cosine = cluster_range[np.argmax(silhouette_scores_cosine)]

print(f'best_n_clusters_euclidean : {best_n_clusters_euclidean}\n')
print(f'best_n_clusters_minkowski : {best_n_clusters_minkowski}\n')
print(f'best_n_clusters_cosine : {best_n_clusters_cosine}')

from sklearn.model_selection import KFold

# Define the number of splits (K)
n_splits = 3 # You can change this to the desired number of splits

# Create a K-Fold cross-validation object
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)


svm_classifier = SVC(kernel='linear')
# Perform k-fold cross-validation for each similarity measure
# Euclidean Distance
cross_val_scores_euclidean = cross_val_score(svm_classifier, X_train, agg_labels_euclidean, cv=kfold)

# Minkowski Distance
cross_val_scores_minkowski = cross_val_score(svm_classifier, X_train, agg_labels_minkowski, cv=kfold)

# Cosine Similarity
cross_val_scores_cosine = cross_val_score(svm_classifier, X_train, agg_labels_cosine, cv=kfold)

# Print the cross-validation scores for each similarity measure

print("Cross-Validation Scores - Euclidean Distance:", cross_val_scores_euclidean)
print("Mean Cross-Validation Score - Euclidean Distance:", cross_val_scores_euclidean.mean())

print('\n\n')
print("Cross-Validation Scores - Minkowski Distance:", cross_val_scores_minkowski)
print("Mean Cross-Validation Score - Minkowski Distance:", cross_val_scores_minkowski.mean())

print('\n\n')
print("Cross-Validation Scores - Cosine Similarity:", cross_val_scores_cosine)
print("Mean Cross-Validation Score - Cosine Similarity:", cross_val_scores_cosine.mean())






