# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:20:01 2023

@author: Kugavathanan
"""
import os

# Set the OMP_NUM_THREADS environment variable to 2
os.environ['OMP_NUM_THREADS'] = '2'

from sklearn.datasets import fetch_olivetti_faces
import numpy as np
import matplotlib.pyplot as plt


data = fetch_olivetti_faces(shuffle=True, random_state=1)

X = data.images
y= data.target

print(f"Number of samples: {X.shape[0]}")
print(f"Image dimensions: {X.shape[1]} x {X.shape[2]}")
print(f"Number of unique target labels: {len(set(y))}")


face_index = np.unique(y, return_index=True)
unique_face = X[face_index[1:]]

# unique_face
for i in range(40):
    face_img = unique_face[i].reshape(64, 64)
    plt.figure(figsize=(3, 3))
    plt.imshow(face_img, cmap='gray')
    plt.show()


from sklearn.model_selection import train_test_split

#training and temporary data set(80-20)
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

#Temporary data set into valid and test data sets (50%-50%)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)


print(f"Number of samples in training set: {len(X_train_temp)}")
print(f"Number of samples in validation set: {len(X_valid)}")
print(f"Number of samples in test set: {len(X_test)}")


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC

#no of folds for communication

k_folds=5

classifier = SVC(kernel='linear')



#K fold validation
cross_val_scores = cross_val_score(classifier, X_train_temp.reshape(len(X_train_temp), -1), y_train_temp, cv=k_folds)


classifier.fit(X_train_temp.reshape(len(X_train_temp), -1), y_train_temp)

validation_accuracy = classifier.score(X_valid.reshape(len(X_valid), -1), y_valid)


# Print K-fold cross-validation scores and validation accuracy
print(f"K-fold Cross-Validation Scores: {cross_val_scores}")
print(f"Mean Cross-Validation Accuracy: {cross_val_scores.mean():.2f}")
print(f"Validation Accuracy: {validation_accuracy:.2f}")



from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#dimensionality reduction using PCA
n_components = 100
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X.reshape(len(X), -1))


# Calculate silhouette scores for a range of K values
k_values = range(2, 11)  
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=1, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)
    silhouette_avg = silhouette_score(X_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)


# Find the number of clusters with the highest silhouette score
best_num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # +2 because we started from 2 clusters
print(f'Best Cluster : {best_num_clusters}')

# Initialize K-Means with the best number of clusters
kmeans = KMeans(n_clusters=best_num_clusters, random_state=1)
X_reduced = kmeans.fit_transform(X.reshape(len(X), -1))
#X_validation_reduced = kmeans.transform(X)

#print(len(X_reduced))


# Create a silhouette score graph
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='-')
plt.title("Silhouette Score vs. DBSCAN Eps Value")
plt.xlabel("Eps Value")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

# Perform K-fold cross-validation on the reduced training set
cross_val_scores_reduced = cross_val_score(classifier, X_reduced, y, cv=k_folds)


print('\n\n\n')
# Print K-fold cross-validation scores for the reduced data
print(f"K-fold Cross-Validation Scores with Reduced Data: {cross_val_scores_reduced}")
print(f"Mean Cross-Validation Accuracy with Reduced Data: {cross_val_scores_reduced.mean():.2f}")




print("\n\n\n\n")

#-----------------------------------------------------------
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_distances


# Preprocess the images (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.reshape(len(X), -1))


# pairwise cosine distances for DBSCAN
cosine_dist = cosine_distances(X_scaled)

#  DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='precomputed')
cluster_labels_dbscan = dbscan.fit_predict(cosine_dist)

print(f'cluster labels of DBSCAN  {cluster_labels_dbscan}')

# Number of clusters found by DBSCAN (including noise points, labeled as -1)
num_clusters_dbscan = len(set(cluster_labels_dbscan)) - (1 if -1 in cluster_labels_dbscan else 0)

print(f'number of clusters in DBSCAN   {num_clusters_dbscan}')


# Assessing the clustering results
unique_clusters = np.unique(cluster_labels_dbscan)
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

for cluster_label in unique_clusters:
    if cluster_label == -1:
        print(f"Cluster {cluster_label}: Noise Points")
    else:
        num_samples_in_cluster = np.sum(cluster_labels_dbscan == cluster_label)
        print(f"Cluster {cluster_label}: {num_samples_in_cluster} samples")

print(f"Number of Clusters (excluding noise): {num_clusters_dbscan}")



plt.figure(figsize=(10, 8))
for cluster_label, color in zip(unique_clusters, colors):
    if cluster_label == -1:
        # Plot noise points in black
        plt.scatter(X_pca[cluster_labels_dbscan == cluster_label, 0],
                    X_pca[cluster_labels_dbscan == cluster_label, 1],
                    color='black', marker='o', s=20, label=f'Cluster {cluster_label} (Noise)')
    else:
        # Plot clustered points with different colors
        plt.scatter(X_pca[cluster_labels_dbscan == cluster_label, 0],
                    X_pca[cluster_labels_dbscan == cluster_label, 1],
                    color=color, marker='o', s=20, label=f'Cluster {cluster_label}')

plt.title("DBSCAN Clustering Results (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(loc="best")
plt.grid(True)
plt.show()


"""
In this code:

We reduce the dimensionality of the data using PCA to two principal components for visualization.
We assign different colors to different clusters and plot the data points in a scatter plot.
Noise points (not assigned to any cluster) are plotted in black.
This scatter plot will give you a visual representation of how the DBSCAN algorithm has grouped the Olivetti Faces dataset into clusters based on cosine similarity. You can observe the cluster structure and the distribution of data points.
"""