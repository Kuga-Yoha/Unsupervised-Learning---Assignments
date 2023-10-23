# -*- coding: utf-8 -*-
'''
Kugavathanan
'''

import random
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_olivetti_faces
from PIL import Image, ImageEnhance
import random
from scipy.ndimage import rotate
from skimage import exposure


import os

# Set the OMP_NUM_THREADS environment variable to control the number of threads
os.environ["OMP_NUM_THREADS"] = "2"

# Load the Olivetti faces dataset
olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
X = olivetti.data

# Initialize PCA with 99% variance explained
pca = PCA(n_components=0.99, whiten=True, svd_solver='full')

# Fit PCA to the data and transform the data
X_reduced = pca.fit_transform(X)

# Print the number of components selected and the total variance explained
print("Number of components selected:", pca.n_components_)
print("Total variance explained:", np.sum(pca.explained_variance_ratio_))




# Define the parameter grid for covariance type
param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'], }

# Create the GMM model
gmm = GaussianMixture(n_components=40)

# Use GridSearchCV to find the best covariance type
grid_search = GridSearchCV(gmm, param_grid=param_grid, cv=5)  
grid_search.fit(X_reduced)  # X_reduced is the reduced dataset from 

# Get the best covariance type
best_covariance_type = grid_search.best_params_['covariance_type']
print("Best Covariance Type:", best_covariance_type)


# Create an array to store AIC and BIC values
n_components_range = range(30, 100, 10) 
aic_values = []
bic_values = []

# Fit GMM models for different numbers of components
for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type=best_covariance_type)
    gmm.fit(X_reduced)  # X_reduced is the reduced dataset 
    aic_values.append(gmm.aic(X_reduced))
    bic_values.append(gmm.bic(X_reduced))

# Plot AIC and BIC values
plt.figure()
plt.plot(n_components_range, aic_values, label='AIC', marker='o')
plt.plot(n_components_range, bic_values, label='BIC', marker='x')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion Value')
plt.legend()
plt.title('AIC and BIC for Different Numbers of Components')
plt.show()

# Find the number of components with the minimum AIC and BIC
best_aic = 30 + np.argmin(aic_values) *10
best_bic = 30 + np.argmin(bic_values) *10

print("Minimum number of components (AIC):", best_aic)
print("Minimum number of components (BIC):", best_bic)


# Plot the results 
plt.figure(figsize=(12, 5))

# Plot the chosen covariance type
plt.subplot(1, 2, 1)
covariance_types = ['full', 'tied', 'diag', 'spherical']
chosen_covariance_type = covariance_types.index(best_covariance_type)
plt.bar(covariance_types, [1 if i == chosen_covariance_type else 0 for i in range(4)])
plt.title('Best Covariance Type')
plt.xlabel('Covariance Type')
plt.ylabel('Chosen')

# Plot the results
plt.subplot(1, 2, 2)
plt.plot(n_components_range, aic_values, label='AIC', marker='o')
plt.plot(n_components_range, bic_values, label='BIC', marker='x')
plt.xlabel('Number of Components')
plt.ylabel('Information Criterion Value')
plt.legend()
plt.title('Best Number of Components')

plt.tight_layout()
plt.show()


# Create a GMM model with the best parameters 
best_gmm = GaussianMixture(n_components=best_bic, covariance_type=best_covariance_type)
best_gmm.fit(X_reduced)  # X_reduced is the reduced dataset 

# Predict cluster assignments for each instance
cluster_assignments = best_gmm.predict(X_reduced)

# Print or use cluster assignments as needed
print("Hardcluster Cluster Assignments:", cluster_assignments)


# Predict soft cluster probabilities for each instance
soft_cluster_probabilities = best_gmm.predict_proba(X_reduced)

# Print or use the soft cluster probabilities as needed
print("Soft Cluster Probabilities:")
print(soft_cluster_probabilities)


# Number of components in the GMM 
n_components = best_bic 

# Create and fit a GMM model on the reduced data
gmm = GaussianMixture(n_components=n_components, covariance_type=best_covariance_type)
gmm.fit(X_reduced)  # X_reduced is the reduced dataset

# Generate new faces using the GMM sample method
num_samples = 10  # You can adjust the number of samples as needed
generated_samples = gmm.sample(num_samples)
generated_samples = generated_samples[0]  # Extract the generated samples

# Transform the generated samples back to the original space using the PCA
generated_faces = pca.inverse_transform(generated_samples)

# Reshape the generated faces to match the original image shape
generated_faces = generated_faces.reshape((num_samples, 64, 64))

# Visualize the generated faces
plt.figure(figsize=(12, 5))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(generated_faces[i], cmap='gray')
    plt.title(f"Sample {i + 1}")
    plt.axis('off')

plt.suptitle('Generated Faces')
plt.show()


def random_rotate(image):
    angle = random.uniform(-30, 30)
    return rotate(image, angle, reshape=False)


def random_flip(image):
    if random.random() < 0.5:
        return np.fliplr(image)
    else:
        return image


def random_darken(image):
    low_intensity, high_intensity = np.percentile(image, (0.2, 99.8))
    return exposure.rescale_intensity(image, in_range=(low_intensity, high_intensity))


random_indexes = random.sample(range(len(X)), 5)

modified_images = []  # tranform images at these random indexes
for idx in random_indexes:
    image = X[idx].reshape(64, 64)
    image = random_rotate(image)
    image = random_flip(image)
    image = random_darken(image)
    modified_images.append(image)

# Visualize modified images
plt.figure(figsize=(10, 4))
for i, (original, modified) in enumerate(zip(X[random_indexes], modified_images)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(original.reshape(64, 64), cmap='gray')
    plt.axis('off')
    plt.title(f'Original #{i+1}')

    plt.subplot(2, 5, i + 6)
    plt.imshow(modified, cmap='gray')
    plt.axis('off')
    plt.title(f'Modified #{i+1}')
plt.tight_layout()
plt.show()


# 9.Determine if the model can detect the anomalies produced in (8) by comparing the output of the score_samples() method for normal images and for anomalies). [10 points]
normal_scores = best_gmm.score_samples(X_reduced)

modified_images_reshaped = np.array(modified_images).reshape(len(modified_images), -1)
X_modified_pca = pca.transform(modified_images_reshaped)
modified_scores = best_gmm.score_samples(X_modified_pca)

plt.figure(figsize=(8, 6))
plt.scatter(normal_scores, np.zeros_like(normal_scores), color='blue', alpha=0.5, label='Normal Images')
plt.scatter(modified_scores, np.ones_like(modified_scores), color='red', alpha=0.5, label='Modified Images')
plt.xlabel('Negative Log-Likelihood')
plt.yticks([0, 1], ['Normal', 'Modified'])
plt.title('Distribution of Scores')
plt.legend()
plt.show()
