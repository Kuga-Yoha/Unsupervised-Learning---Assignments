import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

#1: Generate Swiss roll data
X_swiss_roll, y_swiss_roll = make_swiss_roll(n_samples=1000, noise=0.05, random_state=10)

#2:Plot the resulting generated Swiss roll dataset
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_swiss_roll[:, 0], X_swiss_roll[:, 1], X_swiss_roll[:, 2], c=y_swiss_roll, cmap=plt.cm.hot)
plt.show()

#3: Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points)
#4 Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points) from (3
#Explain and compare the results
kernels = ['linear', 'rbf', 'sigmoid']
for kernel in kernels:
    print(f"--------------------- {kernel} ---------------------")
    pca = KernelPCA(n_components=2, kernel=kernel)
    X_reduced = pca.fit_transform(X_swiss_roll)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_swiss_roll, cmap=plt.cm.hot)
    plt.show()


#5 Using kPCA and a kernel of your choice, apply Logistic Regression for classification.
#Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best classification accuracy at the end of the pipeline. 
# Print out best parameters found by GridSearchCV.



#: Prepare data for classification
new_y_swiss_roll = [2 if yi >= 10 else 1 if 7 <= yi < 10 else 0 for yi in y_swiss_roll]
X_train_swiss_roll, X_test_swiss_roll, y_train_swiss_roll, y_test_swiss_roll = train_test_split(X_swiss_roll, new_y_swiss_roll, test_size=0.2, random_state=10)

# Create a pipeline with GridSearchCV for hyperparameter tuning
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression(max_iter=10000))
])

# Define a parameter grid for GridSearchCV
param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],
    'kpca__gamma': np.linspace(0.03, 0.05, 10),
    'log_reg__C': [0.1, 1, 10]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_swiss_roll, y_train_swiss_roll)

# Print the best parameters found by GridSearchCV
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)


Plot the results from using GridSearchCV in
#6: Evaluate the best model on the test set and visualize the results
best_model = grid_search.best_estimator_
y_pred_swiss_roll = best_model.predict(X_test_swiss_roll)

accuracy_swiss_roll = accuracy_score(y_test_swiss_roll, y_pred_swiss_roll)
print(f"Accuracy on the test set: {accuracy_swiss_roll:.2f}")

# Visualize the classification results
plt.scatter(X_test_swiss_roll[:, 0], X_test_swiss_roll[:, 1], c=y_pred_swiss_roll, cmap=plt.cm.Spectral)
plt.title("Classification Results")
plt.show()
