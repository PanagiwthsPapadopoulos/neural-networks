import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import accuracy_score


# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Transform data into 1D array
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

pca = PCA(n_components=0.9)  
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# ------------------------------------  KNN CLassification (1)  ------------------------------ #

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
knn.fit(X_train, y_train.ravel())
y_pred = knn.predict(X_test)
acc1 = accuracy_score(y_test, y_pred)
print(f"Accuracy for 1 neighbour: {acc1:.2f}")

# ------------------------------------  KNN CLassification (3)  ------------------------------ #

knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train.ravel())
y_pred = knn.predict(X_test)
acc3 = accuracy_score(y_test, y_pred)
print(f"Accuracy for 3 neighbours: {acc3:.2f}")


# ------------------------------------  NCC CLassification  ------------------------------ #

# Train NearestCentroid classifier
ncc = NearestCentroid(metric='euclidean')
ncc.fit(X_train, y_train.ravel())

# Predict and evaluate
y_pred = ncc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"NCC Accuracy: {accuracy:.2f}")



# ------------------------------------------     KNN Classifier    --------------------------------------------------
# K-Nearest_neighbors classifier

# knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
# knn.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = knn.predict(X_test)

# # Evaluate the model
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Overall Accuracy: {accuracy:.4f}")

# # Compute the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot the confusion matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# ------------------------------------------  KNN Classifier END  -------------------------------------------------