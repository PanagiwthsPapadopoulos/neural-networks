import matplotlib.pyplot as plt
import numpy as np
import joblib
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

## Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Keep a copy of the original test images for visualization
original_test_images = X_test.copy()

# Flatten the images into 1D vectors (32x32x3 -> 3072)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Standardize features to have mean 0 and variance 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduce dimensions using PCA (keeping 90% of the variance)
pca = PCA(n_components=0.90)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Load the trained SVM model
svm_model = joblib.load("svm_model.pkl")

# Choose some random images from the test set
num_images = 10  # Number of images to display
random_indices = np.random.choice(X_test.shape[0], num_images, replace=False)
sample_images = original_test_images[random_indices]
sample_labels = y_test[random_indices]
sample_transformed = X_test[random_indices]

# Predict the classes using the SVM
predicted_classes = svm_model.predict(sample_transformed)

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Plot the results
plt.figure(figsize=(15, 8))
for i in range(num_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(sample_images[i])
    plt.title(
        f"Predicted: {class_names[predicted_classes[i]]}\nActual: {class_names[sample_labels[i][0]]}"
    )
    plt.axis("off")
plt.tight_layout()
plt.show()