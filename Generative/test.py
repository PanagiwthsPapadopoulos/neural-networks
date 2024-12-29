# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import load_model
# from tensorflow.keras.losses import MeanSquaredError


# # Explicitly map 'mse' to the correct metric function
# custom_objects = {'mse': MeanSquaredError()}

# # Load the model from the .h5 file
# model = load_model('mnist_addition_model1-100epochs.h5', custom_objects=custom_objects)

# # Load MNIST dataset
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# # Function to display inputs and predicted outputs
# def display_prediction(input1, input2, output1, output2):
#     plt.figure(figsize=(10, 4))
    
#     # Input 1
#     plt.subplot(1, 4, 1)
#     plt.imshow(input1.reshape(28, 28), cmap='gray')
#     plt.title("Input 1")
#     plt.axis('off')
    
#     # Input 2
#     plt.subplot(1, 4, 2)
#     plt.imshow(input2.reshape(28, 28), cmap='gray')
#     plt.title("Input 2")
#     plt.axis('off')
    
#     # Predicted Output 1
#     plt.subplot(1, 4, 3)
#     plt.imshow(output1.reshape(28, 28), cmap='gray')
#     plt.title("Predicted Output 1")
#     plt.axis('off')
    
#     # Predicted Output 2
#     plt.subplot(1, 4, 4)
#     plt.imshow(output2.reshape(28, 28), cmap='gray')
#     plt.title("Predicted Output 2")
#     plt.axis('off')
    
#     plt.show()


# # Assume test_input1 and test_input2 are 2D tensors (28, 28)
# test_input1 = np.expand_dims(train_images[np.random.randint(0, 500)], axis=0)  # Add batch dimension
# test_input1 = np.expand_dims(test_input1, axis=-1)  # Add channel dimension

# test_input2 = np.expand_dims(train_images[np.random.randint(0, 500)], axis=0)  # Add batch dimension
# test_input2 = np.expand_dims(test_input2, axis=-1)  # Add channel dimension

# # Predict outputs
# predicted_output1, predicted_output2 = model.predict([test_input1, test_input2])

# # Display inputs and predictions
# display_prediction(test_input1[0], test_input2[0], predicted_output1[0], predicted_output2[0])




import random
import time
from matplotlib import pyplot as plt
import deeplake
import numpy as np
from tensorflow.keras.datasets import mnist
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, Conv2DTranspose, Multiply
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy

Op_labels = [0, 1, 3]
Op_labels_categorical = to_categorical(Op_labels, num_classes=4)
print(Op_labels_categorical)