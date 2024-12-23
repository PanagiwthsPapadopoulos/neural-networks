import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError


# Explicitly map 'mse' to the correct metric function
custom_objects = {'mse': MeanSquaredError()}

# Load the model from the .h5 file
model = load_model('mnist_addition_model1.h5', custom_objects=custom_objects)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Function to display inputs and predicted outputs
def display_prediction(input1, input2, output1, output2):
    plt.figure(figsize=(10, 4))
    
    # Input 1
    plt.subplot(1, 4, 1)
    plt.imshow(input1.reshape(28, 28), cmap='gray')
    plt.title("Input 1")
    plt.axis('off')
    
    # Input 2
    plt.subplot(1, 4, 2)
    plt.imshow(input2.reshape(28, 28), cmap='gray')
    plt.title("Input 2")
    plt.axis('off')
    
    # Predicted Output 1
    plt.subplot(1, 4, 3)
    plt.imshow(output1.reshape(28, 28), cmap='gray')
    plt.title("Predicted Output 1")
    plt.axis('off')
    
    # Predicted Output 2
    plt.subplot(1, 4, 4)
    plt.imshow(output2.reshape(28, 28), cmap='gray')
    plt.title("Predicted Output 2")
    plt.axis('off')
    
    plt.show()


# Assume test_input1 and test_input2 are 2D tensors (28, 28)
test_input1 = np.expand_dims(train_images[np.random.randint(0, 500)], axis=0)  # Add batch dimension
test_input1 = np.expand_dims(test_input1, axis=-1)  # Add channel dimension

test_input2 = np.expand_dims(train_images[np.random.randint(0, 500)], axis=0)  # Add batch dimension
test_input2 = np.expand_dims(test_input2, axis=-1)  # Add channel dimension

# Predict outputs
predicted_output1, predicted_output2 = model.predict([test_input1, test_input2])

# Display inputs and predictions
display_prediction(test_input1[0], test_input2[0], predicted_output1[0], predicted_output2[0])