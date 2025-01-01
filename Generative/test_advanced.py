import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import cv2
import deeplake
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
import h5py
from tensorflow.keras.models import load_model

with h5py.File('ModelResults/weighted_loss_10.h5', 'r') as f:
    print(list(f.keys())) 

def custom_loss(alpha=1.0, beta=1.0):
    mse_loss = MeanSquaredError()
    cce_loss = CategoricalCrossentropy()

    def loss(y_true, y_pred):
        # Split y_true and y_pred
        output1_true = y_true[0]
        output2_true = y_true[1]
        operator_true = y_true[2]  # Assuming one-hot encoded operator
        
        output1_pred = y_pred[0]
        output2_pred = y_pred[1]
        operator_pred = y_pred[2]

        # Calculate MSE for image outputs
        output_image_loss = mse_loss(output1_true, output1_pred) + mse_loss(output2_true, output2_pred)

        # Calculate categorical cross-entropy for operator
        operator_loss = cce_loss(operator_true, operator_pred)

        # Combine the losses
        total_loss = alpha * output_image_loss + beta * operator_loss
        return total_loss

    return loss


# Explicitly map 'mse' to the correct metric function
custom_objects = {
    'mse': MeanSquaredError(),
}

# Load the model from the .h5 file
model = load_model('ModelResults/weighted_loss_10.h5', custom_objects=custom_objects)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Load and preprocess symbol dataset
ds = deeplake.load("hub://activeloop/hasy-test")

ADDITION = 113
SUBTRACTION = 112
MULTIPLICATION = 131
DIVISION = 138

addition_images = []
subtraction_images = []
multiplication_images = []
division_images = []

def resizeImage(image):
    # Resize to 28x28
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    # Convert to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    # Normalize to [0, 1] and add a channel dimension
    normalized_image = grayscale_image / 255.0
    # Invert image
    inverted_image = 1 - normalized_image
    return inverted_image.reshape(28, 28, 1)


# Iterate through the dataset using enumerate for more efficiency
for i, sample in enumerate(ds):
    label = sample['latex'].numpy()  

    if label in [ADDITION, SUBTRACTION, MULTIPLICATION, DIVISION]:
        image = resizeImage(sample['images'].numpy())
        if label == ADDITION:  
            addition_images.append(image)  
        elif label == SUBTRACTION:
            subtraction_images.append(image)
        elif label == MULTIPLICATION:
            multiplication_images.append(image)
        elif label == DIVISION:
            division_images.append(image)

# Function to display inputs and predicted outputs
def display_prediction(input1, input2, operator_input, output1, output2, operator_symbol, predicted_operator_symbol):
    # size = (14, 14)
    size = (28, 28)
    output_size = (28, 28)
    plt.figure(figsize=(12, 5))
    
    # Input 1
    plt.subplot(1, 7, 1)
    plt.imshow(input1.reshape(size), cmap='gray')
    plt.title("Input 1")
    plt.axis('off')

    # Op
    plt.subplot(1, 7, 2)
    plt.imshow(operator_input.reshape(size), cmap='gray')
    plt.title("Input 2")
    plt.axis('off')
    
    # Input 2
    plt.subplot(1, 7, 3)
    plt.imshow(input2.reshape(size), cmap='gray')
    plt.title("Input 3")
    plt.axis('off')
    
    # Predicted Output 1
    plt.subplot(1, 7, 4)
    plt.imshow(output1.reshape(output_size), cmap='gray')
    plt.title("Predicted Output 1")
    plt.axis('off')
    
    # Predicted Output 2
    plt.subplot(1, 7, 5)
    plt.imshow(output2.reshape(output_size), cmap='gray')
    plt.title("Predicted Output 2")
    plt.axis('off')

    # Actual Operator Symbol
    plt.subplot(1, 7, 6)
    plt.text(0.5, 0.5, operator_symbol, fontsize=20, ha='center', va='center')
    plt.title("Actual Operator")
    plt.axis('off')

    # Predicted Operator Symbol
    plt.subplot(1, 7, 7)
    plt.text(0.5, 0.5, predicted_operator_symbol, fontsize=20, ha='center', va='center')
    plt.title("Predicted Operator")
    plt.axis('off')
    
    plt.show()

for i in range(0,30):

    

    operator = random.choice(['+', '-', '*', '/'])
    match (operator):
        case '+': image = random.choice(addition_images)
        case '-': image = random.choice(subtraction_images)
        case '*': image = random.choice(multiplication_images)
        case '/': image = random.choice(division_images)

    image1 = train_images[np.random.randint(0, 500)]
    image2 = train_images[np.random.randint(0, 500)]


    # resized_image1 = cv2.resize(image1, (14, 14), interpolation=cv2.INTER_AREA)
    # # If you need to use the image with a channel dimension for neural networks:
    # image1 = resized_image1.reshape(14, 14, 1)


    # resized_image2 = cv2.resize(image2, (14, 14), interpolation=cv2.INTER_AREA)
    # # If you need to use the image with a channel dimension for neural networks:
    # image2 = resized_image2.reshape(14, 14, 1)


    # resized_image = cv2.resize(image, (14, 14), interpolation=cv2.INTER_AREA)
    # # If you need to use the image with a channel dimension for neural networks:
    # image = resized_image.reshape(14, 14, 1)



   

    # Assume test_input1 and test_input2 are 2D tensors (28, 28)
    test_input1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    # test_input1 = np.expand_dims(image1, axis=-1)  # Add channel dimension

    test_input2 = np.expand_dims(image2, axis=0)  # Add batch dimension
    # test_input2 = np.expand_dims(image2, axis=-1)  # Add channel dimension

    

    operator_image = np.expand_dims(image, axis=0)
    # image = np.expand_dims(image, axis = -1)

    print(test_input1.shape)
    print(test_input2.shape)
    print(operator_image.shape)

    # Predict outputs
    predicted_output1, predicted_output2, predicted_operator = model.predict(
        [test_input1, test_input2, operator_image]
    )

    # Map the predicted operator index to its corresponding symbol
    operator_mapping = {0: '+', 1: '-', 2: '*', 3: '/'}
    predicted_operator_symbol = operator_mapping[np.argmax(predicted_operator)]

    # Display inputs and predictions
    display_prediction(test_input1[0], test_input2[0], operator_image, predicted_output1[0], predicted_output2[0], operator, predicted_operator_symbol)