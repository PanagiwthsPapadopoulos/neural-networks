import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import cv2
import deeplake
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img




# Explicitly map 'mse' to the correct metric function
custom_objects = {'mse': MeanSquaredError()}

# Load the model from the .h5 file
model = load_model('arithmetic_model_100kdataset_150iter.h5', custom_objects=custom_objects)

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
def display_prediction(input1, input2, operator_input, output1, output2):
    # size = (14, 14)
    size = (28, 28)
    output_size = (28, 28)
    plt.figure(figsize=(10, 5))
    
    # Input 1
    plt.subplot(1, 5, 1)
    plt.imshow(input1.reshape(size), cmap='gray')
    plt.title("Input 1")
    plt.axis('off')

    # Op
    plt.subplot(1, 5, 2)
    plt.imshow(operator_input.reshape(size), cmap='gray')
    plt.title("Input 2")
    plt.axis('off')
    
    # Input 2
    plt.subplot(1, 5, 3)
    plt.imshow(input2.reshape(size), cmap='gray')
    plt.title("Input 2")
    plt.axis('off')
    
    # Predicted Output 1
    plt.subplot(1, 5, 4)
    plt.imshow(output1.reshape(output_size), cmap='gray')
    plt.title("Predicted Output 1")
    plt.axis('off')
    
    # Predicted Output 2
    plt.subplot(1, 5, 5)
    plt.imshow(output2.reshape(output_size), cmap='gray')
    plt.title("Predicted Output 2")
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

    

    image = np.expand_dims(image, axis=0)
    # image = np.expand_dims(image, axis = -1)

    print(test_input1.shape)
    print(test_input2.shape)
    print(image.shape)

    # Predict outputs
    predicted_output1, predicted_output2 = model.predict([test_input1, test_input2, image])

    # Display inputs and predictions
    display_prediction(test_input1[0], test_input2[0], image, predicted_output1[0], predicted_output2[0])