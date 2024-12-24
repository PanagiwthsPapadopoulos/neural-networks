import random
from matplotlib import pyplot as plt
import deeplake
import numpy as np
from tensorflow.keras.datasets import mnist
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.layers import Dropout



ds = deeplake.load("hub://activeloop/hasy-test")

# 112 minus 
# 113 plus
# 131 multiplication
# 138 division

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

# ------------------------------------------------------- Dataset -------------------------------------------------------------- #

# Step 1: Load and Normalize MNIST Dataset
(train_images, train_labels), (_, _) = mnist.load_data()
train_images = train_images / 255.0  # Normalize pixel values to [0, 1]

# Function to Retrieve a Random MNIST Image for a Given Digit
def get_mnist_image(digit):
    indices = np.where(train_labels == digit)[0]
    selected_index = random.choice(indices)
    return train_images[selected_index]

def get_op_image(operator):
    
    match (operator):
        case '+': image = random.choice(addition_images)
        case '-': image = random.choice(subtraction_images)
        case '*': image = random.choice(multiplication_images)
        case '/': image = random.choice(division_images)
    return image
    

def generate_dataset_with_operator_images(num_samples=50000, operator_images=None):
    input1_images, input2_images = [], []
    operator_images, output1_images, output2_images = [], [], []

    for _ in range(num_samples):
        # Randomly select two digits and an operator
        digit1 = random.randint(0, 9)
        digit2 = random.randint(0, 9)
        operator = random.choice(['+', '-', '*', '/'])

        # Compute the result of the operation
        if operator == '+':
            total = digit1 + digit2
        elif operator == '-':
            total = digit1 - digit2
        elif operator == '*':
            total = digit1 * digit2
        elif operator == '/':
            total = digit1 // digit2 if digit2 != 0 else 0  # Avoid division by zero

        # Split the result into two digits
        total_str = f"{abs(total):02}"  # Ensure two digits
        output1 = int(total_str[0])  # First digit
        output2 = int(total_str[1])  # Second digit

        # Retrieve MNIST images
        input1 = get_mnist_image(digit1)
        input2 = get_mnist_image(digit2)
        operator_input = get_op_image(operator)
        output1_img = get_mnist_image(output1)
        output2_img = get_mnist_image(output2)
        
        # Plot the first operator image
        # if _ == 0:  # Plot only the first operator image
        #     plt.imshow(operator_input.reshape(28, 28), cmap='gray')
        #     plt.title(f"Operator: {operator}")
        #     plt.axis('off')
        #     plt.show()


        # Append to dataset
        input1_images.append(input1)
        input2_images.append(input2)
        operator_images.append(operator_input)  # Append operator image
        output1_images.append(output1_img)
        output2_images.append(output2_img)

    # Convert to NumPy arrays and reshape for Keras compatibility
    return (
        np.array(input1_images).reshape(-1, 28, 28, 1),
        np.array(input2_images).reshape(-1, 28, 28, 1),
        np.array(operator_images).reshape(-1, 28, 28, 1),
        np.array(output1_images).reshape(-1, 28, 28, 1),
        np.array(output2_images).reshape(-1, 28, 28, 1),
    )

# Function to display inputs and predicted outputs
def display_prediction(input1, input2, operation_input, output1, output2):
    plt.figure(figsize=(10, 5))
    
    # Input 1
    plt.subplot(1, 5, 1)
    plt.imshow(input1.reshape(28, 28), cmap='gray')
    plt.title("Input 1")
    plt.axis('off')
    
    # Operation
    plt.subplot(1, 5, 2)
    plt.imshow(operation_input.reshape(28, 28), cmap='gray')
    plt.title("Operation")
    plt.axis('off')

    # Input 2
    plt.subplot(1, 5, 3)
    plt.imshow(input2.reshape(28, 28), cmap='gray')
    plt.title("Input 2")
    plt.axis('off')
    
    # Predicted Output 1
    plt.subplot(1, 5, 4)
    plt.imshow(output1.reshape(28, 28), cmap='gray')
    plt.title("Predicted Output 1")
    plt.axis('off')
    
    # Predicted Output 2
    plt.subplot(1, 5, 5)
    plt.imshow(output2.reshape(28, 28), cmap='gray')
    plt.title("Predicted Output 2")
    plt.axis('off')
    
    plt.show()


# Generate dataset
X1, X2, Ops, Y1, Y2 = generate_dataset_with_operator_images(num_samples=50000)


# size=(14, 14)
# images = X1
# X1 = np.array([img_to_array(array_to_img(im, scale=False).resize(size)) for im in images])
# X1 = X1.reshape((-1, size[0], size[1], 1))

# images = X2
# X2 = np.array([img_to_array(array_to_img(im, scale=False).resize(size)) for im in images])
# X2 = X2.reshape((-1, size[0], size[1], 1))

# images = Ops
# Ops = np.array([img_to_array(array_to_img(im, scale=False).resize(size)) for im in images])
# Ops = Ops.reshape((-1, size[0], size[1], 1))

# images = Y1
# resized_images = np.array([img_to_array(array_to_img(im, scale=False).resize(size)) for im in images])
# resized_images.reshape((-1, size[0], size[1], 1))

# images = Y2
# resized_images = np.array([img_to_array(array_to_img(im, scale=False).resize(size)) for im in images])
# resized_images.reshape((-1, size[0], size[1], 1))


# Parameterized shared CNN for feature extraction
def build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation):
    input_layer = Input(shape=input_shape)
    x = input_layer
    for i in range(conv_layers):
        x = Conv2D(filters[i], (kernel_size[i], kernel_size[i]), activation=activation, padding='same')(x)
        x = MaxPooling2D((pool_size[i], pool_size[i]))(x)
    x = Flatten()(x)
    return input_layer, x

# Parameterized decoder
def build_decoder(latent, dense_size, output_shape, transposed_conv_layers, filters, kernel_size, strides, activation):
    x = Dense(dense_size, activation=activation)(latent)
    x = Reshape(output_shape)(x)
    for i in range(transposed_conv_layers):
        x = Conv2DTranspose(filters[i], (kernel_size[i], kernel_size[i]), strides=(strides[i], strides[i]), activation=activation, padding='same')(x)
    output = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    return output




MULTIPLE_TRAINING = True

if MULTIPLE_TRAINING:
    # Define fully parameterized configurations
    configurations = [
        # {
        #     'input_shape': (28, 28, 1),
        #     'conv_layers': 3,
        #     'filters': [32, 64, 128],
        #     'kernel_size': [3, 3, 3],
        #     'pool_size': [2, 2, 2],
        #     'activation': 'relu',
        #     'latent_size': 256,
        #     'dense_size': 7 * 7 * 128,
        #     'output_shape': (7, 7, 128),
        #     'transposed_conv_layers': 3,
        #     'transposed_filters': [128, 64, 32],
        #     'transposed_kernel_size': [3, 3, 3],
        #     'transposed_strides': [2, 2, 1],
        #     'learning_rate': 0.001,
        #     'batch_size': 32,
        #     'epochs': 30
        # },
        # {
        #     'input_shape': (28, 28, 1),
        #     'conv_layers': 3,
        #     'filters': [32, 64, 128],
        #     'kernel_size': [3, 3, 3],
        #     'pool_size': [2, 2, 2],
        #     'activation': 'relu',
        #     'latent_size': 256,
        #     'dense_size': 7 * 7 * 128,
        #     'output_shape': (7, 7, 128),
        #     'transposed_conv_layers': 3,
        #     'transposed_filters': [128, 64, 32],
        #     'transposed_kernel_size': [3, 3, 3],
        #     'transposed_strides': [2, 2, 1],
        #     'learning_rate': 0.01,
        #     'batch_size': 32,
        #     'epochs': 30
        # },
        # {
        #     'input_shape': (28, 28, 1),
        #     'conv_layers': 3,
        #     'filters': [32, 64, 128],
        #     'kernel_size': [3, 3, 3],
        #     'pool_size': [2, 2, 2],
        #     'activation': 'relu',
        #     'latent_size': 256,
        #     'dense_size': 7 * 7 * 128,
        #     'output_shape': (7, 7, 128),
        #     'transposed_conv_layers': 3,
        #     'transposed_filters': [128, 64, 32],
        #     'transposed_kernel_size': [3, 3, 3],
        #     'transposed_strides': [2, 2, 1],
        #     'learning_rate': 0.1,
        #     'batch_size': 32,
        #     'epochs': 30
        # },
        {
            'input_shape': (28, 28, 1),
            'conv_layers': 4,
            'filters': [32, 64, 128, 256],
            'kernel_size': [3, 3, 3, 3],
            'pool_size': [2, 2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 3,
            'transposed_filters': [128, 64, 32],
            'transposed_kernel_size': [3, 3, 3],
            'transposed_strides': [2, 2, 1],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 4,
            'filters': [32, 64, 128, 256],
            'kernel_size': [3, 3, 3, 3],
            'pool_size': [2, 2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 3,
            'transposed_filters': [128, 64, 32],
            'transposed_kernel_size': [3, 3, 3],
            'transposed_strides': [2, 2, 1],
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 4,
            'filters': [32, 64, 128, 256],
            'kernel_size': [3, 3, 3, 3],
            'pool_size': [2, 2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 3,
            'transposed_filters': [128, 64, 32],
            'transposed_kernel_size': [3, 3, 3],
            'transposed_strides': [2, 2, 1],
            'learning_rate': 0.1,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 3,
            'filters': [32, 64, 128],
            'kernel_size': [3, 3, 3],
            'pool_size': [2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 4,
            'transposed_filters': [256, 128, 64, 32],
            'transposed_kernel_size': [3, 3, 3, 3],
            'transposed_strides': [2, 2, 2, 1],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 3,
            'filters': [32, 64, 128],
            'kernel_size': [3, 3, 3],
            'pool_size': [2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 4,
            'transposed_filters': [256, 128, 64, 32],
            'transposed_kernel_size': [3, 3, 3, 3],
            'transposed_strides': [2, 2, 2, 1],
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 3,
            'filters': [32, 64, 128],
            'kernel_size': [3, 3, 3],
            'pool_size': [2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 4,
            'transposed_filters': [256, 128, 64, 32],
            'transposed_kernel_size': [3, 3, 3, 3],
            'transposed_strides': [2, 2, 2, 1],
            'learning_rate': 0.1,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 4,
            'filters': [32, 64, 128, 256],
            'kernel_size': [3, 3, 3, 3],
            'pool_size': [2, 2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 4,
            'transposed_filters': [256, 128, 64, 32],
            'transposed_kernel_size': [3, 3, 3, 3],
            'transposed_strides': [2, 2, 2, 1],
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 4,
            'filters': [32, 64, 128, 256],
            'kernel_size': [3, 3, 3, 3],
            'pool_size': [2, 2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 4,
            'transposed_filters': [256, 128, 64, 32],
            'transposed_kernel_size': [3, 3, 3, 3],
            'transposed_strides': [2, 2, 2, 1],
            'learning_rate': 0.01,
            'batch_size': 32,
            'epochs': 30
        },{
            'input_shape': (28, 28, 1),
            'conv_layers': 4,
            'filters': [32, 64, 128, 256],
            'kernel_size': [3, 3, 3, 3],
            'pool_size': [2, 2, 2, 2],
            'activation': 'relu',
            'latent_size': 256,
            'dense_size': 7 * 7 * 128,
            'output_shape': (7, 7, 128),
            'transposed_conv_layers': 4,
            'transposed_filters': [256, 128, 64, 32],
            'transposed_kernel_size': [3, 3, 3, 3],
            'transposed_strides': [2, 2, 2, 1],
            'learning_rate': 0.1,
            'batch_size': 32,
            'epochs': 30
        }
        # Add more configurations as needed
    ]

    # Model training loop
    for i, config in enumerate(configurations):
        print(f"Training model {i+1} with configuration: {config}")

        # Extract parameters for the current configuration
        input_shape = config['input_shape']
        conv_layers = config['conv_layers']
        filters = config['filters']
        kernel_size = config['kernel_size']
        pool_size = config['pool_size']
        activation = config['activation']
        latent_size = config['latent_size']
        dense_size = config['dense_size']
        output_shape = config['output_shape']
        transposed_conv_layers = config['transposed_conv_layers']
        transposed_filters = config['transposed_filters']
        transposed_kernel_size = config['transposed_kernel_size']
        transposed_strides = config['transposed_strides']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        epochs = config['epochs']

        # Build shared CNNs for inputs
        input1, features1 = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)
        input2, features2 = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)
        operator_input, operator_features = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)

        # Merge features
        merged_features = concatenate([features1, features2, operator_features])

        # Latent space
        latent = Dense(latent_size, activation=activation)(merged_features)

        # Decoders for two outputs
        output1 = build_decoder(latent, dense_size, output_shape, transposed_conv_layers, transposed_filters, transposed_kernel_size, transposed_strides, activation)
        output2 = build_decoder(latent, dense_size, output_shape, transposed_conv_layers, transposed_filters, transposed_kernel_size, transposed_strides, activation)

        # Define the model
        model = Model(inputs=[input1, input2, operator_input], outputs=[output1, output2])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse', 'mse'])

        # Train the model
        history = model.fit(
            [X1, X2, Ops],              # Input data: two digit images and one operator image
            [Y1, Y2],                   # Target data: two output images
            batch_size=batch_size,      # Number of samples per batch
            epochs=epochs,              # Number of training epochs
            validation_split=0.2,       # Use 20% of data for validation
            verbose=1                   # Print training progress
        )

        # Save the model and history
        model_name = f'ModelResults/model__convlayers{conv_layers}_latent{latent_size}_transposed_conv_layers{transposed_conv_layers}_learningrate{learning_rate}_batch_size{batch_size}_epochs{epochs}.h5'
        model.save(model_name)
        print(f"Model saved as {model_name}")

        history_path = f'HistoryResults/history_convlayers{conv_layers}_latent{latent_size}_transposed_conv_layers{transposed_conv_layers}_learningrate{learning_rate}_batch_size{batch_size}_epochs{epochs}.npy'
        np.save(history_path, history.history)
        print(f"Training history saved as {history_path}")

else:

    # Parameters
    input_shape = (28, 28, 1)
    conv_layers = 3
    filters = [32, 64, 128]
    kernel_size = [3, 3, 3]
    pool_size = [2, 2, 2]
    activation = 'relu'

    latent_size = 256
    dense_size = 7 * 7 * 128
    output_shape = (7, 7, 128)

    transposed_conv_layers = 3
    transposed_filters = [128, 64, 32]
    transposed_kernel_size = [3, 3, 3]
    transposed_strides = [2, 2, 1]

    learning_rate = 0.001
    batch_size = 32
    epochs = 20

    # Build shared CNNs for inputs
    input1, features1 = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)
    input2, features2 = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)
    operator_input, operator_features = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)

    # Merge features
    merged_features = concatenate([features1, features2, operator_features])

    # Latent space
    latent = Dense(latent_size, activation=activation)(merged_features)

    # Decoders for two outputs
    output1 = build_decoder(latent, dense_size, output_shape, transposed_conv_layers, transposed_filters, transposed_kernel_size, transposed_strides, activation)
    output2 = build_decoder(latent, dense_size, output_shape, transposed_conv_layers, transposed_filters, transposed_kernel_size, transposed_strides, activation)

    # Define the model
    model = Model(inputs=[input1, input2, operator_input], outputs=[output1, output2])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mse', 'mse'])

    # Display the model summary
    model.summary()

    # Train the model
    history = model.fit(
        [X1, X2, Ops],              # Input data: two digit images and one operator image
        [Y1, Y2],                   # Target data: two output images
        batch_size=batch_size,      # Number of samples per batch
        epochs=epochs,              # Number of training epochs
        validation_split=0.2,       # Use 20% of data for validation
        verbose=1                   # Print training progress
    )

    # Save the model
    model.save('parameterized_model.h5')


    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs. Validation Loss')
    plt.legend()
    plt.show()


