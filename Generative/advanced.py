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
    operator_images, output1_images, output2_images, operators = [], [], [], []

    # Map operators to integers
    operator_map = {'+': 0, '-': 1, '*': 2, '/': 3}  

    # Define target samples per operator
    operator_samples = {'+': 12500, '-': 12500, '*': 12500, '/': 12500}

    # Counters for each operator
    operator_counts = {'+': 0, '-': 0, '*': 0, '/': 0}

    for _ in range(num_samples):
        # Randomly select two digits and an operatord
        digit1 = random.randint(0, 9)
        digit2 = random.randint(0, 9)
        # operator = random.choice(['+', '-', '*', '/'])
        # print(f'Sample Index: {_}')
        # print(f'Division Result: {np.floor(_ / (num_samples/4))}')
        # if(np.floor(_ / (num_samples/4)) == 0): operator = '+'
        # elif(np.floor(_ / (num_samples/4)) == 1): operator = '-'
        # elif(np.floor(_ / (num_samples/4)) == 2): operator = '*'
        # elif(np.floor(_ / (num_samples/4)) == 3): operator = '/'



        # Generate dataset with balanced operators

        operator = random.choice(['+', '-', '*', '/'])
        if(operator_counts['+'] < operator_samples['+']):
            operator = '+'
            operator_counts[operator] += 1
        elif(operator_counts['-'] < operator_samples['-']):
            operator = '-'
            operator_counts[operator] += 1
        elif(operator_counts['*'] < operator_samples['*']):
            operator = '*'
            operator_counts[operator] += 1
        elif(operator_counts['/'] < operator_samples['/']):
            operator = '/'
            operator_counts[operator] += 1

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
        operators.append(operator_map[operator])

    # Convert to NumPy arrays and reshape for Keras compatibility
    return (
        np.array(input1_images).reshape(-1, 28, 28, 1),
        np.array(input2_images).reshape(-1, 28, 28, 1),
        np.array(operator_images).reshape(-1, 28, 28, 1),
        np.array(output1_images).reshape(-1, 28, 28, 1),
        np.array(output2_images).reshape(-1, 28, 28, 1),
        np.array(operators)
    )

# Function to display inputs and predicted outputs
def display_prediction(input1, input2, operator_input, output1, output2, operator_symbol):
    # size = (14, 14)
    size = (28, 28)
    output_size = (28, 28)
    plt.figure(figsize=(12, 5))
    
    # Input 1
    plt.subplot(1, 6, 1)
    plt.imshow(input1.reshape(size), cmap='gray')
    plt.title("Input 1")
    plt.axis('off')

    # Op
    plt.subplot(1, 6, 2)
    plt.imshow(operator_input.reshape(size), cmap='gray')
    plt.title("Input 2")
    plt.axis('off')
    
    # Input 2
    plt.subplot(1, 6, 3)
    plt.imshow(input2.reshape(size), cmap='gray')
    plt.title("Input 3")
    plt.axis('off')
    
    # Predicted Output 1
    plt.subplot(1, 6, 4)
    plt.imshow(output1.reshape(output_size), cmap='gray')
    plt.title("Predicted Output 1")
    plt.axis('off')
    
    # Predicted Output 2
    plt.subplot(1, 6, 5)
    plt.imshow(output2.reshape(output_size), cmap='gray')
    plt.title("Predicted Output 2")
    plt.axis('off')

    # Actual Operator Symbol
    plt.subplot(1, 6, 6)
    plt.text(0.5, 0.5, operator_symbol, fontsize=20, ha='center', va='center')
    plt.title("Actual Operator")
    plt.axis('off')

    
    plt.show()


# Generate dataset
X1, X2, Ops, Y1, Y2, Op_labels = generate_dataset_with_operator_images(num_samples=50000)

# ind=0
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])
# ind=18749
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])
# ind=18750
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])
# ind=37499
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])
# ind=37500
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])
# ind = 43749
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])
# ind = 43750
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])
# ind=62499
# display_prediction(X1[ind], X2[ind], Ops[ind], Y1[ind], Y2[ind], Op_labels[ind])

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
def build_decoder(latent, dense_size, output_shape, transposed_conv_layers, filters, kernel_size, strides, activation, name):
    x = Dense(dense_size, activation=activation)(latent)
    x = Reshape(output_shape)(x)
    for i in range(transposed_conv_layers):
        x = Conv2DTranspose(filters[i], (kernel_size[i], kernel_size[i]), strides=(strides[i], strides[i]), activation=activation, padding='same')(x)
    output = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', name=name)(x)
    return output

def custom_loss(y_true, y_pred):
    output1_true, output2_true, operator_true = y_true
    output1_pred, output2_pred, operator_pred = y_pred

    # Image losses
    image_loss1 = tf.reduce_mean(tf.square(output1_true - output1_pred))
    image_loss2 = tf.reduce_mean(tf.square(output2_true - output2_pred))

    # Operator loss
    operator_loss = tf.keras.losses.CategoricalCrossentropy()(operator_true, operator_pred)

    # Operator-specific weighting
    operator_weights = tf.reduce_sum(operator_true * tf.constant([1.0, 1.2, 1.5, 2.0]), axis=1)
    weighted_operator_loss = tf.reduce_mean(operator_loss * operator_weights)

    # Combine losses
    total_loss = image_loss1 + image_loss2 + 5.0 * weighted_operator_loss
    return total_loss


start_time = time.time()

MULTIPLE_TRAINING = True

if MULTIPLE_TRAINING:
    # Define fully parameterized configurations
    configurations = [
       
        {
            'input_shape': (28, 28, 1),
            'conv_layers': 4,  # Increased number of convolutional layers
            'filters': [32, 64, 128, 256],  # Added an extra layer with higher filter count
            'kernel_size': [3, 3, 3, 3],
            'pool_size': [2, 2, 2, 2],
            'activation': 'relu',
            'latent_size': 128,  # Increased latent size for a richer latent space
            'dense_size': 7 * 7 * 256,  # Recalculated dense size (based on the final output dimensions of the encoder)
            'output_shape': (7, 7, 256),  # Adjusted output shape to match the dense size
            'transposed_conv_layers': 3,  # Three transposed convolutional layers
            'transposed_filters': [256, 128, 64],
            'transposed_kernel_size': [3, 3, 3],
            'transposed_strides': [2, 2, 1],  # Adjusted strides to ensure correct output size
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 15,  # Reduced to match the updated requirements
            
        },
    
        
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
        # op_weight = config['operator_weight']

        # Build shared CNNs for inputs
        input1, features1 = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)
        input2, features2 = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)
        operator_input, operator_features = build_shared_cnn(input_shape, conv_layers, filters, kernel_size, pool_size, activation)

        # # Define the weight for the operator
        # operator_weight = tf.constant(op_weight, dtype=tf.float32)

        # # Expand dimensions to match the shape of operator features
        # operator_weight_tensor = tf.reshape(operator_weight, (1, 1, 1, 1))

        # # Add weight to operator features
        # # operator_weight = 2.0  # You can tune this value
        # weighted_operator_features = Multiply()([operator_features, operator_weight_tensor])

        # Merge features
        merged_features = concatenate([features1, operator_features, features2])

        # Latent space
        latent = Dense(latent_size, activation=activation)(merged_features)

        # Decoders for two outputs
        output1 = build_decoder(latent, dense_size, output_shape, transposed_conv_layers, transposed_filters, transposed_kernel_size, transposed_strides, activation, 'output1')
        output2 = build_decoder(latent, dense_size, output_shape, transposed_conv_layers, transposed_filters, transposed_kernel_size, transposed_strides, activation, 'output2')

        # New output for operator classification
        operator_output = Dense(4, activation='softmax', name='operator_output')(latent)

        # Define the model
        model = Model(inputs=[input1, input2, operator_input], outputs=[output1, output2, operator_output])

        # Compile the model with weighted losses
        losses = {
            'output1': 'mse',  # Loss for the first digit output
            'output2': 'mse',  # Loss for the second digit output
            'operator_output': 'categorical_crossentropy'  # Loss for operator prediction
        }


        # output_weights = {'output1': 1.0, 'output2': 1.0, 'operator': 5.0}  # Adjust weights as needed
        # loss_function = custom_loss(output_weights)


        # Model Summary
        model.summary()

        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate), 
            loss={
                'output1': 'mse',                  # Loss for Image Output 1
                'output2': 'mse',                  # Loss for Image Output 2
                'operator_output': 'categorical_crossentropy',  # Loss for Operator
            },
            loss_weights={
                'output1': 1.0,                    # Weight for Image Output 1
                'output2': 1.0,                    # Weight for Image Output 2
                'operator_output': 3.0,            # Weight for Operator
            },
            metrics={
                'output1': ['mse'],
                'output2': ['mse'],
                'operator_output': ['accuracy', 'categorical_crossentropy']
            }
        )

        Op_labels_categorical = to_categorical(Op_labels, num_classes=4)

        # Train the model
        history = model.fit(
            [X1, X2, Ops],              # Input data: two digit images and one operator image
            [Y1, Y2, Op_labels_categorical],                   # Target data: two output images
            batch_size=batch_size,      # Number of samples per batch
            epochs=epochs,              # Number of training epochs
            validation_split=0.2,       # Use 20% of data for validation
            verbose=1,                   # Print training progress
            shuffle = True
        )

        # Save the model and history
        # model_name = f'ModelResults/model__convlayers{conv_layers}_latent{latent_size}_transposed_conv_layers{transposed_conv_layers}_learningrate{learning_rate}_batch_size{batch_size}_epochs{epochs}_size100k.h5'
        model_name = f'ModelResults/weighted_loss_3.h5'
        model.save(model_name, save_format='tf')
        print(f"Model saved as {model_name}")

        # history_path = f'HistoryResults/history_convlayers{conv_layers}_latent{latent_size}_transposed_conv_layers{transposed_conv_layers}_learningrate{learning_rate}_batch_size{batch_size}_epochs{epochs}_size100k.npy'
        history_path = f'HistoryResults/weighted_loss_3.npy'
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
    epochs = 15

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

stop_time = time.time()
print(f'Total time: {stop_time-start_time} seconds.')
