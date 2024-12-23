from matplotlib import pyplot as plt
import numpy as np
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model

# Step 1: Load and Normalize MNIST Dataset
(train_images, train_labels), (_, _) = mnist.load_data()
train_images = train_images / 255.0  # Normalize pixel values to [0, 1]

# Function to Retrieve a Random MNIST Image for a Given Digit
def get_mnist_image(digit):
    indices = np.where(train_labels == digit)[0]
    selected_index = random.choice(indices)
    return train_images[selected_index]

# Step 2: Generate Dataset with Input Pairs and Ground Truth
def generate_dataset(num_samples=50000):
    input1_images, input2_images = [], []
    output1_images, output2_images = [], []

    for _ in range(num_samples):
        # Randomly select two digits
        digit1 = random.randint(0, 9)
        digit2 = random.randint(0, 9)

        # Retrieve MNIST images for the input digits
        input1 = get_mnist_image(digit1)
        input2 = get_mnist_image(digit2)

        # Compute the sum and split into digits
        total = digit1 + digit2
        total_str = f"{total:02}"  # Ensure two digits (e.g., "08" for single-digit sums)

        # Retrieve MNIST images for the output digits (ground truth)
        output1 = get_mnist_image(int(total_str[0]))  # First digit of the sum
        output2 = get_mnist_image(int(total_str[1]))  # Second digit of the sum

        # Append to dataset
        input1_images.append(input1)
        input2_images.append(input2)
        output1_images.append(output1)
        output2_images.append(output2)

    # Convert to NumPy arrays and reshape for Keras compatibility
    return (
        np.array(input1_images).reshape(-1, 28, 28, 1),
        np.array(input2_images).reshape(-1, 28, 28, 1),
        np.array(output1_images).reshape(-1, 28, 28, 1),
        np.array(output2_images).reshape(-1, 28, 28, 1),
    )

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

# Generate the Dataset
num_samples = 50000  # Number of training examples
X1, X2, Y1, Y2 = generate_dataset(num_samples)

print("Dataset Prepared:")
print(f"Input 1 Shape: {X1.shape}")
print(f"Input 2 Shape: {X2.shape}")
print(f"Output 1 Shape: {Y1.shape}")
print(f"Output 2 Shape: {Y2.shape}")

# # Display the images
# plt.figure(figsize=(10, 4))

# # Input 1
# plt.subplot(1, 4, 1)
# plt.imshow(X1[0], cmap='gray')
# plt.title(f"Input 1: ")
# plt.axis('off')

# # Input 2
# plt.subplot(1, 4, 2)
# plt.imshow(X2[0], cmap='gray')
# plt.title(f"Input 2: ")
# plt.axis('off')

# # Output 1 (Sum - First Digit)
# plt.subplot(1, 4, 3)
# plt.imshow(Y1[0], cmap='gray')
# plt.title(f"Output 1: ")
# plt.axis('off')

# # Output 2 (Sum - Second Digit)
# plt.subplot(1, 4, 4)
# plt.imshow(Y2[0], cmap='gray')
# plt.title(f"Output 2: ")
# plt.axis('off')

# plt.show()





# -------------------------------------------------------  Structure  ----------------------------------------------------------- #



# Shared CNN for feature extraction
def build_shared_cnn(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    return input_layer, x

# Input layers for the two images
input1, features1 = build_shared_cnn((28, 28, 1))
input2, features2 = build_shared_cnn((28, 28, 1))

# Combine features
merged_features = concatenate([features1, features2])

# Debugging: Check shape after concatenation
print(f"Merged features shape: {merged_features.shape}")

# Fully connected layer for latent representation
latent = Dense(128, activation='relu')(merged_features)

# Debugging: Check shape of latent representation
print(f"Latent representation shape: {latent.shape}")

# Decoder for generating the first output image
decoder1 = Dense(7 * 7 * 64, activation='relu')(latent)
decoder1 = Reshape((7, 7, 64))(decoder1)
decoder1 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(decoder1)
decoder1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(decoder1)
output1 = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(decoder1)

# Decoder for generating the second output image
decoder2 = Dense(7 * 7 * 64, activation='relu')(latent)
decoder2 = Reshape((7, 7, 64))(decoder2)
decoder2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(decoder2)
decoder2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(decoder2)
output2 = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(decoder2)

# Define the model
model = Model(inputs=[input1, input2], outputs=[output1, output2])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mse'])

# Display the model summary
model.summary()

# Train the model
history = model.fit(
    [X1, X2],              # Input data: two inputs
    [Y1, Y2],              # Target data: two outputs
    batch_size=32,         # Batch size: 32 samples per batch
    epochs=20,             # Train for 10 epochs
    validation_split=0.2,  # Use 20% of the data for validation
    verbose=0             # Print progress during training
)

# Save the model
model.save('mnist_addition_model1.h5')

# Generate a single example
test_input1 = X1[0:1]  # First image in the dataset
test_input2 = X2[0:1]  # Second image in the dataset

# Predict outputs
predicted_output1, predicted_output2 = model.predict([test_input1, test_input2])

# Display inputs and predictions
display_prediction(test_input1[0], test_input2[0], predicted_output1[0], predicted_output2[0])

# Plot training and validation loss
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.show()

# Plot accuracy for each output
plt.figure(figsize=(12, 5))
plt.plot(history.history['accuracy'], label='Accuracy (Output 1)', color='green')
plt.plot(history.history['val_accuracy'], label='Val Accuracy (Output 1)', color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs. Validation Accuracy (Output 1)')
plt.legend()
plt.show()

# Plot accuracy for the second output if metrics are defined for it
if 'accuracy_1' in history.history:
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy_1'], label='Accuracy (Output 2)', color='purple')
    plt.plot(history.history['val_accuracy_1'], label='Val Accuracy (Output 2)', color='brown')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training vs. Validation Accuracy (Output 2)')
    plt.legend()
    plt.show()
