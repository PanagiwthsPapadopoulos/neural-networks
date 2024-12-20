import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import seaborn as sns


# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Split into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


EPOCHS = 60

# Define hyperparameters
neurons = [64, 128, 256, 512, 1024]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128, 256]

resultsAdam = []
start_time = time.time()

# Train the model for different combinations of hyperparameters
for k in range (len(batch_sizes)):
    for j in range (len(learning_rates)):
        for i in range (len(neurons)):


            # -------------------------------------------------  Adam  ----------------------------------------------------------------#

            # Create the MLP model
            model = Sequential([
                Flatten(input_shape=(32, 32, 3)),  
                Dense(neurons[i], activation='relu'),    
                Dense(10, activation='linear')    
            ])

            # Define the optimizer and the loss function
            optimizer = Adam(learning_rate=learning_rates[j])
            model.compile(optimizer=optimizer, loss='categorical_hinge', metrics=['accuracy'])

            # Train the model
            history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_sizes[k], validation_data=(x_val, y_val), verbose=0)

            # Evaluate on the test set
            test_loss, test_acc = model.evaluate(x_test, y_test)
            print(f"Test Accuracy: {test_acc:.2f}")
            test_loss, test_accuracy = model.evaluate(x_test, y_test)

            # Store the results
            resultsAdam.append({'learning_rate': learning_rates[j], 'neurons': neurons[i], 'batch_size': batch_sizes[k], 'test_accuracy': test_accuracy})

end_time = time.time()
total_time = end_time - start_time
print(f'Total time: {total_time:.2f} ')


# ---------------------------------------------------  Heatmaps  -------------------------------------------------------------- #

    
results_df = pd.DataFrame(resultsAdam)
print(results_df)

# Generate heatmaps for each batch size
batch_values = results_df['batch_size'].unique()

for batch in batch_values:
    subset = results_df[(results_df['batch_size'] == batch)]

    pivot_table = results_df.pivot_table(
        index="learning_rate",                      # Row labels (C values)
        columns="neurons",                          # Column labels (gamma values)
        values="test_accuracy"                      # Values to display (mean test score)
    )
    
    # Create the heatmap
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="coolwarm")
    plt.title(f"Hyperparameter Heatmap (Learning Rate vs. No of Neurons) for optimizer Adam & batch size {batch}")
    plt.xlabel("No of Neurons")
    plt.ylabel("Learning Rate")
    plt.show()
    

