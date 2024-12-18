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


# Φόρτωση και προεπεξεργασία του CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Κανονικοποίηση εικόνων
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Μετατροπή των ετικετών σε κατηγορίες (one-hot encoding)
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)


EPOCHS = 60

neurons = [64, 128, 256, 512, 1024]
# neurons = [64, 128]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
# learning_rates = [0.1, 0.01]
batch_sizes = [32, 64, 128, 256]
# batch_sizes = [32, 64]

resultsAdam = []
resultsSGD = []


start_time = time.time()

for k in range (len(batch_sizes)):
    for j in range (len(learning_rates)):
        for i in range (len(neurons)):


            # -------------------------------------------------  Adam  ----------------------------------------------------------------#

            # Δημιουργία MLP μοντέλου
            model = Sequential([
                Flatten(input_shape=(32, 32, 3)),  # Μετατροπή εικόνων σε επίπεδο διάνυσμα
                Dense(neurons[i], activation='relu'),    # Κρυφό επίπεδο
                Dense(10, activation='linear')    # Γραμμική έξοδος για Hinge Loss
            ])

            # Ορισμός του optimizer και της συνάρτησης απώλειας
            optimizer = Adam(learning_rate=learning_rates[j])
            model.compile(optimizer=optimizer, loss='categorical_hinge', metrics=['accuracy'])

            # Εκπαίδευση του μοντέλου
            history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_sizes[k], validation_data=(x_val, y_val), verbose=0)

            # Αξιολόγηση στο test set
            test_loss, test_acc = model.evaluate(x_test, y_test)
            print(f"Test Accuracy: {test_acc:.2f}")
            # train_accuracy = history.history['accuracy']
            # val_accuracy = history.history['val_accuracy']
            test_loss, test_accuracy = model.evaluate(x_test, y_test)

            # resultsAdam.append((f'Accuracy vs. Epochs for Adam Optimizer, {neurons[i]} neurons at the hidden layer, {learning_rates[j]} learning rate and {batch_sizes[k]} batch size', history, test_accuracy))
            resultsAdam.append({'learning_rate': learning_rates[j], 'neurons': neurons[i], 'batch_size': batch_sizes[k], 'test_accuracy': test_accuracy})


            # -------------------------------------------------  SGD  ----------------------------------------------------------------#


            # Δημιουργία MLP μοντέλου
            model = Sequential([
                Flatten(input_shape=(32, 32, 3)),  # Μετατροπή εικόνων σε επίπεδο διάνυσμα
                Dense(neurons[i], activation='relu'),    # Κρυφό επίπεδο
                Dense(10, activation='linear')    # Γραμμική έξοδος για Hinge Loss
            ])

            # Ορισμός του optimizer και της συνάρτησης απώλειας
            optimizer = SGD(learning_rate=learning_rates[j], momentum = 0.9)
            model.compile(optimizer=optimizer, loss='categorical_hinge', metrics=['accuracy'])

            # Εκπαίδευση του μοντέλου
            history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=batch_sizes[k], validation_data=(x_val, y_val), verbose=0)

            # Αξιολόγηση στο test set
            test_loss, test_acc = model.evaluate(x_test, y_test)
            print(f"Test Accuracy: {test_acc:.2f}")

            test_loss, test_accuracy = model.evaluate(x_test, y_test)
            # resultsSGD.append((f'Accuracy vs. Epochs for SGD Optimizer, {neurons[i]} neurons at the hidden layer, {learning_rates[k]} learning rate and {batch_sizes[k]} batch size', history, test_accuracy))
            resultsAdam.append({'learning_rate': learning_rates[j], 'neurons': neurons[i], 'batch_size': batch_sizes[k], 'test_accuracy': test_accuracy})


end_time = time.time()
total_time = end_time - start_time
print(f'Total time: {total_time:.2f} ')


# ---------------------------------------------------  Heatmaps  -------------------------------------------------------------- #

    
results_df = pd.DataFrame(resultsAdam)

batch_values = results_df['batch_size'].unique()

print(results_df)

for batch in batch_values:

    subset = results_df[(results_df['batch_size'] == batch)]

    pivot_table = results_df.pivot_table(
        index="learning_rate",          # Row labels (C values)
        columns="neurons",    # Column labels (gamma values)
        values="test_accuracy"  # Values to display (mean test score)
    )
    
    
    
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="coolwarm")
    plt.title(f"Hyperparameter Heatmap (Learning Rate vs. No of Neurons) for optimizer Adam & batch size {batch}")
    plt.xlabel("No of Neurons")
    plt.ylabel("Learning Rate")
    plt.show()
    
    

results_df = pd.DataFrame(resultsSGD)

batch_values = results_df['batch_size'].unique()

print(results_df)

for batch in batch_values:

    subset = results_df[(results_df['batch_size'] == batch)]

    pivot_table = results_df.pivot_table(
        index="learning_rate",          # Row labels (C values)
        columns="neurons",    # Column labels (gamma values)
        values="test_accuracy"  # Values to display (mean test score)
    )
    
    
    
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="coolwarm")
    plt.title(f"Hyperparameter Heatmap (Learning Rate vs. No of Neurons) for optimizer SGD & batch size {batch}")
    plt.xlabel("No of Neurons")
    plt.ylabel("Learning Rate")
    plt.show()
    
    
    
    
    
    
    
# Accuracy plots

# for i in range (len(resultsAdam)): 

#     title = resultsAdam[i][0]
#     history = resultsAdam[i][1]
#     test_accuracy = resultsAdam[i][2]
#     epochs = range(1, EPOCHS + 1)

#     plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', marker='o')
#     plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', marker='o')
#     plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
#     plt.title(title)
#     plt.tight_layout()
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(loc=4)
#     plt.show()


# for i in range (len(resultsSGD)): 

#     title = resultsSGD[i][0]
#     history = resultsSGD[i][1]
#     test_accuracy = resultsSGD[i][2]
#     epochs = range(1, EPOCHS + 1)

#     plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', marker='o')
#     plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', marker='o')
#     plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
#     plt.title(title)
#     plt.tight_layout()
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend(loc=4)
#     plt.show()

# test_accuracies = np.array([])

# for i in range (len(resultsAdam)): 
#     test_accuracies = np.append(test_accuracies, resultsAdam[i][2])

# test_accuracies = np.array([11,12,13,21,22,23,31,32,33])

