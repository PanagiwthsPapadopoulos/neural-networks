# train | 100k rows, 200 classes --> 500 rows/class
# valid | 10k rows, 200 classes  -->  50 rows/class

# PANAGIOTIS PAPADOPOULOS 10697 HMMY 

import time
from datasets import load_dataset
from matplotlib import patches, pyplot as plt
import numpy as np
from keras import layers, models, optimizers
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

ds = load_dataset("zh-plus/tiny-imagenet")
print('DATASET LOADED SUCCESSFULLY')

# Extract images and labels from database into variables
train_images = [ds['train'][i]['image'] for i in range(0, len(ds['train']))]
testing_images_original = [ds['valid'][i]['image'] for i in range(0, len(ds['valid']))]
train_labels = [ds['train'][i]['label'] for i in range(0, len(ds['train']))]
testing_labels_original = [ds['valid'][i]['label'] for i in range(0, len(ds['valid']))]

# Split valid data to testing and valid images and labels
testing_images = [testing_images_original[i+j] for i in range(0, len(testing_images_original), 50) for j in range(25)]
validation_images = [testing_images_original[i+j] for i in range(0, len(testing_images_original), 50) for j in range(25, 50)]
testing_labels = [testing_labels_original[i+j] for i in range(0, len(testing_labels_original), 50) for j in range(25)]
validation_labels = [testing_labels_original[i+j] for i in range(0, len(testing_labels_original), 50) for j in range(25, 50)]




# Make images and labels into numpy arrays
train_images = [np.array(e) for e in train_images]
train_labels = [np.array(e) for e in train_labels]
testing_images = [np.array(e) for e in testing_images]
testing_labels = [np.array(e) for e in testing_labels]
validation_images = [np.array(e) for e in validation_images]
validation_labels = [np.array(e) for e in validation_labels]


# print dataset length
print('Train images len: '+str(len(train_images)))
print('Train labels len: '+str(len(train_labels)))
print('Testing images len: '+str(len(testing_images)))
print('Testing labels len: '+str(len(testing_labels)))
print('Validation images len: '+str(len(validation_images)))
print('Validation labels len: '+str(len(validation_labels)))


# Remove grayscale images from training set

indexes = []
for j in range(len(train_images)):
    if(len(train_images[j].shape)!=3):
        indexes.append(j)

for element in sorted(indexes, reverse=True):
    del train_images[element]
    del train_labels[element]


# Remove grayscale images from testing set

indexes = []
for j in range(len(testing_images)):
    if(len(testing_images[j].shape)!=3):
        indexes.append(j)

for element in sorted(indexes, reverse=True):
    del testing_images[element]
    del testing_labels[element]        


# Remove grayscale images from validation set

for j in range(len(validation_images)):
    if(len(validation_images[j].shape)!=3):
        indexes.append(j)

for element in sorted(indexes, reverse=True):
    del validation_images[element]
    del validation_labels[element]  


print('Deleted grayscale images')


# Make whole array into numpy array
train_images = np.array(train_images)
train_labels = np.array(train_labels)
testing_images = np.array(testing_images)
testing_labels = np.array(testing_labels)
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

# Rescale image RGB values to [0, 1]
train_images = train_images / 255
testing_images = testing_images / 255
validation_images = validation_images / 255

input_shape=(64, 64, 3)
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation= 'relu', input_shape = input_shape, strides=(2,2), padding='valid'))
# model.add(layers.Dropout(0.05))
model.add(layers.Conv2D(32, (3,3), activation= 'relu', input_shape = input_shape, strides=(2,2), padding='valid'))
# model.add(layers.Dropout(0.05))
model.add(layers.Conv2D(64, (3,3), activation= 'relu', input_shape = input_shape, strides=(2,2), padding='valid'))
# model.add(layers.Dropout(0.05))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu' )) 
model.add(layers.Dense(200, activation='softmax'))

model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# print(f"Train Images Shape: {train_images.shape}")
# print(f"Train Labels Shape: {train_labels.shape}")
# print(f"Validation Images Shape: {validation_images.shape}")
# print(f"Validation Labels Shape: {validation_labels.shape}")

start_time = time.time()
history = model.fit(train_images, train_labels, epochs=30, validation_data=(validation_images, validation_labels), shuffle=True)
end_time = time.time()


total_time = end_time - start_time
print(f"Total training time: {total_time:.2f} seconds")
# Extract accuracy and loss
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
test_loss, test_accuracy = model.evaluate(testing_images, testing_labels)

model.save('30EpochImageClassifier.h5')

# Plot training and validation accuracy/loss
epochs = range(1, len(history.history['accuracy']) + 1)
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=4)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], label='Training Loss', marker='o')
plt.plot(epochs, history.history['val_loss'], label='Validation Loss', marker='o')
plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

