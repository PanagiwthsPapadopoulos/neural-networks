from PIL import Image 
import keras
import numpy as np

# PANAGIOTIS PAPADOPOULOS 10697 HMMY 


# Convert image to 64x64

# Open an image file
img = Image.open("/image-name.jpeg")                    # Change image and directory 

# Resize the image
resized_img = img.resize((64, 64))  

# Save or display the resized image
resized_img.save("resized-image-name.jpg", dir="/directory-path")     # Change image and directory 
resized_img.show()

img.show()

# Expand dimensions to add batch size

img = np.array(img)
print(img.shape)

# Regularization
img = img / 255.0                               
img = np.expand_dims(img, axis=0)
print(img.shape)


# Load Model
loaded_model = keras.saving.load_model("/directory-path/model-name")

# Process image
results = loaded_model.predict(img)

# Get the index of the highest probability (class label)
predicted_class = np.argmax(results, axis=1)

# Output the guessed classification
print("Predicted class:", predicted_class[0])

