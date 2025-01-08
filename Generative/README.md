# Neural Network for Mathematical Operations Classification

This project is designed to develop a neural network that can generate images representing the results of simple mathematical operations. The network takes three input images: two images representing digits from the MNIST dataset and one image containing a mathematical symbol from the HASYv2 dataset. The network outputs two images representing the result of the arithmetic operation. This network uses an encoder-decoder architecture, where the features of the input images are encoded into a latent representation and then decoded to produce the output images.

## Dataset

The MNIST dataset is used for the numerical digits, and the HASYv2 dataset is used for the mathematical symbols (addition, subtraction, multiplication, and division). The following transformations are applied to the images:

- Resizing the images to match the MNIST format
- Inverting the colors from black characters on a white background to white characters on a black background

The network is trained to perform the following four basic operations: addition, subtraction, multiplication, and division.

## Tools Used

- **Python**: The primary programming language used for this project
- **Keras**: For building and training the neural network
- **TensorFlow**: Backend engine for Keras
- **NumPy**: For data manipulation
- **OpenCV**: For image processing

## Network Architecture

The network consists of convolutional layers followed by transposed convolutional layers for the decoder part. A latent space representation of size 256 is used to allow the model to learn the correct features and transformations between the images.

## How to Run

1. Clone the repository to your local machine.
2. Install the required dependencies:
   `pip install -r requirements.txt`
3. Run the training script to train the model:
   `python train_model.py`
4. Evaluate the model with:
   `python evaluate_model.py`
5. After training, the results of the classification task will be saved, and images will be generated showing the output.

## Evaluation

The model is evaluated on a test set of images where the output will be compared to the expected results for different mathematical operations. The network's performance is measured in terms of accuracy, and the final output images are visualized.

## Acknowledgments

- **MNIST Dataset**: [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- **HASYv2 Dataset**: [HASYv2 Dataset](https://www.arxiv-vanity.com/papers/1605.06191/)
- **Keras**: [Keras Documentation](https://keras.io/)
- **TensorFlow**: [TensorFlow Documentation](https://www.tensorflow.org/)
- **Inspiration and Information Source**: [NeuPSL: Neural Probabilistic Soft Logic](https://www.researchgate.net/publication/360961440_NeuPSL_Neural_Probabilistic_Soft_Logic)
