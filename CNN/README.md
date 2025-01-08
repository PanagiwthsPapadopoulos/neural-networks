# Neural Networks - Tiny ImageNet Classification

This repository contains an implementation of a convolutional neural network (CNN) for image classification using the Tiny ImageNet dataset. The project is designed to explore hyperparameter optimization and evaluate the performance of different configurations to achieve the best results.

## Projects Overview

### 1. Image Classification using Convolutional Neural Networks (CNN)
- **Dataset:** Tiny ImageNet
- **Goal:** Build a convolutional neural network to classify images from the Tiny ImageNet dataset. The dataset consists of 200 classes, each containing 500 training images, and 50 validation images per class.
- **Approach:** The CNN model utilizes various layers such as convolutional, pooling, and dense layers, with different activation functions and optimization techniques.

### 2. Hyperparameter Optimization and Performance Evaluation
- **Goal:** Conduct an extensive evaluation of the model's performance based on different hyperparameters.
- **Hyperparameters Tuned:**
  - Learning rate
  - Batch size
  - Number of epochs
  - Number of convolutional filters
  - Optimizer choice (Adam, SGD, etc.)
  
- **Result:** The optimization aims to improve model accuracy while balancing training time and computational resources.

## Files in this Repository

### 1. `PanagiotisPapadopoulos10697Ergasia1.py`
This script contains the code for the neural network model. It implements a CNN that can be trained on the Tiny ImageNet dataset, with various configurations for the convolutional and dense layers, activation functions, and optimizers. The model is built using Keras and TensorFlow.

### 2. `PanagiotisPapadopoulos10697Ergasia1Results.py`
This script evaluates the trained model and displays the performance metrics (accuracy, loss) during training and testing. It also generates plots for training history, including accuracy and loss curves.

### 3. `PapadopoulosPanagiotis10697Ergasia1.pdf`
This is the report that documents the steps taken to build and evaluate the CNN model. It includes explanations of the model architecture, the dataset used, the hyperparameter tuning process, and the final results.

## Setup

To run the scripts, you need the following dependencies:

- Python 3.x
- TensorFlow (version compatible with Keras)
- NumPy
- Matplotlib
- pandas
- scikit-learn

You can install the necessary dependencies by running:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```
## How to Run

### 1. Prepare the Data
- Download the Tiny ImageNet dataset from the [official website](http://www.image-net.org/challenges/LSVRC/2015/).
- Preprocess the dataset according to the script requirements. This usually includes resizing images and organizing them into training and validation directories.

### 2. Train the Model
- To train the CNN model, run the `PanagiotisPapadopoulos10697Ergasia1.py` script. This script will automatically handle the training process, including the configuration of hyperparameters and the choice of optimizer.
  
```bash
python PanagiotisPapadopoulos10697Ergasia1.py
```

### Evaluate the Model
After training the model, use the script `PanagiotisPapadopoulos10697Ergasia1Results.py` to evaluate its performance. The script will calculate and display performance metrics such as accuracy and loss over the training epochs. Additionally, it will generate visual plots that depict the training and validation accuracy/loss curves.

Run the script with the following command:

```bash
python PanagiotisPapadopoulos10697Ergasia1Results.py
```

### View the Report
For a detailed analysis of the model, training procedure, and results, refer to the `PapadopoulosPanagiotis10697Ergasia1.pdf` report. The report includes:
- A detailed summary of the model architecture and its performance.
- Graphs that visualize the accuracy and loss curves during training and validation.
- A comparison of the results across different hyperparameter configurations.
- A summary of the best performing configuration along with its final accuracy on the test set.

## Results

In the provided report (`PapadopoulosPanagiotis10697Ergasia1.pdf`), you will find:
- A detailed summary of the model architecture and its performance.
- Graphs that visualize the accuracy and loss curves during training and validation.
- A comparison of the results across different hyperparameter configurations.
- A summary of the best performing configuration along with its final accuracy on the test set.

## Acknowledgments

- **Tiny ImageNet Dataset**: [Tiny ImageNet](http://www.image-net.org/challenges/LSVRC/2015/)
- **Keras**: [Keras Documentation](https://keras.io/)
- **TensorFlow**: [TensorFlow Documentation](https://www.tensorflow.org/)
