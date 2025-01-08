# README for Project 2: Image Classification with Support Vector Machine (SVM), KNN, and MLP

## Overview

This project focuses on image classification using the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes. The primary objective is to implement and compare three different machine learning models: **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, and **Multilayer Perceptron (MLP)**. The project involves steps like data preprocessing, model training, parameter tuning, and performance evaluation.

## Dataset

The **CIFAR-10** dataset used in this project contains:
- 60,000 images, each of size 32x32 pixels, classified into 10 classes.
- The dataset is divided into 50,000 training images and 10,000 test images.
- Image data is normalized, and preprocessing is done to scale pixel values into the range [0,1] and convert them into a format suitable for machine learning models.

## Tools and Libraries Used

- **Python**: Programming language used for the project.
- **Keras**: For building and training the MLP model.
- **Scikit-learn**: For implementing the SVM and KNN classifiers, as well as performing grid search for hyperparameter tuning.
- **TensorFlow**: For advanced model training and evaluation.
- **NumPy**: For numerical operations and data manipulation.
- **Pandas**: For data manipulation and storing results.

## Files

The repository contains several Python files, each performing different tasks related to image classification:

1. **PanagiotisPapadopoulosErgasia2_10697.py**: Contains the main implementation of the image classification, including data preprocessing, model training, and evaluation.
2. **PanagiotisPapadopoulosErgasia2_10697_MLP.py**: Implements the Multilayer Perceptron (MLP) model for image classification.
3. **PanagiotisPapadopoulosErgasia2_10697_KNN_NCC.py**: Implements the KNN classifier for image classification, including feature extraction and grid search.
4. **PanagiotisPapadopoulos10697Ergasia2_10697_Test.py**: Includes testing scripts to evaluate the models on the CIFAR-10 test dataset.

## How to Run

1. **Install the required libraries**:
   Ensure that all necessary libraries are installed by running the following command:
```bash
  pip install numpy pandas tensorflow scikit-learn matplotlib
 ```
2. **Load and preprocess the dataset**:
The CIFAR-10 dataset can be automatically loaded using Keras. The dataset will be preprocessed and normalized to be ready for model training.

3. **Run the models**:
You can run the models by executing the corresponding Python scripts. For example, to run the KNN and MLP models, use:
```bash
python PanagiotisPapadopoulosErgasia2_10697_KNN_NCC.py
python PanagiotisPapadopoulosErgasia2_10697_MLP.py
```
4. **Model Evaluation**:
After running the training scripts, the models will be evaluated on the test dataset and the performance metrics (accuracy, precision, recall, etc.) will be displayed.

The models' performance was compared using various metrics and visualized for better understanding.

## Conclusion

This project successfully implemented and compared three different image classification models on the CIFAR-10 dataset. The SVM, KNN, and MLP models were evaluated, and the results showed how each model performed under different conditions and parameter settings. 

The findings can be used to draw conclusions about the trade-offs between classical machine learning models like SVM and KNN, and more complex models like MLP, when applied to image classification tasks.
