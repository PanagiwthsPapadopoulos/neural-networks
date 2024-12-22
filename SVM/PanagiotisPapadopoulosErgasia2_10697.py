import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Load the CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Flatten the images into 1D vectors (32x32x3 -> 3072)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Standardize features to have mean 0 and variance 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reduce dimensions using PCA (keeping 99% of the variance)
pca = PCA(n_components=0.90)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Toggle grid search mode
gridSearch = False

if not gridSearch:

    # Train and evaluate a single SVM model
    start_time = time.time()

    # Create the SVM model
    svm = SVC(kernel='rbf', C=0.001, gamma = 0.1, max_iter=4000)
    svm.fit(X_train, y_train)


    # Evaluate the final model on the test data
    y_pred_test = svm.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred_test))
    print("Classification Report (Test):")
    print(classification_report(y_test, y_pred_test))

   # Measure training time
    end_time = time.time()
    print("training time: " + str(-start_time + end_time))

    # Plot the confusion matrix for test results
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Test Data)")
    plt.show()

    joblib.dump(svm, 'svm_model.pkl')

else:

# --------------------------------------------  Grid Search  --------------------------------------------- #

    # gridIndex
    # Determines which grid search will be executed
    # Value 0: linear Kernel
    # Value 1: rbf Kernel
    # Value 2: polynomial Kernel
    # Value 3: sigmoid Kernel

    # The actual parameter grid is shown below
    gridIndex = 3

    # Define the parameter grid
    param_grid = [
        {
            'kernel': ['linear'],
            'C': [0.01, 0.1, 1, 10, 100]
        },
        {
            'kernel': ['rbf'],
            'C': [0.00001, 0.0001, 0.001, 0.01, 0.1],
            'gamma': [0.1]
        },
        {
            'kernel': ['poly'],
            'C': [100, 1000, 10000, 100000],
            'gamma': [0.0001, 0.001, 0.01, 0.1],
            'degree': [2],
            'coef0': [2.0]
        },
        {
            'kernel': ['sigmoid'],
            'C': [0.1, 1, 10, 100],
            'gamma': [0.1, 1, 10],
            'coef0': [0, 0.5, 1]
        }
    ]

    # -------------------------------------- Training --------------------------------------------- #
    start_time = time.time()

    # Train an SVM classifier with RBF kernel
    svm = SVC(max_iter=60)  
    svm.fit(X_train, y_train)  

    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(svm, param_grid[gridIndex], cv=cv, n_jobs=-1)
    # Fit the model
    grid_search.fit(X_train, y_train)



    # ----------------------------------------------- Results ------------------------------------------------ #
    end_time = time.time()
    print("training time: " + str(end_time - start_time))

    results = grid_search.cv_results_
    print(results)



    # Print the best parameters found
    print(f"Best Parameters: {grid_search.best_params_}")

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Evaluate the model on the test set
    test_accuracy = best_model.score(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")


    # Make predictions on the test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    y_pred = svm.predict(X_test)

    # Evaluate the SVM classifier
    print("Classification report on validation data:")
    print(classification_report(y_test, y_pred))


    # --------------------------------    Confusion Matrix for best estimator  --------------------------  #

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


    # ------------------------------------ HeatMap of C and Gamma ------------------------------------------- #

    if(param_grid == 1):
        # Convert the GridSearchCV results to a DataFrame
        results_df = pd.DataFrame(grid_search.cv_results_)
        pd.set_option('display.max_rows', None)  # Show all rows
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Adjust the display width to fit the data
        pd.set_option('display.max_colwidth', None)  # Show full column content
        print(results_df)

        # Create a pivot table to visualize the mean_test_score for C and gamma
        pivot_table = results_df.pivot_table(
            index="param_C",          # Row labels (C values)
            columns="param_gamma",    # Column labels (gamma values)
            values="mean_test_score"  # Values to display (mean test score)
        )

        # Visualize the pivot table as a heatmap
        sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="coolwarm")
        plt.title("Hyperparameter Heatmap (C vs. Gamma)")
        plt.xlabel("Gamma")
        plt.ylabel("C")
        plt.show()

    


    # ------------------------------------------------------ Polynomial Kernel Visualization ----------------------------------------- #


    if(param_grid == 2):

        # Convert results to a DataFrame
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Unique values for coef0 and degree
        coef0_values = results_df['param_coef0'].unique()
        degree_values = results_df['param_degree'].unique()

        # Loop through each combination of coef0 and degree
        for coef0 in coef0_values:
            for degree in degree_values:
                # Filter data for the current slice
                subset = results_df[(results_df['param_coef0'] == coef0) & (results_df['param_degree'] == degree)]

                pivot_table = subset.pivot_table(
                  index="param_gamma",          # Row labels (C values)
                  columns="param_C",    # Column labels (gamma values)
                  values="mean_test_score"  # Values to display (mean test score)
               )

                

                # Create the heatmap
                plt.figure(figsize=(8, 6))
                sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean Accuracy'})

                # Add labels and title
                plt.title(f"Heatmap for Coef0={coef0}, Degree={degree}")
                plt.xlabel("C (Regularization)")
                plt.ylabel("Gamma (Kernel Coefficient)")
                plt.tight_layout()

                # Show the heatmap
                plt.show()


    # ------------------------------------------------------ Sigmoid Kernel Visualization ----------------------------------------- #

    if(param_grid == 3):


        # Convert results to a DataFrame
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Unique values for coef0 and degree
        coef0_values = results_df['param_coef0'].unique()


        # Loop through each combination of coef0 and degree
        for coef0 in coef0_values:
            
            # Filter data for the current slice
            subset = results_df[(results_df['param_coef0'] == coef0)]

            pivot_table = subset.pivot_table(
                index="param_gamma",          # Row labels (C values)
                columns="param_C",    # Column labels (gamma values)
                values="mean_test_score"  # Values to display (mean test score)
            )

            

            # Create the heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean Accuracy'})

            # Add labels and title
            plt.title(f"Heatmap for Coef0={coef0}")
            plt.xlabel("C (Regularization)")
            plt.ylabel("Gamma (Kernel Coefficient)")
            plt.tight_layout()

            # Show the heatmap
            plt.show()

