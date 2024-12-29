import numpy as np
import os
import matplotlib.pyplot as plt
import re

# Load the history
# history_path = 'HistoryResults/history_convlayers4_latent256_transposed_conv_layers3_learningrate0.1_batch_size32_epochs30.npy'  # Replace with your file path
# history_data = np.load(history_path, allow_pickle=True).item()
# print("Available keys in history:", history_data.keys())

files = sorted(os.listdir('HistoryResults'))

for file_name in files:

    history_path = 'HistoryResults/' + file_name
    history_data = np.load(history_path, allow_pickle=True).item()

    print("Available keys in history:", history_data.keys())

    plt.figure(2)
    plt.plot(history_data['loss'], label='Training Loss')
    plt.plot(history_data['val_loss'], label='Validation Loss')
    plt.xticks(range(1, len(history_data['loss']) ))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{file_name}\nTraining vs. Validation Loss')
    plt.legend()
    # plt.show()


    operator_loss = history_data['operator_output_loss']
    val_operator_loss = history_data['val_operator_output_loss']



    def find_output_keys_dynamically(history_keys):
        """
        Dynamically identifies keys for outputs and assigns them based on numerical order.

        Args:
            history_keys (list): List of keys in the history dictionary.

        Returns:
            dict: Mapped keys for output1, output2, and their validation metrics.
        """
        mse_keys = [key for key in history_keys if 'mse' in key and 'val' not in key]
        val_mse_keys = [key for key in history_keys if 'mse' in key and 'val' in key]
        
        # Extract numbers from keys
        mse_numbers = [(int(re.search(r'(\d+)', key).group()), key) for key in mse_keys]
        val_mse_numbers = [(int(re.search(r'(\d+)', key).group()), key) for key in val_mse_keys]
        
        # Sort keys by their numerical identifier
        mse_numbers.sort()
        val_mse_numbers.sort()
        
        # Assign dynamically based on sorted order
        output_keys = {
            'output1_mse': mse_numbers[0][1] if len(mse_numbers) > 0 else None,
            'output2_mse': mse_numbers[1][1] if len(mse_numbers) > 1 else None,
            'val_output1_mse': val_mse_numbers[0][1] if len(val_mse_numbers) > 0 else None,
            'val_output2_mse': val_mse_numbers[1][1] if len(val_mse_numbers) > 1 else None
        }
        
        return output_keys
    
    

    output_keys = find_output_keys_dynamically(history_data.keys())

    print(output_keys)

    # First output
    plt.figure(1)
    plt.plot(history_data[output_keys['output1_mse']], label='Training MSE (Output 1)')
    plt.plot(history_data[output_keys['val_output1_mse']], label='Validation MSE (Output 1)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('MSE for Output 1')
    plt.legend()
    

    # Second output
    plt.plot(history_data[output_keys['output2_mse']], label='Training MSE (Output 2)')
    plt.plot(history_data[output_keys['val_output2_mse']], label='Validation MSE (Output 2)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.xticks(range(1, len(history_data['loss'])))
    plt.title('MSE for both Outputs')
    plt.legend()

    # Operand output ce
    plt.plot(history_data['operator_output_categorical_crossentropy'], label='Training Loss (Operand ce)')
    plt.plot(history_data['val_operator_output_categorical_crossentropy'], label='Validation Loss (Operand ce)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Operand Identification Loss')
    plt.legend()

    # # Operand output 
    # plt.plot(history_data['operator_output_loss'], label='Training Loss (Operand)')
    # plt.plot(history_data['val_operator_output_loss'], label='Validation Loss (Operand)')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Operand Identification Loss')
    # plt.legend()
    plt.show()


    # plt.plot(history_data['conv2d_transpose_19_loss'], label='Loss (Output 1)')
    # plt.plot(history_data['conv2d_transpose_23_loss'], label='Loss (Output 2)')
    # plt.plot(history_data['loss'], label='Total Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss for All Outputs')
    # plt.legend()
    # plt.show()


    # plt.plot(history_data['val_conv2d_transpose_19_loss'], label='Val Loss (Output 1)')
    # plt.plot(history_data['val_conv2d_transpose_23_loss'], label='Val Loss (Output 2)')
    # plt.plot(history_data['val_loss'], label='Total Val Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Validation Loss for All Outputs')
    # plt.legend()
    # plt.show()
    # Extract relevant metrics
    