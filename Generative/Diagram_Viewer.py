import numpy as np
import os
import matplotlib.pyplot as plt
import re

# Load all files in the HistoryResults directory
files = sorted(os.listdir('HistoryResults'))

for file_name in files:
    history_path = f'HistoryResults/{file_name}'
    history_data = np.load(history_path, allow_pickle=True).item()

    print(f"Available keys in {file_name}:", history_data.keys())
    
    plt.figure(figsize=(15, 7))

    # Dynamically find output keys for MSE
    def find_output_keys_dynamically(history_keys):
        mse_keys = [key for key in history_keys if 'mse' in key and 'val' not in key]
        val_mse_keys = [key for key in history_keys if 'mse' in key and 'val' in key]
        
        mse_numbers = [(int(re.search(r'(\d+)', key).group()), key) for key in mse_keys]
        val_mse_numbers = [(int(re.search(r'(\d+)', key).group()), key) for key in val_mse_keys]
        
        mse_numbers.sort()
        val_mse_numbers.sort()
        
        return {
            'output1_mse': mse_numbers[0][1] if len(mse_numbers) > 0 else None,
            'output2_mse': mse_numbers[1][1] if len(mse_numbers) > 1 else None,
            'val_output1_mse': val_mse_numbers[0][1] if len(val_mse_numbers) > 0 else None,
            'val_output2_mse': val_mse_numbers[1][1] if len(val_mse_numbers) > 1 else None
        }
    
    output_keys = find_output_keys_dynamically(history_data.keys())
    
    # Unified plot for MSE outputs and Operand Loss
    if output_keys['output1_mse'] and output_keys['val_output1_mse']:
        plt.plot(history_data[output_keys['output1_mse']], label='Training MSE (Output 1)', linestyle='--')
        plt.plot(history_data[output_keys['val_output1_mse']], label='Validation MSE (Output 1)', linestyle='-')
    
    if output_keys['output2_mse'] and output_keys['val_output2_mse']:
        plt.plot(history_data[output_keys['output2_mse']], label='Training MSE (Output 2)', linestyle='--')
        plt.plot(history_data[output_keys['val_output2_mse']], label='Validation MSE (Output 2)', linestyle='-')
    
    if 'operator_output_categorical_crossentropy' in history_data and 'val_operator_output_categorical_crossentropy' in history_data:
        plt.plot(history_data['operator_output_categorical_crossentropy'], label='Training Loss (Operand CE)', linestyle='-.')
        plt.plot(history_data['val_operator_output_categorical_crossentropy'], label='Validation Loss (Operand CE)', linestyle=':')

    plt.xlabel('Epochs')
    plt.ylabel('Loss / MSE')
    plt.title('Unified Loss and MSE Metrics')
    plt.legend()
    
    # General loss
    if 'loss' in history_data and 'val_loss' in history_data:
        plt.figure()
        plt.plot(history_data['loss'], label='Training Loss')
        plt.plot(history_data['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{file_name}\nTraining vs. Validation Loss')
        plt.legend()

    if 'accuracy' in history_data and 'val_accuracy' in history_data:
        plt.figure()
        plt.plot(history_data['accuracy'], label='Training Accuracy')
        plt.plot(history_data['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'{file_name}\nTraining vs. Validation Accuracy')
        plt.legend()
    
    plt.tight_layout()
    plt.show()