import os
import numpy as np
import scipy.io as sio

# Define input and output folders
input_folder = 'xxx/data/raw_data/wiar/processed_dat/'
output_folder = 'xxx/data/raw_data/wiar/processed_npy/'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize a dictionary to store data for each action
action_data = {i: [] for i in range(1, 17)}

# Define the expected dimensions
expected_shape = (270, 3, 30)

# Iterate through all .mat files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.mat'):
        # Extract the action number
        parts = filename.split('_')
        action_num = int(parts[1][1:])
        
        # Load data from the .mat file
        filepath = os.path.join(input_folder, filename)
        mat = sio.loadmat(filepath)
        csidata = mat['csidata']
        
        # Ensure the data dimension is (270, 3, 30)
        if csidata.shape[0] < 270:
            # If the number of samples is less than 270, use zero padding
            padding = np.zeros((270 - csidata.shape[0], 3, 30))
            csidata = np.vstack((csidata, padding))
        elif csidata.shape[0] > 270:
            # If the number of samples is more than 270, take the first 270 samples
            csidata = csidata[:270, :, :]
        
        # Add the data to the corresponding action list
        action_data[action_num].append(csidata)

# Save the data for each action as .npy files
for action_num, data_list in action_data.items():
    # Convert the data list to a NumPy array
    data_array = np.array(data_list)
    
    # Save as .npy file
    output_path = os.path.join(output_folder, f'wiar_{action_num:02d}.npy')
    np.save(output_path, data_array)
    
    print(f'Saved action {action_num} data to {output_path}')

print('All files processed and saved.')
