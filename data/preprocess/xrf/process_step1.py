import numpy as np
import os
import glob

# Path to the new file
newfile = 'xxx/data/raw_data/XRF55/xrf55_processed'
# Find all .npy files
npy_files = glob.glob('xxx/data/raw_data/XRF55/Scene1/Scene1/WiFi/*.npy')

# Path to source and target folders
source_folder = 'xxx/data/raw_data/XRF55/Scene1/Scene1/WiFi'
target_folder = 'xxx/data/raw_data/XRF55/xrf55_processed'

# Get all file names
files = os.listdir(source_folder)

# Create target folder if it does not exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Initialize dictionary to store lists of files for each label
label_files_dict = {}

# Iterate over all files and categorize them by label
for file in files:
    if file.endswith('.npy'):
        parts = file.split('_')
        label = int(parts[1])  # Extract the label
        if label not in label_files_dict:
            label_files_dict[label] = []
        label_files_dict[label].append(file)

# Process files by label and save to the target folder
for label in range(1, 56):
    array_list = []
    if label in label_files_dict:
        print('Processing label', label)
        for file in label_files_dict[label]:
            file_path = os.path.join(source_folder, file)
            data = np.load(file_path)
            array_list.append(data)
        # Combine the list of arrays into a single array
        combined_array = np.array(array_list)
        # Save to the target folder
        target_file_name = f'xrf_{label}.npy'
        target_file_path = os.path.join(target_folder, target_file_name)
        np.save(target_file_path, combined_array)
        print(f'Saved {target_file_name} to {target_file_path}')
    else:
        print(f'No files found for label {label}')
