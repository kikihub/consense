import os
import numpy as np
import scipy.io as sio
zipfilesPath = 'xxx/data/raw_data/mmfi'
fileS_list = os.listdir(zipfilesPath)
# actionsMap {label:1-27,data:[40*260,3,114,10]}
actionsMap = {}
for fileS_name in fileS_list:
    print(fileS_name)
    fileS_path = os.path.join(zipfilesPath, fileS_name)
    fileA_list = os.listdir(fileS_path)
    for fileA_name in fileA_list:
        actionNum = int(fileA_name[1:])
        fileA_path = os.path.join(fileS_path, fileA_name)
        file_csi_path = os.path.join(fileA_path,'wifi-csi')
        file_csi_list = os.listdir(file_csi_path)
        list = np.empty((0,3, 114, 10))
        for csi_name in file_csi_list:
            csi_path = os.path.join(file_csi_path, csi_name)
            data = sio.loadmat(csi_path)['CSIamp']
            data[np.isinf(data)] = np.nan
            for i in range(10):  # 32
                temp_col = data[:, :, i]
                nan_num = np.count_nonzero(temp_col != temp_col)
                if nan_num != 0:
                    temp_not_nan_col = temp_col[temp_col == temp_col]
                    temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data_frame = np.array(data)
            list = np.concatenate((list, np.expand_dims(data_frame, axis=0)), axis=0)
        list = list[:260]
        aaa = np.isnan(list).any()
        print(np.transpose(aaa))
        if actionNum in actionsMap:
            actionsMap[actionNum] = np.concatenate((actionsMap[actionNum], list), axis=0)
        else:
            actionsMap[actionNum] = list
print(actionsMap[1].shape)
for i in range(1,28,1):
    np.save(f'xxx/data/raw_data/mmfi_processed/mmfi_processed_a{i}', actionsMap[i])
