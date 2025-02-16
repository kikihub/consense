import os
import numpy as np
import torch
import pickle
# set random seed
torch.manual_seed(1)
np.random.seed(1)

filepath  = 'xxx/data/raw_data/mmfi_processed/'
actions_list = os.listdir(filepath)
actions_list = sorted(actions_list)
csilist = np.empty((0,3, 114, 10))
for action in actions_list:
    print('processing:',action)
    action_path = os.path.join(filepath,action)
    data = np.load(action_path, allow_pickle='TRUE')
    csilist = np.concatenate((csilist, data[:80]), axis=0)
arr = np.arange(27) 
labels = np.repeat(arr,80) 

csilist = np.transpose(csilist, (0, 3, 1, 2))
x, y = csilist, labels


print(x,x.shape)
print(y, y.shape)  


savepath = 'xxx/data/processed_data/mmfi-27-python'

train = {
    'data':x,
    'labels':y,
}


with open(os.path.join(savepath,'train'), 'wb') as train_file:
    pickle.dump(train, train_file)


