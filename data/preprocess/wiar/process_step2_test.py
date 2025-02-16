import os
import numpy as np
import torch
import pickle
# set random seed
torch.manual_seed(1)
np.random.seed(1)

filepath  = 'xxx/data/raw_data/wiar_processed/'
actions_list = os.listdir(filepath)
actions_list = sorted(actions_list)
csilist = np.empty((0,270,3,30))
for action in actions_list:
    print('processing:',action)
    action_path = os.path.join(filepath,action)
    data = np.load(action_path, allow_pickle='TRUE')
    csilist = np.concatenate((csilist, data[24:30]), axis=0)
arr = np.arange(16)
labels = np.repeat(arr,6) 

x, y = csilist, labels

print(x,x.shape)
print(y,y.shape)  

savepath = 'xxx/data/processed_data/wiar-16-python'

test = {
    'data':x,
    'labels':y,
}

with open(os.path.join(savepath,'test'), 'wb') as test_file:
    pickle.dump(test, test_file)


