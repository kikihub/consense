import os
import numpy as np
import torch
import pickle
# set random seed
torch.manual_seed(1)
np.random.seed(1)

filepath  = 'xxx/data/raw_data/xrf_processed/'
actions_list = os.listdir(filepath)
actions_list = sorted(actions_list)
csilist = np.empty((0,270,1000))
skiplist = [15,16,17,18,19,20,21] # remove seven double actions
i = 0
for action in actions_list[0:55]:
    if i in skiplist:
        i = i + 1
        continue
    else:
        print('processing:',action)
        action_path = os.path.join(filepath,action)
        data = np.load(action_path, allow_pickle='TRUE')
        csilist = np.concatenate((csilist, data[:14]), axis=0)
        i = i + 1
arr = np.arange(48)
labels = np.repeat(arr,14) 

x, y = csilist, labels

print(x,x.shape)
print(y,y.shape)  

savepath = 'xxx/data/processed_data/xrf-48-python'

train = {
    'data':x,
    'labels':y,
}

with open(os.path.join(savepath,'train'), 'wb') as train_file:
    pickle.dump(train, train_file)


