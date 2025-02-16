import numpy as np
import torch
from torch import nn
import copy
import math
from model import MaskedLinearDynamic


#This function is used to calculate the average activation value of the neural network on a given training set. It traverses the entire training set, calculates the sum of the activation values ​​of each batch, and finally obtains the average activation value of each neuron on the entire training set.
def compute_average_activation(network, train_loader, dev):
    total_activations = []
    network.train()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            _, activations = network.forward_activations(network,data)
            
            # Calculate the sum of activation values ​​for each batch
            batch_sum_activation = [
                torch.sum(activation, axis=(0, 1)) if len(activation.shape) == 3 else
                torch.sum(activation, axis=(0, 2, 3)) if len(activation.shape) == 4 else
                torch.sum(activation, axis=0)
                for activation in activations
            ]
            
            if len(total_activations) == 0:
                total_activations = batch_sum_activation
            else:
                total_activations = [total_activations[i] + activation for i, activation in enumerate(batch_sum_activation)]
                
    average_activations = [total_activation / len(train_loader.dataset) for total_activation in total_activations]
    return [average_activation.detach().cpu().numpy() for average_activation in average_activations]

def compute_average_activation_old(model,network, train_loader, dev):
    total_activations = []
    network.train()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(dev)
            _, activations = network.forward_activations(model,data)
            
            batch_sum_activation = [
                torch.sum(activation, axis=(0, 1)) if len(activation.shape) == 3 else
                torch.sum(activation, axis=(0, 2, 3)) if len(activation.shape) == 4 else
                torch.sum(activation, axis=0)
                for activation in activations
            ]
            
            if len(total_activations) == 0:
                total_activations = batch_sum_activation
            else:
                total_activations = [total_activations[i] + activation for i, activation in enumerate(batch_sum_activation)]
                
    average_activations = [total_activation / len(train_loader.dataset) for total_activation in total_activations]
    return [average_activation.detach().cpu().numpy() for average_activation in average_activations]

# The most active neurons are selected according to the activation values ​​of the neurons to determine the stable neurons.
def pick_top_neurons(activations, percentage,stables_indice_old=None):
    total = len(activations)
    if stables_indice_old is None:
        indices = [] 
    else:
        indices = stables_indice_old
    sort_indices = np.argsort(-activations) 
    num = 0
    for index in sort_indices:
        if stables_indice_old and (index in stables_indice_old):
            continue
        else:
            indices.append(index)
            num += 1
        if num >= total * percentage / 100:
            break
    return indices

def compute_drift(layer_activation,layer_activation_old,stables_indice_old=None):
    threshold = 0.001
    if stables_indice_old is None:
        indices = []
    else:
        indices = stables_indice_old
    l2_distance = np.sqrt((layer_activation - layer_activation_old) ** 2)
    threshold_indices = np.where(l2_distance > threshold)[0]
    indices = np.unique(np.sort(np.concatenate((indices, threshold_indices))))
    return indices

#Find the stable units that satisfies p criterion
def compute_stable_neurons(activations,activation_old, in_channels, activation_perc,stable_indices_old): 

    freezen_strategy = 2 # 1:avg 2:threshold
    stable_indices = []

    if freezen_strategy == 1:
        for i,layer_activation in enumerate(activations[1:]) :
            if stable_indices_old == None:
                stable_indices.append(pick_top_neurons(layer_activation, activation_perc))
            else:
                stable_indices.append(pick_top_neurons(layer_activation, activation_perc,stable_indices_old[i+1]))
    else:
        for i,layer_activation in enumerate(activations[1:]) :
            if stable_indices_old == None:
                stable_indices.append(pick_top_neurons(layer_activation, activation_perc))
            else:
                stable_indices.append(compute_drift(activations[i+1],activation_old[i+1],stable_indices_old[i+1]))
        

    return [list(range(in_channels))] + stable_indices
    

#Compute which connections to freeze and drop
def compute_freeze_and_drop(stable_indices, model):
    weight = []
    for module in model.modules():
        if isinstance(module, MaskedLinearDynamic):
            weight_mask, bias_mask = module.get_mask()
            weight.append((copy.deepcopy(weight_mask).cpu().numpy(), copy.deepcopy(bias_mask).cpu().numpy()))
    #Create Freeze Masks        
    freeze_masks = []
    for i, (source_stable, target_stable) in enumerate(zip(stable_indices[:-1], stable_indices[1:])):
        source_stable, target_stable = np.array(source_stable, dtype=np.int32), np.array(target_stable, dtype=np.int32)
        mask_w = np.zeros(weight[i][0].shape)
        #Conv2Conv
        if len(weight[i][0].shape) == 4:
            for src_unit_stable in source_stable:
               mask_w[target_stable, src_unit_stable, :, :] = np.ones((weight[i][0].shape[2], weight[i][0].shape[3]))
        #Conv2Linear or Linear2Linear
        else:
            #Conv2Linear
            if len(weight[i-1][0].shape) == 4:
                for src_unit_stable in source_stable:
                    
                    mask_w[target_stable, src_unit_stable*model.conv2lin_kernel_size:(src_unit_stable + 1)*model.conv2lin_kernel_size] = 1 
            #Linear2Linear
            else:
                for src_unit_stable in source_stable:
                    mask_w[target_stable, src_unit_stable] = 1 
        mask_b = np.zeros(weight[i][1].shape)
        mask_b[target_stable] = 1
        freeze_masks.append((mask_w, mask_b))
    return freeze_masks