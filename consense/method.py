import copy
from typing import Any, Callable, Dict
from os import listdir
from os.path import isfile, join
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis as FCA
from utils import set_optimizer
from matplotlib import pyplot as plt
from model import model_init
from model import MaskedLinearDynamic

def reset_frozen_gradients(network, freeze_masks):
    mask_index = 0
    for module in network.modules():
        if isinstance(module,MaskedLinearDynamic):
            module.weight.grad[freeze_masks[mask_index][0]] = 0
            module.bias.grad[freeze_masks[mask_index][1]] = 0
            mask_index += 1
    return network

class Method(nn.Module):
    def __init__(self, model, args,mylogger,device):
        super(Method, self).__init__()

        self.args = args
        self.model = model
        self.mylogger = mylogger
        self.device = device
        self.loss = F.cross_entropy
        self.opt = set_optimizer(args, parameters=list(self.model.parameters()))
        self.premodel = None

        model_init(self.model)
        self.freeze_masks = None
        self.stable_indices = None


    @property
    def name(self):
        return "myMethod"

    def observe(self, inc_data,freeze_masks):
        inc_loss = self.process_inc(inc_data)
        self.update(inc_loss,freeze_masks)
        return inc_loss

    
    def process_inc(self, inc_data: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        x1, x2 = (inc_data["x"], inc_data["x"])
        aug_data = torch.cat((x1, x2), dim=0)
        pred = self.chosemodel(aug_data)
        loss_c = self.loss(pred, inc_data["y"].repeat(2))
        return  loss_c 

    def chosemodel(self, aug_data):
        features = self.model.return_hidden(self.model,aug_data)
        pred = self.model.forward_classifier(features)
        return pred
    
    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        features = self.model.return_hidden(self.model,x)
        logits = self.model.forward_pre(features,x)
        return logits, features
    
    def update(self, loss,freeze_masks):
        self.opt.zero_grad()
        loss.backward()
        
        if freeze_masks is not None:
            weight_masks = [mask[0] for mask in freeze_masks]
            bias_masks = [mask[1] for mask in freeze_masks]   
            mask_index = 0
            for module in self.model.modules():
                if isinstance(module,MaskedLinearDynamic):
                    for row in range(module.weight.grad.shape[0]):
                        for col in range(module.weight.grad.shape[1]):
                            if weight_masks[mask_index][row][col] == 1:
                                module.weight.grad[row][col] = 0
                    module.bias.grad[bias_masks[mask_index] == 1] = 0
                    mask_index += 1

        self.opt.step()
        
    def train(self):
        self.model.train()

    def eval(self, freeze_linear_heads=True):
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, loader, task):

        self.eval(freeze_linear_heads=True)
        accs = np.zeros(shape=(self.args.n_tasks,))
        for task_t in range(task + 1):
            # for task_t in range(1):

            n_ok, n_total = 0, 0
            loader.sampler.set_task(task_t)

            # iterate over samples from task
            for i, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                logits, features = self.predict(data)
                n_total += data.size(0)
                if logits is not None:
                    pred = logits.max(1)[1]
                    n_ok += pred.eq(target).sum().item()

            accs[task_t] = (n_ok / n_total) * 100
        avg_acc = np.mean(accs[: task + 1])
        accs_msg = "\t".join([str(int(x)) for x in accs])
        avg_acc_msg = f"\tAvg Acc: {avg_acc:.2f}"
        self.mylogger.info(f"\nAccuracy:{accs_msg}{avg_acc_msg}")

        return accs.tolist(),avg_acc

    def eval_agent(self, loader, task):
        eval_loader = loader
        accs,acc = self.evaluate(eval_loader, task)
        return accs,acc

    def on_task_start(self,task,start_task):

        if task <= start_task:
            pass
        if task == start_task + 1:
            pass
            # Freeze the qkv of the attention layer
            for name, param in self.model.named_parameters():
                if 'qkv' in name:
                    param.requires_grad = False
        if task >= start_task + 2:
            # Freeze the qkv of the attention layer
            for name, param in self.model.named_parameters():
                if 'qkv' in name:
                    param.requires_grad = False
            # Freeze the prev_conPerfix layer
            for name, param in self.model.named_parameters():
                if 'prev_conPerfix' in name:
                    param.requires_grad = False
                    
        for i, status in enumerate(self.model.feats_status):
            if status == 0:  
                self.model.feats_status[i] = 1
            

    def on_task_finish(self,task,start_task):

        if task <= start_task:
            pass
        else:
            # Save the conPerfix layer
            self.model.transformer.model.layers.self_attn.prev_conPerfix = copy.deepcopy(self.model.transformer.model.layers.self_attn.conPerfix)
        
        self.premodel = copy.deepcopy(self.model)
        
        for i, status in enumerate(self.model.feats_status):
            if status == 1:  
                self.model.feats_status[i] = 2
        #self.model.feats.append(self.model._create_feat())
        self.model.feats_status.append(0)
        
        
        
        
