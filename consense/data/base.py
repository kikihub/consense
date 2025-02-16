import os
import sys
import pdb
import torch
import numpy as np
from copy import deepcopy
from data import *
from torch.utils.data import DataLoader
from utils import getconst

class ContinualSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n_tasks):

        self.first_task = False
        self.n_tasks = n_tasks 
        self.classes = np.unique(dataset.targets) 

        ds_targets = np.array(dataset.targets) 
        self.n_samples = ds_targets.shape[0] 
        self.n_classes = self.classes.shape[0] 

        assert self.n_classes % n_tasks == 0
        self.cpt = self.n_classes // n_tasks 

        self.sample_all_seen_tasks = False

        self.task = None
        self.target_indices = {}

        for label in self.classes:
            self.target_indices[label] = np.squeeze(np.argwhere(ds_targets == label))
            np.random.shuffle(self.target_indices[label])
            
    def __iter__(self):

        task = self.task
        
        if self.sample_all_seen_tasks:
            task_classes = self.classes[: self.cpt * (task + 1)] 
            
        else:
            task_classes = self.classes[self.cpt * task : self.cpt * (task + 1)] 

        task_samples = []

        for class_ in task_classes:
            t_indices = self.target_indices[class_] 
            task_samples += [t_indices] 

        task_samples = np.concatenate(task_samples)
        np.random.shuffle(task_samples)

        self.task_samples = task_samples
        for item in self.task_samples:
            yield item

    def __len__(self):
        samples_per_task = self.n_samples // self.n_tasks
        if self.first_task:
            return samples_per_task*(self.n_tasks//2)
        else:
            return samples_per_task

    def set_task(self, task, sample_all_seen_tasks=False,first_task=False):
        self.task = task
        self.sample_all_seen_tasks = sample_all_seen_tasks
        self.first_task = first_task
 
def get_data(args):
    
    dataset = getconst[args.dataset].dataset

    train_ds = dataset(train=True,root = args.data_root)
    test_ds = dataset(train=False,root = args.data_root)

    train_sampler = ContinualSampler(train_ds, args.n_tasks)
    train_loader = DataLoader(
        train_ds,
        num_workers=args.num_workers,
        sampler=train_sampler,
        batch_size=args.batch_size,
    )

    test_sampler = ContinualSampler(test_ds, args.n_tasks)
    test_loader = DataLoader(
        test_ds,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        sampler=test_sampler,
    )

    args.n_classes = train_sampler.n_classes
    args.n_classes_per_task = args.n_classes // args.n_tasks
    
    return train_loader, test_loader
