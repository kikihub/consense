from data.wiar16 import WIAR16
from data.mmfi27 import MMFI27
from data.xrf48 import XRF48
import torch

def compute_prototypes_wiar(n_tasks, half_iid, dataloader):
    task_cls_num = 16//n_tasks
    if half_iid:
        first_task = ((n_tasks // 2) - 1)
    else:
        first_task = 0
    prototypes = [torch.zeros(task_cls_num, 270, 3, 30)
                    for _ in range(n_tasks)]

    for task_t in range(n_tasks):
        task_prototypes = prototypes[task_t]

        dataloader.sampler.set_task(task_t)

        class_samples = {i: [] for i in range(task_cls_num)}

        task_classes = [task_t * task_cls_num,
                        task_t * task_cls_num + 1]

        for data, target in dataloader:
            for i in range(len(target)):
                label = target[i].item()
                if label in task_classes:
                    class_samples[label - task_classes[0]].append(data[i])

        for class_id in range(task_cls_num):
            label = task_classes[class_id]
            if len(class_samples[class_id]) > 0:
                class_samples_tensor = torch.stack(
                    class_samples[class_id], dim=0)
                class_prototype = class_samples_tensor.mean(dim=0)
                task_prototypes[class_id] = class_prototype

    return [torch.cat(prototypes[:first_task+1], dim=0)] + prototypes[first_task+1:]

def compute_prototypes_mmfi(n_tasks, half_iid, dataloader):
    task_cls_num = 27//n_tasks
    if half_iid:
        first_task = ((n_tasks // 2) -1)
    else:
        first_task = 0
    prototypes = [torch.zeros(task_cls_num, 10, 3, 114)
                    for _ in range(n_tasks)]

    for task_t in range(n_tasks):
        task_prototypes = prototypes[task_t]

        dataloader.sampler.set_task(task_t)

        class_samples = {i: [] for i in range(task_cls_num)}

        task_classes = [task_t * task_cls_num,
                        task_t * task_cls_num + 1, task_t * task_cls_num + 2]

        for data, target in dataloader:
            for i in range(len(target)):
                label = target[i].item()
                if label in task_classes:
                    class_samples[label - task_classes[0]].append(data[i])

        for class_id in range(task_cls_num):
            label = task_classes[class_id]
            if len(class_samples[class_id]) > 0:
                class_samples_tensor = torch.stack(
                    class_samples[class_id], dim=0)
                class_prototype = class_samples_tensor.mean(dim=0)
                task_prototypes[class_id] = class_prototype

    return [torch.cat(prototypes[:first_task+1],dim=0)] + prototypes[first_task+1:]

def compute_prototypes_xrf(n_tasks, half_iid, dataloader):
    task_cls_num = 48//n_tasks
    if half_iid:
        first_task = ((n_tasks // 2) - 1)
    else:
        first_task = 0
    prototypes = [torch.zeros(task_cls_num, 270, 50)
                    for _ in range(n_tasks)]

    for task_t in range(n_tasks):
        task_prototypes = prototypes[task_t]

        dataloader.sampler.set_task(task_t)

        class_samples = {i: [] for i in range(task_cls_num)}

        task_classes = [task_t * task_cls_num,
                        task_t * task_cls_num + 1,
                        task_t * task_cls_num + 2,
                        task_t * task_cls_num + 3,
                        task_t * task_cls_num + 4,
                        task_t * task_cls_num + 5]

        for data, target in dataloader:
            for i in range(len(target)):
                label = target[i].item()
                if label in task_classes:
                    class_samples[label - task_classes[0]].append(data[i])

        for class_id in range(task_cls_num):
            label = task_classes[class_id]
            if len(class_samples[class_id]) > 0:
                class_samples_tensor = torch.stack(
                    class_samples[class_id], dim=0)
                class_prototype = class_samples_tensor.mean(dim=0)
                task_prototypes[class_id] = class_prototype

    return [torch.cat(prototypes[:first_task+1], dim=0)] + prototypes[first_task+1:]


class WIARConstants:
    dataset = WIAR16
    class_num = 16
    temporal_dim = 270
    channel_dim = 90
    neurons_in_channels = 270
    compute_prototypes = compute_prototypes_wiar


class MMFIConstants:
    dataset = MMFI27
    class_num = 27
    temporal_dim = 10
    channel_dim = 342
    neurons_in_channels = 10
    compute_prototypes = compute_prototypes_mmfi


class XRFConstants:
    dataset = XRF48
    class_num = 48
    temporal_dim = 50
    channel_dim = 270
    neurons_in_channels = 50
    compute_prototypes = compute_prototypes_xrf

# 字典映射数据集名称到对应的常量类
getconst = {
    "wiar": WIARConstants,
    "mmfi": MMFIConstants,
    "xrf": XRFConstants,
}
