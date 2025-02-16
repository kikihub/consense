import time
import numpy as np
from collections import OrderedDict as OD
from data.base import *
from model import HARTrans
from method import Method
from utils import get_parser, set_seed,get_logger, log_hyperparameters, print_model_parameters, log_gpu_memory_usage
import copy
from utils_main import (compute_average_activation, compute_freeze_and_drop,compute_stable_neurons,compute_average_activation_old)
from utils import getconst

def main():

    args = get_parser()
    mylogger = get_logger(args.log_path)

    log_hyperparameters(args, mylogger)
    if args.seed is not None:
        set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_id}")
    else:
        device = torch.device("cpu")
    if args.half_iid:
        args.task_num = ((args.n_tasks+1) // 2) + 1
    else:
        args.task_num = args.n_tasks
    args.temporal_dim = getconst[args.dataset].temporal_dim
    args.channel_dim = getconst[args.dataset].channel_dim
    args.class_num = getconst[args.dataset].class_num

    args.device = device
    train_loader, test_loader = get_data(args)

    eval_accs = []

    model = HARTrans(args)

    model = model.to(device)
    model.train() 

    agent = Method(model, args,mylogger,device)
    print_model_parameters(model, mylogger)
    log_gpu_memory_usage(device, mylogger, step=f"init model")

    eval_accs,best_accs = train(
        args,
        agent=agent,
        train_loader=train_loader,
        eval_loader=test_loader,
        device=device,
        mylogger=mylogger
    )


    log(eval_accs,best_accs,mylogger)


def train(args, agent, train_loader, eval_loader, device,mylogger):
    eval_accs = []
    best_accs = []
    start_task = 0
    if args.half_iid:
        start_task = (args.n_tasks // 2) - 1 
    freeze_masks = None
    stable_indices = None
    activation_old = None
    agent.model.prototypes = getconst[args.dataset].compute_prototypes(args.n_tasks,args.half_iid, train_loader)
    for task in range(start_task, args.n_tasks):
        # set task
        train_loader.sampler.set_task(task, sample_all_seen_tasks=(task == start_task),first_task=(task == start_task))
        agent.train()
        start = time.time()
        n_epochs = args.n_epochs
        if task == start_task:
            n_epochs += args.n_warmup_epochs
        agent.on_task_start(task,start_task)
        mylogger.info("\n>>> Task #{} --> Model Training".format(task+1-args.n_tasks//2))
        bestacc = -1.0
        for epoch in range(n_epochs):
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                inc_data = {"x": x, "y": y}
                loss = agent.observe(inc_data,freeze_masks=freeze_masks)
                print(
                    f"Epoch: {epoch + 1} / {n_epochs} | {i+1} / {len(train_loader)} - Loss: {loss}",
                    end="\r",
                )
            if (epoch + 1) % 1 == 0 or (epoch + 1 == n_epochs):
                mylogger.info('Task {}. Time {:.2f}'.format(task,time.time() - start))
                accs,acc = agent.eval_agent(eval_loader, task)
                eval_accs += [accs]
                if acc > bestacc:
                    bestacc = acc
                    epoch_best_model = agent.model.state_dict()  
                agent.train()

        best_accs.append(round(bestacc,2))
        agent.model.load_state_dict(epoch_best_model)  
        agent.on_task_finish(task,start_task)

        log_gpu_memory_usage(device, mylogger, step=f"End training-{task}")

        # At the end of each task, calculate the stable neurons of the mlp layer and freeze them
        activation = compute_average_activation(agent.model, train_loader,device)
        if task >=start_task+1:
            activation_old = compute_average_activation_old(agent.model,agent.premodel, train_loader,device)
        np.set_printoptions(threshold=np.inf)
        stable_indices = compute_stable_neurons(activation,activation_old,in_channels=getconst[args.dataset].neurons_in_channels, activation_perc = 16,stable_indices_old=stable_indices)
        for i,indices in enumerate(stable_indices):
            mylogger.info('x{}. indices {}'.format(i, indices))

        freeze_masks = compute_freeze_and_drop(stable_indices, agent.model)

    

    
    return eval_accs,best_accs



def log(eval_accs,best_accs,mylogger):
    # ----- Final Results ----- #
    accs = np.stack(eval_accs).T
    avg_acc = accs[:, -1].mean()
    mylogger.info('\nFinal Results')
    mylogger.info('Acc:{}- Avg Acc:{}'.format(best_accs,round(sum(best_accs)/len(best_accs), 2)))
    


if __name__ == "__main__":
    main()
