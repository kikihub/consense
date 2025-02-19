import argparse
from pathlib import Path

def get_parser():
    # >>> Arguments <<<
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--dataset", type=str, default="mmfi")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half_iid", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=60)
    parser.add_argument("--n_warmup_epochs", type=int, default=0)
    parser.add_argument("--n_tasks", type=int, default=-1)

    parser.add_argument("--log_path", type=str, default="xxx/consense/log/")

    parser.add_argument("--lr", type=float, default=0.0001)

    args = parser.parse_args()

    return args
