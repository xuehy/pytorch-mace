import os
import torch as th
import torch.multiprocessing as mp
from model import NeuralCoord
from train import train
import my_optim
from params import args
from test import test

if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    th.manual_seed(args.seed)
    shared_model = NeuralCoord(args.n_pursuers,
                               args.n_states,
                               args.dim_action)
    shared_model.share_memory()
    optimizer = my_optim.SharedAdam(shared_model.parameters(), args.lr)
    optimizer.share_memory()
    processes = []
    p = mp.Process(target=test, args=(0, shared_model))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_process):
        p = mp.Process(target=train, args=(rank, shared_model, optimizer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
