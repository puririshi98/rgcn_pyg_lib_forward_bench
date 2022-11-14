import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import RGCNConv


def run(rank, world_size, dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]

    n1 = data.node_types[0]
    train_loader = (NeighborLoader(data,input_nodes=(n1, torch.arange(data[n1].num_nodes)), num_neighbors=[25, 10], batch_size=128, num_workers=5))


    torch.manual_seed(12345)


    for epoch in range(1, 21):

        for i, batch in enumerate(train_loader):
            batch = batch.to(rank)
        dist.barrier()
        if rank == 0:
            print('finished epoch', epoch)




    dist.destroy_process_group()


if __name__ == '__main__':
    dataset = FakeHeteroDataset()
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
