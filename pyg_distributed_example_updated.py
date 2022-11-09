import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import Reddit
from torch_geometric.sampler import NeighborSampler
from torch_geometric.nn import RGCNConv


def run(rank, world_size, dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]
    train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    train_idx = train_idx.split(train_idx.size(0) // world_size)[rank]

    train_loader = (NeighborSampler(data, num_neighbors=[25, 10]))


    torch.manual_seed(12345)
    model = RGCNConv(dataset.num_features, dataset.num_classes, 1).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    x, y = data.x.to(rank), data.y.to(rank)

    for epoch in range(1, 21):
        model.train()

        for i in range(10):
            samp = train_loader._sample(torch.tensor([i]))
            e_idx = torch.cat((samp.row.reshape(1, -1), samp.col.reshape(1, -1)))

            optimizer.zero_grad()
            out = model(x, e_idx.long(), torch.zeros(e_idx.shape[-1]).long())

        dist.barrier()
        if rank == 0:
            print('finished epoch', epoch)




    dist.destroy_process_group()


if __name__ == '__main__':
    dataset = Reddit('../../data/Reddit')

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
