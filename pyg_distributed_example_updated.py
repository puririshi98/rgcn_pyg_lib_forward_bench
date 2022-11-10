import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from torch_geometric.datasets import FakeDataset
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import RGCNConv


def run(rank, world_size, dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]

    train_loader = (LinkNeighborLoader(data, num_neighbors=[25, 10], batch_size=128))


    torch.manual_seed(12345)
    model = RGCNConv(dataset.num_features, dataset.num_classes, 1).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    for epoch in range(1, 21):
        model.train()

        for i, batch in enumerate(train_loader):
            e_idx = batch.edge_index.to(rank)
            x = batch.x.to(rank)
            etype = torch.zeros(e_idx.shape[-1]).to(rank)
            out = model(x, e_idx.long(), etype.long())
        if rank==0:
            print("finished step", i)
        dist.barrier()
        if rank == 0:
            print('finished epoch', epoch)




    dist.destroy_process_group()


if __name__ == '__main__':
    dataset = FakeDataset()

    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
