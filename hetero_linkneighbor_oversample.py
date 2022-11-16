import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch_geometric.datasets import FakeHeteroDataset
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric


def run(rank, world_size, dataset):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    data = dataset[0]
    print("Full graph:", data)

    e1 = data.edge_types[0]
    train_loader = (LinkNeighborLoader(data, edge_label_index=(e1, (data[e1].edge_index)), num_neighbors=[50, 50], batch_size=1024, num_workers=int(len(os.sched_getaffinity(0)) / 2)))


    torch.manual_seed(12345)


    for i, batch in enumerate(train_loader):
        if rank == 0:
            print('Minibatch:', batch)
        batch.to('cuda:'+str(rank))
        for etype in data.edge_types:
            batch_e_dim = batch[etype].edge_index.shape[1]
            fullgraph_e_dim = data[etype].edge_index.shape[1]
            assert batch_e_dim < fullgraph_e_dim, 'batch is bigger than full graph: ' + str((batch_e_dim)) + ' > ' + str(fullgraph_e_dim)
        break





    dist.destroy_process_group()


if __name__ == '__main__':
    torch_geometric.seed_everything(42)
    dataset = FakeHeteroDataset(avg_num_nodes=50000)
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    mp.spawn(run, args=(world_size, dataset), nprocs=world_size, join=True)
