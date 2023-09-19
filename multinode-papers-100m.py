import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel
from torchmetrics import Accuracy

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv
import warnings

warnings.filterwarnings("ignore")

# srun -l -N2 --ntasks-per-node=<n> --container-workdir=<example_folder> \
# --container-image=<insert_container_image_url> \
# python3 -u multinode_papers100m_draft.py --epochs 2 --no-test

# mine:
# srun -l -N2 --ntasks-per-node=2 --container-image=gitlab-master.nvidia.com:5005/dl/dgx/pyg:main-py3-devel python3 -u x.py --epochs 3 --no-test
def pyg_num_work():
    num_work = None
    if hasattr(os, "sched_getaffinity"):
        try:
            num_work = len(os.sched_getaffinity(0)) / 2
        except Exception:
            pass
    if num_work is None:
        num_work = os.cpu_count() / 2
    return int(num_work)

_LOCAL_PROCESS_GROUP = None

def create_local_process_group(num_workers_per_node):
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    assert world_size % num_workers_per_node == 0

    num_nodes = world_size // num_workers_per_node
    node_rank = rank // num_workers_per_node
    for i in range(num_nodes):
        ranks_on_i = list(range(i * num_workers_per_node, (i + 1) * num_workers_per_node))
        pg = dist.new_group(ranks_on_i)
        if i == node_rank:
            _LOCAL_PROCESS_GROUP = pg

def get_local_process_group():
    assert _LOCAL_PROCESS_GROUP is not None
    return _LOCAL_PROCESS_GROUP


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def run_train(device, data, world_size, model, epochs, batch_size, fan_out,
              split_idx, num_classes, monitor):
    local_group = get_local_process_group()
    loc_id = dist.get_rank(group=local_group)
    rank = torch.distributed.get_rank()
    os.environ['NVSHMEM_SYMMETRIC_SIZE'] = "107374182400"
    split_idx['train'] = split_idx['train'].split(
        split_idx['train'].size(0) // world_size, dim=0)[rank].clone()
    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[loc_id])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
                                 weight_decay=0.0005)
    train_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                  input_nodes=split_idx['train'],
                                  batch_size=batch_size,
                                  num_workers=pyg_num_work())
    if rank == 0:
        eval_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                     input_nodes=split_idx['valid'],
                                     batch_size=batch_size,
                                     num_workers=pyg_num_work())
        test_loader = NeighborLoader(data, num_neighbors=[fan_out, fan_out],
                                     input_nodes=split_idx['test'],
                                     batch_size=batch_size,
                                     num_workers=pyg_num_work())
    eval_steps = 100
    acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    if rank == 0:
        print("Beginning training...")
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i >= 10:
                start = time.time()
            batch = batch.to(device)
            batch.y = batch.y.to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 10 == 0:
                print("Epoch: " + str(epoch) + ", Iteration: " + str(i) +
                      ", Loss: " + str(loss))
                print(monitor.table({"batch":i}))
        if rank == 0:
            print("Average Training Iteration Time:",
                  (time.time() - start) / (i - 10), "s/iter")
            acc_sum = 0.0
            with torch.no_grad():
                for i, batch in enumerate(eval_loader):
                    if i >= eval_steps:
                        break
                    if i >= 10:
                        start = time.time()
                    batch = batch.to(device)
                    batch.y = batch.y.to(torch.long)
                    out = model(batch.x, batch.edge_index)
                    acc_sum += acc(out[:batch_size].softmax(dim=-1),
                                   batch.y[:batch_size])
            print(f"Validation Accuracy: {acc_sum/(i) * 100.0:.4f}%", )
            print("Average Inference Iteration Time:",
                  (time.time() - start) / (i - 10), "s/iter")
    if rank == 0:
        acc_sum = 0.0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(device)
                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)
                acc_sum += acc(out[:batch_size].softmax(dim=-1),
                               batch.y[:batch_size])
            print(f"Test Accuracy: {acc_sum/(i) * 100.0:.4f}%", )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--fan_out', type=int, default=50)
    parser.add_argument(
        "--ngpu_per_node",
        type=int,
        default="1",
        help="number of GPU(s) for each node for multi-gpu training,",
    )
    args = parser.parse_args()
    # setup multi node
    torch.distributed.init_process_group("nccl")
    nprocs = dist.get_world_size()
    create_local_process_group(args.ngpu_per_node)
    local_group = get_local_process_group()
    device_id = dist.get_rank(group=local_group) if dist.is_initialized() else 0
    torch.cuda.set_device(device_id)
    device = torch.device(device_id)
    # setup mem monitor
    all_pids = torch.zeros(dist.get_world_size(), dtype=torch.int64).to(device)
    all_pids[dist.get_rank()] = os.getpid()
    dist.all_reduce(all_pids)

    dataset = PygNodePropPredDataset(name='ogbn-papers100M')
    split_idx = dataset.get_idx_split()

    data = dataset[0]
    data.y = data.y.reshape(-1)
    model = GCN(dataset.num_features, args.hidden_channels,
                dataset.num_classes)
    print("Data =", data)
    print('Using', nprocs, 'GPUs...')
    run_train(device, data, world_size, model, args.epochs, args.batch_size,
                         args.fan_out, split_idx, dataset.num_classes, monitor)
