from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
import torch
dataset = PygGraphPropPredDataset(name="ogbg-code2")
unique_dict = {}

new_set = []
for data_pt in dataset:
	key = '_'.join(data_pt.y)
	if key in unique_dict.keys():
		unique_dict[key] += 1
	else:
		unique_dict[key] = 1

print("all([i == 1 for i  in unique_dict.values()]) =", all([bool(i == 1) for i  in unique_dict.values()]))
for key in unique_dict.keys():
	if unique_dict[key] != 1:
		print(key, "->", unique_dict[key])
