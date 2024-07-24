from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
import torch
import datasets
dataset = PygGraphPropPredDataset(name="ogbg-code2")
raw_dataset = datasets.load_dataset("claudios/code_search_net")
raw_dataset = datasets.concatenate_datasets([raw_dataset["train"], raw_dataset["validation"], raw_dataset["test"]])
def get_raw_python(func_name_tokens):
	# the ordering of code_search_net does not match ogbg-code2
	# have to search for matching python
	# (TODO) figure most efficient way to get the matching python (currently brute forcing)
	max_hits = 0
	max_hit_data_pt = None
	for raw_data_pt in raw_dataset:
		func_name = raw_data_pt["func_name"].split('.')
		if func_name == ''.join(func_name_tokens):
			return raw_data_pt["whole_func_string"]
	# if we can't find the function just return empty string
	return ""

new_set = []
for i in range(len(dataset)):
	old_obj = dataset[i]
	new_obj = Data()
	# combine all node information into a single feature tensor, let the GNN+LLM figure it out
	new_obj.x = torch.cat((old_obj.x, old_obj.node_is_attributed, old_obj.node_dfs_order, old_obj.node_depth))
	# extract raw python function for use by LLM 
	func_name_tokens = old_obj.y
	new_obj.desc = get_raw_python(func_name_tokens)
	if new_obj.desc == "":
		# skip data points with no raw data to match
		continue
	# extract other data needed for GNN+LLM
	new_obj.y = old_obj.y
	new_obj.edge_index = old_obj.edge_index
	new_obj.num_nodes = old_obj.num_nodes
	new_set.append(new_obj)
	del old_obj
	
