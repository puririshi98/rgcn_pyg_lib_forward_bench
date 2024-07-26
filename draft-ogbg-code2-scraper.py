from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
import torch
import datasets
dataset = PygGraphPropPredDataset(name="ogbg-code2")
raw_dataset = datasets.load_dataset("claudios/code_search_net", "python")
raw_dataset = datasets.concatenate_datasets([raw_dataset["train"], raw_dataset["validation"], raw_dataset["test"]])
def get_raw_python(func_name_tokens):
	# the ordering of code_search_net does not match ogbg-code2
	# have to search for matching python
	# (TODO) figure most efficient way to get the matching python (currently brute forcing way too slow)
	max_hits = 0
	max_hit_data_pt = None
	for raw_data_pt in raw_dataset:
		func_name = raw_data_pt["func_name"].split('.')[-1]
		if func_name == ''.join(func_name_tokens) or func_name == '_'.join(func_name_tokens):
			func_str = raw_data_pt["whole_func_string"]
			# start at the comments, since including the def string is cheating
			return func_str[func_str.find('"""'):]
	# if we can't find the function just return empty string
	raise Error("nothing found for func_name_tokens =", func_name_tokens)

new_set = []
for i in range(len(dataset)):
	old_obj = dataset[i]
	print("Iter", i)
	print("old_obj =", old_obj)
	new_obj = Data()
	# combine all node information into a single feature tensor, let the GNN+LLM figure it out
	new_obj.x = torch.cat((old_obj.x, old_obj.node_is_attributed, old_obj.node_dfs_order, old_obj.node_depth), dim=1)
	# extract raw python function for use by LLM 
	func_name_tokens = old_obj.y
	new_obj.desc = get_raw_python(func_name_tokens)
	# extract other data needed for GNN+LLM
	new_obj.y = old_obj.y
	new_obj.edge_index = old_obj.edge_index
	new_obj.num_nodes = old_obj.num_nodes
	new_set.append(new_obj)
	print("new_obj =", new_obj)
	del old_obj
	
