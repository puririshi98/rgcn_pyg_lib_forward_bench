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
	# (TODO) use fast pythonic search methods (currently brute forcing way too slow)
	max_hits = 0
	max_hit_data_pt = None
	for raw_data_pt in raw_dataset:
		func_name = raw_data_pt["func_name"].split('.')[-1].lower()
		basic_matches = func_name == ''.join(func_name_tokens).lower() or func_name == '_'.join(func_name_tokens).lower() or func_name == "_" + "_".join(func_name_tokens).lower()
		basic_matches = basic_matches or func_name == "_" + ''.join(func_name_tokens).lower()
		basic_matches = basic_matches or func_name == "_" + ''.join(func_name_tokens).lower() + "_"
		basic_matches = basic_matches or func_name == "_" + '_'.join(func_name_tokens).lower() + "_"
		if basic_matches:
			matches = True
		else:
			# slightly slower check, only do if basic matches not hit
			all_in = all([bool(token in func_name) for token in func_name_tokens)
			last_pos = func_name.find(func_name_tokens[0])
			for token in func_name_tokens[1:]:
				cur_pos = func_name.find(token)
				all_in_order = last_pos < cur_pos
				if not all_in_order:
					break
				lost_pos = cur_pos
			matches = basic_matches or (all_in and all_in_order)
		if matches:
			func_str = raw_data_pt["whole_func_string"]
			return func_str[4:(func_str.find(":"))], func_str[func_str.find('"""'):]
	raise ValueError("nothing found for func_name_tokens =", func_name_tokens)

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
	new_obj.func_signature, new_obj.desc  = get_raw_python(func_name_tokens)
	# extract other data needed for GNN+LLM
	new_obj.y = old_obj.y
	new_obj.edge_index = old_obj.edge_index
	new_obj.num_nodes = old_obj.num_nodes
	new_set.append(new_obj)
	print("new_obj =", new_obj)
	del old_obj
	
