# run w/ python -m cudf.pandas x.py for CUDF GPU acceleration
# 
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Data
import torch
import datasets
from tqdm import tqdm
import pandas as pd
import random
ogbg_dataset = PygGraphPropPredDataset(name="ogbg-code2")
raw_dataset = datasets.load_dataset("claudios/code_search_net", "python")
raw_dataset = datasets.concatenate_datasets([raw_dataset["train"], raw_dataset["validation"], raw_dataset["test"]])


def find_wierd_names(func_name_tokens):
	for i, func_name in enumerate(raw_dataset["func_name"]):
		# helper code to find wierd matches
		# since its not clear how to apply such complex search to pandas
		func_name = func_name.lower()
		func_name_tokens = ["pandas", "sql", "builder"]
		all_in = all([bool(token in func_name) for token in func_name_tokens])
		last_pos = func_name.find(func_name_tokens[0])
		all_in_order = True
		for token in func_name_tokens[1:]:
			cur_pos = func_name.find(token)
			all_in_order = last_pos < cur_pos
			if not all_in_order:
				break
			lost_pos = cur_pos
		matches = (all_in & all_in_order)
		if matches:
			return raw_dataset["whole_func_string"][i]
		else:
			ValueError("nothing found for func_name_tokens =", func_name_tokens)
			
def make_raw_data_frame():
	# Create a Data Frame with
	# column 1: "func_name"
	# column 2: "whole_func_string"
	filtered_func_names = []
	for i in raw_dataset["func_name"]:
		split_name = i.split('.')
		if len(split_name) > 1:
			filtered_func_names.append(split_name[-1].lower())
		else:
			filtered_func_names.append(i.lower())
	df = pd.DataFrame({'func_name':filtered_func_names, "whole_func_string":raw_dataset["whole_func_string"]})
	df.set_index('func_name', inplace=True)
	return df
df = make_raw_data_frame()
def get_raw_python_from_df(func_name_tokens):
	# the ordering of code_search_net does not match ogbg-code2
	# have to search for matching python
	func_name = df.index
	# (TODO, get this working)
	basic_matches = (func_name == ''.join(func_name_tokens).lower()) | (func_name == '_'.join(func_name_tokens).lower()) | (func_name == "_" + "_".join(func_name_tokens).lower())
	basic_matches = basic_matches | (func_name == "_" + ''.join(func_name_tokens).lower())
	basic_matches = basic_matches | (func_name == "_" + ''.join(func_name_tokens).lower() + "_")
	basic_matches = basic_matches | (func_name == "_" + '_'.join(func_name_tokens).lower() + "_")
	basic_matches = basic_matches | (func_name == "_" + '_'.join(func_name_tokens).lower() + "_")
	if len(func_name_tokens) > 1:
		basic_matches = basic_matches | (func_name == "_" + func_name_tokens[0].lower() + "_" + ''.join(func_name_tokens[1:]).lower())
		basic_matches = basic_matches | (func_name == ''.join(func_name_tokens[:-1]).lower() + "_" + func_name_tokens[-1].lower())
	matches = basic_matches
	result = df[matches]
	if len(result) > 0:
		"""
  		Randomly select one of the matches. We randomly select since we
    		have no idea how to break these ties. If this proves to be a blocker for SOTA results,
      		will come back to figure out how to do this best."""
		selected_result = result.sample()
		func_str = str(selected_result.iloc[0]["whole_func_string"])		
	else:
		func_str = find_wierd_names(func_name_tokens)
	return func_str[func_str.find("def"):(func_str.find(":"))], func_str[func_str.find('"""'):]

new_set = []
len_set = len(ogbg_dataset)
#print("num_data_pts =", len_set)
for i in tqdm(range(len_set)):
	old_obj = ogbg_dataset[i]
	new_obj = Data()
	# combine all node information into a single feature tensor, let the GNN+LLM figure it out
	new_obj.x = torch.cat((old_obj.x, old_obj.node_is_attributed, old_obj.node_dfs_order, old_obj.node_depth), dim=1)
	# extract raw python function for use by LLM 
	func_name_tokens = old_obj.y
	new_obj.func_signature, new_obj.desc  = get_raw_python_from_df(func_name_tokens)
	# extract other data needed for GNN+LLM
	new_obj.y = old_obj.y
	new_obj.edge_index = old_obj.edge_index
	new_obj.num_nodes = old_obj.num_nodes
	new_set.append(new_obj)
	#print("new_obj =", new_obj)
	del old_obj
