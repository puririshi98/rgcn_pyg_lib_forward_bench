from torch_geometric.data import Data
from typing import Optional
import random
#implement a DocTransformer in nn.text
def get_wiki_data(question: str, model:DocTransformer, seed_nodes: int = 3, fan_out: int = 3, num_hops: int = 2, answer: Optional[str] = None) -> Data:
	import wikipedia
	search_list = wikipedia.search(question)
	seed_doc_names = search_list[:seed_nodes]
	seed_docs = [wikipedia.page(doc_name) for doc_name in seed_doc_names]
	# initialize our doc graph with seed docs
	doc_contents = [doc.content for doc in seed_docs]
	title_2_node_id_map = {doc.title:i for doc in enumerate(seed_docs)}
	# do neighborsampling and create graph
	src_n_ids = []
	dst_n_ids = []
	for hop in num_hops:
		next_hops_seed_docs = []
		for src_doc in seed_docs:
			full_fan_links = src_doc.links
			randomly_chosen_neighbor_links = list(random.sample(full_fan_links, k=fan_out))
			new_seed_docs = [wikipedia.page(link) for link in randomly_chosen_neighbor_links]
			for dst_doc in new_seed_docs
				dst_doc_title = dst_doc.title
				if dst_doc_title not in title_2_node_id_map:
					# add new node to graph
					title_2_node_id_map[dst_doc_name] = len(title_2_node_id_map)
					doc_contents.append(dst_doc.content)
				next_hops_seed_docs.append(doc)
				src_n_ids.append(title_2_node_id_map[src_doc.title])
				dst_n_ids.append(title_2_node_id_map[dst_doc.title])

		# root nodes for the next hop
		seed_docs = next_hops_seed_docs

	# put docs into model
	embedded_docs = model(doc_contents)
	del doc_contents

	# create node features, x
	x = torch.cat(embedded_docs)


	# construct and return Data object
	return Data(x=x, edge_index=torch.tensor([src_n_ids, dst_n_ids]), n_id=torch.tensor(title_2_node_id_map.values()), question=question, answer=answer).to("cpu")


# create SQUAD_WikiGraph dataset by calling wikiloader for each SQUAD question



