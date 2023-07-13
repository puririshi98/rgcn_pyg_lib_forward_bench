import cudf
import torch_geometric
import torch
import cupy
# scale up to see when it breaks
for i in range(2, 20):
  print("trying w/ num nodes = 10^" + str(i))
  max_index = 10**i
  edge_index = torch.randint(low=int(max_index/2), high=max_index, size=(2, int(10**6)))
  
  # edges
  
  graph = cudf.DataFrame(
      {
          "u": cupy.asarray(edge_index[0].reshape(-1)),
          "v": cupy.asarray(edge_index[1].reshape(-1)),
          
      }
  )
  nodes_to_keep = cudf.Series([3, 4, 5], name="nodes")
  
  mask = graph.u.isin(nodes_to_keep) & graph.v.isin(nodes_to_keep)
  
  subgraph = graph.iloc[mask, :]
  
# Now relabel
# We have a mapping from the nodes_to_keep to a contiguous dense set
# and want to apply that mapping to the subgraph u and v. We can do
# this with three merges. The first two construct the relabellings
# separately for u and v, the final one regroups them (on the (u, v)
# pairs), since the two individual merges might not be done in the
# same order.

# The default index for the Series is a RangeIndex, so this turns it
# into a dataframe with two columns, "index" (the new labels) and
# "nodes" (the old labels)
nodes_to_keep = nodes_to_keep.reset_index()

new_u = (
    subgraph.merge(nodes_to_keep, left_on="u", right_on="nodes", how="inner")
    .drop("nodes", axis=1)
    .rename({"index": "new_u"}, axis=1)
)
new_v = (
    subgraph.loc[:, ["u", "v"]]
    .merge(nodes_to_keep, left_on="v", right_on="nodes", how="inner")
    .drop("nodes", axis=1)
    .rename({"index": "new_v"}, axis=1)
)

relabeled_subgraph = new_u.merge(
    new_v, left_on=["u", "v"], right_on=["u", "v"], how="inner"
).drop(["u", "v"], axis=1)
