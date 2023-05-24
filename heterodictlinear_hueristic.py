import torch
from torch_geometric.nn.dense import Linear, HeteroDictLinear
import time
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
fused_times = {}
loop_times = {}
try:
  for num_nodes_per_type in [10**2,10**3,10**4,10**5]:
    for out_feats in [2, 4,8,16,32,64,128,256]:
      for n_feats in [4,8,16,32,64,128,256,512]:
        for num_types in [4, 8, 16, 32, 64, 128, 256, 512]:
          try:
            if n_feats < out_feats:
              continue
            print("benchmarking", num_types,"types w/", num_nodes_per_type, "nodes per type and", n_feats, "input features and", out_feats, "outuput feats")
            x_dict = {'v'+str(i):torch.randn((num_nodes_per_type, n_feats)).cuda() for i in range(num_types)}
            lin = Linear(n_feats, out_feats).cuda()
            heterolin = HeteroDictLinear(n_feats, out_feats, list(x_dict.keys())).cuda()
            for i in range(60):
                if i==10:
                    since=time.time()
                heterolin(x_dict)
            key = (num_types, num_nodes_per_type, n_feats, out_feats)
            fused_times[key] = ((time.time()-since)/50.0)
            print("Avg time for dict based=", fused_times[key])
            for i in range(60):
                if i==10:
                    since=time.time()
                o = {}
                for j in range(num_types):
                    node_type = 'v'+str(j)
                    o[node_type] = lin(x_dict[node_type])
            loop_times[key] = ((time.time()-since)/50.0)
            print("Avg time for for-loop=", loop_times[key])
          except:
            continue
except:
  print("Loop Times:", loop_times)
  print("Fused Times:", fused_times)




print("Loop Times:", loop_times)
print("Fused Times:", fused_times)

print("done timing, now will learn hueristic w/ sklearn")
import numpy as np
X = np.zeros((len(loop_times), 4))
y = np.zeros(len(loop_times))
for i, key in enumerate(loop_times.keys()):
  X[i, :] = key
  loop_time, fused_time = loop_times[key], fused_times[key]
  y[i] = int(fused_time <= loop_time)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
scaler = StandardScaler()
svm = LinearSVC()
clf = make_pipeline(scaler, svm)
clf.fit(X, y)

print("scaler mean=", scaler.mean_)
print("scaler scale=", scaler.scale_)
print("svm weights=", svm.coef_)
print("svm bias=", svm.intercept_)
