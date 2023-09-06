# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
from functools import partial

import numpy as np
import torch

from gp.data_loaders import DataSpec, LPDataObject, MetaLoader
from gp.downstream import GPInferenceCluster, GPTrainCluster
from gp.metrics.metrics import BinaryAccuracy
from gp.models.input_module import InputFeatureModule
from gp.models.node_input_layer import NodeFeatureModule
from gp.models.pyg import HeteroModule, TorchNodeEmbedding
from gp.models.pyg.decoders.edge_dot_product import EdgeDotProductDecoder
from gp.models.pyg.predictor_heads.link_predictor import LinkPredictor
from gp.preprocessing import Preprocessing
from gp.preprocessing.trans_dataset import Tabformer
from gp.workflow import Workflow
from gp.workflow.trainers import Trainer
from gp.workflow.trainers.optimizer import OptWrapper

DATA_DIR = "/data/"


def test_deployment_pyg():
    torch.cuda.empty_cache()
    tmpdir = "/data/" + ("tabformer/")
    source_path = tmpdir + "card_transaction.v1.csv"
    tmpdir = "/workspace/data/" + ("tabformer/")
    destination_path = os.path.join(tmpdir, "GP_Transformed/")
    if os.path.exists(destination_path):
        shutil.rmtree(destination_path)
    prep = Preprocessing(Tabformer(source_path, destination_path))
    prep.transform()
    meta = MetaLoader(destination_path)

    fan = 5
    train_spec = DataSpec(
        shuffle=True,
        batch_size=8192,
        fanouts=[fan, fan],
        metadata=meta,
    )
    test_spec = DataSpec(
        shuffle=False,
        batch_size=8192,
        fanouts=[fan, fan],
        metadata=meta,
    )

    data_object = LPDataObject(
        train_dataloader=train_spec,
        valid_dataloader=test_spec,
        test_dataloader=test_spec,
        data_path=destination_path,
        target="transaction",
        split_name="split",
        use_pandas=True,
        use_uva=True,
        backend="PyG",
        save=False,
    )
    graph = data_object.construct_cache["graph"]
    metadata = data_object.metadata
    EMBEDDING_DIM = 64

    node_layer = NodeFeatureModule(
        metadata, graph, node_id_embedding=EMBEDDING_DIM, backend="PyG"
    )
    input_layer = InputFeatureModule(node_module=node_layer)
    # emb = TorchNodeEmbedding(graph, EMBEDDING_DIM)
    # model = HeteroModule(
    #     metadata=metadata,
    #     dim_hidden=EMBEDDING_DIM,
    #     dim_out=64,
    #     n_layers=2,
    #     embedding=emb,
    # )

    # model = HeteroModule(
    #     metadata,
    #     emb,
    #     [{edge: SAGEConv for edge in edges} for layer_index in range(2)],
    #     dim_hidden=EMBEDDING_DIM,
    #     dim_out=64,
    # )
    model = HeteroModule(
        dim_hidden=EMBEDDING_DIM,
        dim_out=64,
        n_layers=2,
        activation=torch.nn.ReLU(),
        input_layer=input_layer,
        add_reverse_edges=True,
    )

    link_model = LinkPredictor(
        encoder=model,
        targeted_etype=("card", "transaction", "merchant"),
        decoder=EdgeDotProductDecoder(),
    )
    opt1 = partial(torch.optim.Adam, lr=1e-2)

    opt2 = partial(torch.optim.SGD, lr=1e-2)
    sched2 = partial(
        torch.optim.lr_scheduler.CyclicLR, base_lr=1e-2, max_lr=7e-1, mode="triangular"
    )
    opt_object_1 = OptWrapper(model=input_layer, opt_partial=opt1)
    opt_object_2 = OptWrapper(model=model, opt_partial=opt2, scheduler_partial=sched2)
    optimizers = [opt_object_1, opt_object_2]

    trainer = Trainer(
        data_object=data_object,
        model=link_model,
        optimizers=optimizers,
        criterion=torch.nn.BCELoss(),
        n_gpus=1,
        epochs=1,
        metrics={"acc": BinaryAccuracy()},
        amp=False,
        output_dir="0000-00-00/01-00-00/",
        limit_batches=10,
    )

    wrk = Workflow(
        trainer=trainer,
    )
    wrk.fit()
    wrk.generate_embeddings_pyg()
    train_cluster = GPTrainCluster(world_size=1, device_pool=0.5)
    inference_cluster = GPInferenceCluster(world_size=1)
    import time
    ap = wrk.fit_xgboost(
        label="is_fraud",
        split="split",
        training_cluster=train_cluster,
        inference_cluster=inference_cluster,
        n_rounds=10,
    )
    wrk.deploy("/tmp/gp_deployment_outputs/", triton_batch_size=8192, downstream=True)
    wrk.launch_inference_server()
    since=time.time()
    b = wrk.pydep.run_inference_on_triton(dataloader=wrk.gp_xgb.triton_dataloader())
    print("time to run inference on triton:", time.time() - since)
    wrk.stop_inference_server()
    since=time.time()
    labels = np.concatenate(b["labels"])
    preds = np.concatenate(b["preds"])

    from sklearn.metrics import average_precision_score

    ap2 = average_precision_score(labels, preds)
    print(ap, ap2)
    print("time to score:", time.time() - since)
    torch.cuda.empty_cache()
