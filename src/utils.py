import subprocess
from typing import List, Tuple

import networkx as nx
import numpy as np
import torch
from gensim.models import KeyedVectors
from tqdm import tqdm


# From PyG utils
def to_networkx(
    data, node_attrs=None, edge_attrs=None, to_undirected=False, remove_self_loops=False
) -> nx.Graph:
    if to_undirected:
        G = nx.Graph()
    else:
        G = nx.DiGraph()
    G.add_nodes_from(range(data.num_nodes))
    node_attrs, edge_attrs = node_attrs or [], edge_attrs or []
    values = {}
    for key, item in data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]
    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        if to_undirected and v > u:
            continue
        if remove_self_loops and u == v:
            continue
        G.add_edge(u, v)
        for key in edge_attrs:
            G[u][v][key] = values[key][i]
    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    return G


def add_node_emb(data, embed):
    data_with_node_emb = None
    for idx, text_emb in enumerate(data.x):
        node_emb = embed.get_vector(str(idx))
        concatenated_vec = np.concatenate((text_emb, node_emb))

        if data_with_node_emb is None:
            data_with_node_emb = concatenated_vec
        else:
            data_with_node_emb = np.vstack((data_with_node_emb, concatenated_vec))

    return data_with_node_emb


def aggregate_node_emb_to_dataset(dataset) -> Tuple[List, List]:
    x_dataset = []
    y_dataset = []

    for data_sample in tqdm(dataset):
        G = to_networkx(data_sample)
        nx.write_adjlist(G, "g.adjlist")

        subprocess.run(
            [
                "deepwalk",
                "--input",
                "g.adjlist",
                "--output",
                "g.embeddings",
                "--workers",
                "16",
            ]
        )

        node_embed = KeyedVectors.load_word2vec_format("g.embeddings")
        data_with_node_emb = add_node_emb(data_sample, node_embed)

        x_dataset.append(data_with_node_emb)
        y_dataset.append(data_sample.y)

    return x_dataset, y_dataset
