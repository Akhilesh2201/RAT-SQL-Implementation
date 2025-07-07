# Dataset Preprocessing

import torch
from torch.utils.data import Dataset
from torch.nn.functional import pad
from typing import List, Dict, Tuple, Any
from ratsql.graph import get_relations

class SpiderMiniDataset(Dataset):
    def __init__(self, data: List[Dict], schema_dict: Dict[str, Dict], db_dir: str):
        self.examples = data
        self.schema_dict = schema_dict
        self.db_dir = db_dir

    def __getitem__(self, idx: int) -> Tuple[str, Dict, str, str]:
        ex = self.examples[idx]
        schema = self.schema_dict[ex['db_id']]
        graph = get_relations(ex['question'], schema, self.db_dir)
        return ex['question'], graph, ex['query'], ex['db_id']

    def __len__(self) -> int:
        return len(self.examples)


def build_relation_matrix(graph_info: Dict, num_relations: int = 10) -> torch.Tensor:
    rel_types = {
        "match": 1,
        "belongs_to": 2,
        "value_match": 3,
        "same_table": 4,
        "foreign_key_forward": 5,
        "foreign_key_backward": 6,
        "primary_key": 7
    }
    size = len(graph_info['tokens'])
    mat = torch.zeros(size, size, dtype=torch.long)
    for (i, j), rel in graph_info['edges'].items():
        mat[i, j] = rel_types.get(rel, 0)
    return mat


def collate_fn(batch: List[Tuple[str, Dict, str, str]]) -> Dict[str, Any]:
    questions, graphs, queries, db_ids = zip(*batch)
    max_len = max(len(g['tokens']) for g in graphs)

    tokens = [g['tokens'] for g in graphs]

    rel_mats = [build_relation_matrix(g) for g in graphs]
    padded_rels = [
        pad(r, (0, max_len - r.size(0), 0, max_len - r.size(0)), value=0)
        for r in rel_mats
    ]
    rel_mats = torch.stack(padded_rels)

    return {
        "tokens": tokens,
        "rel_mat": rel_mats,
        "graphs": graphs,
        "queries": queries,
        "db_ids": db_ids
    }

__all__ = ["SpiderMiniDataset", "build_relation_matrix", "collate_fn"]
