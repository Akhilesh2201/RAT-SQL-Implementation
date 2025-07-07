# Encoder(Transformer + Relation-Aware Attention)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from typing import List, Tuple

# Make device configurable from outside
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingEncoder(nn.Module):
    def __init__(self, model_name='microsoft/deberta-v3-base'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.output_dim = self.transformer.config.hidden_size

    def forward(self, word_lists: List[List[str]]) -> Tuple[torch.Tensor, None]:
        tokenized = self.tokenizer(
            word_lists,
            padding=True,
            is_split_into_words=True,
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True
        )

        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        word_embeddings = []
        for i in range(len(word_lists)):
            word_ids = tokenized.word_ids(batch_index=i)
            grouped = defaultdict(list)
            for j, wid in enumerate(word_ids):
                if wid is not None:
                    grouped[wid].append(last_hidden[i, j])
            avg_embeds = [torch.stack(group).mean(0) for _, group in sorted(grouped.items())]
            word_embeddings.append(torch.stack(avg_embeds))

        max_len = max(x.size(0) for x in word_embeddings)
        padded = torch.stack([
            F.pad(x, (0, 0, 0, max_len - x.size(0))) for x in word_embeddings
        ])
        return padded.to(device), None


class RATEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1, num_relations=10):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.rel_k = nn.Embedding(num_relations, dim)
        self.rel_v = nn.Embedding(num_relations, dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, rel_mat: torch.Tensor) -> torch.Tensor:
        B, L, D = x.size()
        rel_kv = self.rel_k(rel_mat)
        rel_vv = self.rel_v(rel_mat)

        q = x
        k = x.unsqueeze(2) + rel_kv
        v = x.unsqueeze(2) + rel_vv

        k = k.view(B * L, L, D)
        v = v.view(B * L, L, D)
        q = q.view(B * L, 1, D)

        out, _ = self.attn(q, k, v)
        out = out.view(B, L, D)

        x = self.norm1(x + self.dropout(out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class RATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_relations=10):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            RATEncoderLayer(hidden_dim, num_heads=8, dropout=0.1, num_relations=num_relations)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, rel_mat: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, rel_mat)
        return x
