# Decoder(Not AST) 

import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlparse
import re
from typing import Dict, List
from ratsql.graph import get_relations


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RULES = {
    'ROOT': ['sql'],
    'sql': ['select', 'from', 'where', 'group_by', 'having', 'order_by', 'limit', 'set_op'],
    'select': ['Select'],
    'from': ['From'],
    'where': ['None', 'Where'],
    'group_by': ['None', 'GroupBy'],
    'having': ['None', 'Having'],
    'order_by': ['None', 'OrderBy'],
    'limit': ['None', 'Limit'],
    'set_op': ['None', 'Union', 'Intersect', 'Except'],
    'cond': ['Eq', 'Gt', 'Lt', 'Ge', 'Le', 'Ne', 'And', 'Or', 'In', 'Like', 'Between', 'Exists', 'Not'],
    'val_unit': ['Column', 'Minus', 'Plus', 'Times', 'Divide'],
    'table_unit': ['Table', 'TableUnitSql'],
    'agg': ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
}

class SQLTreeDecoder(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, num_values=100):
        super().__init__()
        self.lstm = nn.LSTMCell(encoder_dim, hidden_dim)
        self.rule_classifiers = nn.ModuleDict({
            rule: nn.Linear(hidden_dim, len(constructors)) for rule, constructors in RULES.items()
        })
        self.column_pointer = nn.Linear(hidden_dim, encoder_dim)
        self.table_pointer = nn.Linear(hidden_dim, encoder_dim)
        self.value_generator = nn.Linear(hidden_dim, num_values)
        self.order_direction_classifier = nn.Linear(hidden_dim, 2)
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim

    def forward(self, encoder_output, graph, beam_size=5):
        h = encoder_output.mean(dim=1).squeeze(0)
        c = torch.zeros_like(h)
        h, c = self.lstm(h, (h, c))
        beam = self.decode_node_beam("ROOT", h, encoder_output, graph, beam_size=beam_size)
        return beam[0]["sql"]

    def forward_supervised(self, encoder_output, graph, labels):
        loss = 0.0
        h = encoder_output.mean(dim=1).squeeze(0)
        c = torch.zeros_like(h)
        h, c = self.lstm(h, (h, c))

        rule_logits = self.rule_classifiers['select'](h)
        loss += F.cross_entropy(rule_logits.unsqueeze(0), torch.tensor([0], device=h.device))

        for col_idx, agg_id in zip(labels['select_cols'], labels['select_aggs']):
            agg_logits = self.rule_classifiers['agg'](h)
            loss += F.cross_entropy(agg_logits.unsqueeze(0), torch.tensor([agg_id], device=h.device))

            col_query = self.column_pointer(h)
            col_scores = torch.matmul(col_query, encoder_output.squeeze(0).T)
            col_idx = min(col_idx, encoder_output.size(1) - 1)
            loss += F.cross_entropy(col_scores.unsqueeze(0), torch.tensor([col_idx], device=h.device))

            h, c = self.lstm(h, (h, c))

        rule_logits = self.rule_classifiers['from'](h)
        loss += F.cross_entropy(rule_logits.unsqueeze(0), torch.tensor([0], device=h.device))

        tab_query = self.table_pointer(h)
        tab_scores = torch.matmul(tab_query, encoder_output.squeeze(0).T)
        tab_idx = min(labels['tab_idx'], encoder_output.size(1) - 1)
        loss += F.cross_entropy(tab_scores.unsqueeze(0), torch.tensor([tab_idx], device=h.device))

        h, c = self.lstm(h, (h, c))

        if labels['conds']:
            rule_logits = self.rule_classifiers['where'](h)
            loss += F.cross_entropy(rule_logits.unsqueeze(0), torch.tensor([1], device=h.device))

            for cond in labels['conds']:
                op_logits = self.rule_classifiers['cond'](h)
                op_id = RULES['cond'].index(cond['op'].capitalize()) if cond['op'].capitalize() in RULES['cond'] else 0
                loss += F.cross_entropy(op_logits.unsqueeze(0), torch.tensor([op_id], device=h.device))

                col_query = self.column_pointer(h)
                col_scores = torch.matmul(col_query, encoder_output.squeeze(0).T)
                col_idx = min(cond['col'], encoder_output.size(1) - 1)
                loss += F.cross_entropy(col_scores.unsqueeze(0), torch.tensor([col_idx], device=h.device))

                val_logits = self.value_generator(h)
                loss += F.cross_entropy(val_logits.unsqueeze(0), torch.tensor([cond['val_id']], device=h.device))

                h, c = self.lstm(h, (h, c))

        return loss


    def decode_node_beam(self, rule, state, enc, graph, beam_size=5):
        if rule not in self.rule_classifiers:
            return [{"sql": f"--{rule}--", "score": 0.0}]

        logits = self.rule_classifiers[rule](state)
        probs = F.log_softmax(logits, dim=-1)
        k = min(beam_size, len(RULES[rule]), probs.size(-1))
        topk_probs, topk_indices = probs.view(-1).topk(k)

        candidates = []
        for i in range(k):
            pred_idx = topk_indices[i].item()
            score = topk_probs[i].item()
            if pred_idx >= len(RULES[rule]):
                continue
            constructor = RULES[rule][pred_idx]

            if constructor == 'Select':
                result = self.decode_select(state, enc, graph)
            elif constructor == 'From':
                result = self.decode_from(state, enc, graph)
            elif constructor == 'Where':
                result = self.decode_where(state, enc, graph)
            elif constructor == 'GroupBy':
                result = self.decode_group_by(state, enc, graph)
            elif constructor == 'Having':
                result = self.decode_having(state, enc, graph)
            elif constructor == 'OrderBy':
                result = self.decode_order_by(state, enc, graph)
            elif constructor == 'Limit':
                result = self.decode_limit(state)
            elif constructor in {'Union', 'Intersect', 'Except'}:
                result = self.decode_set_op(constructor, state)
            elif rule == 'ROOT':
                result = self.decode_node_beam('sql', state, enc, graph)[0]["sql"]
            elif rule == 'sql':
                clauses = []
                total_score = score
                for clause_rule in RULES['sql']:
                    if clause_rule not in self.rule_classifiers:
                        continue
                    clause_logits = self.rule_classifiers[clause_rule](state)
                    clause_probs = F.log_softmax(clause_logits, dim=-1)
                    clause_probs = clause_probs.view(-1)
                    clause_idx = torch.argmax(clause_probs).item()
                    clause_idx = min(clause_idx, len(RULES[clause_rule]) - 1)
                    clause_score = clause_probs[clause_idx].item()
                    clause_constructor = RULES[clause_rule][clause_idx]

                    total_score += clause_score
                    if clause_constructor != 'None':
                        sub_beam = self.decode_node_beam(clause_rule, state, enc, graph, beam_size=1)
                        if sub_beam and "sql" in sub_beam[0]:
                            clauses.append(sub_beam[0]["sql"])
                        else:
                            print(f"[Warning] Empty beam for clause '{clause_rule}' — skipping.")

                has_group_by = any("GROUP BY" in c for c in clauses)
                clauses = [c for c in clauses if not c.startswith("HAVING") or has_group_by]

                result = ' '.join(clauses)
                candidates.append({"sql": result, "score": total_score})
                return candidates
            else:
                result = f"--Unhandled {constructor}--"


            candidates.append({"sql": result, "score": score})

        return sorted(candidates, key=lambda x: -x["score"])[:beam_size]

    def decode_select(self, state, enc, graph, threshold=0.2):
        agg_logits = self.rule_classifiers['agg'](state)
        agg_probs = F.softmax(agg_logits, dim=-1)
        col_query = self.column_pointer(state).unsqueeze(0)
        enc_ = enc[0]
        col_scores = torch.matmul(col_query, enc_.T).squeeze(0)
        col_probs = F.softmax(col_scores, dim=-1).view(-1)
        selected = [i for i, p in enumerate(col_probs.detach().cpu().tolist()) if p > threshold]
        if not selected:
            selected = [torch.argmax(col_scores).item()]
        parts = []
        for i in selected:
            i = min(i, len(graph['column_names']) - 1)
            col = graph['column_names'][i]
            agg_idx = torch.argmax(agg_probs).item()
            agg_idx = min(agg_idx, len(RULES['agg']) - 1)
            agg = RULES['agg'][agg_idx]

            parts.append(f"{agg}({col})" if agg else col)
        return "SELECT " + ", ".join(parts)

    def decode_from(self, state, enc, graph):
        tab_query = self.table_pointer(state).unsqueeze(0)
        enc_ = enc[0]
        tab_scores = torch.matmul(tab_query, enc_.T).squeeze(0)
        tab_idx = torch.argmax(tab_scores).item()
        tab_idx = min(tab_idx, len(graph['table_names']) - 1)
        return f"FROM {graph['table_names'][tab_idx]}"

    def decode_where(self, state, enc, graph):
        col_query = self.column_pointer(state).unsqueeze(0)
        enc_ = enc[0]
        col_scores = torch.matmul(col_query, enc_.T).squeeze(0)
        col_idx = torch.argmax(col_scores).item()
        col_idx = min(col_idx, len(graph['column_names']) - 1)
        col = graph['column_names'][col_idx]
        op_logits = self.rule_classifiers['cond'](state)
        op_idx = torch.argmax(op_logits).item()
        op_idx = min(op_idx, len(RULES['cond']) - 1)
        op = RULES['cond'][op_idx]

        val_logits = self.value_generator(state)
        val_id = torch.argmax(val_logits).item()
        val = graph.get("value_vocab_inv", {}).get(val_id, f"val{val_id}")
        return f"WHERE {col} {op} '{val}'"


    def decode_group_by(self, state, enc, graph, threshold=0.2):
        col_query = self.column_pointer(state).unsqueeze(0)
        enc_ = enc[0]
        col_scores = torch.matmul(col_query, enc_.T).squeeze(0)
        col_probs = F.softmax(col_scores, dim=-1).view(-1)
        selected = [i for i, p in enumerate(col_probs.detach().cpu().tolist()) if p > threshold]
        if not selected:
            selected = [torch.argmax(col_scores).item()]
        cols = [graph['column_names'][i] if i < len(graph['column_names']) else f"col{i}" for i in selected]
        return "GROUP BY " + ", ".join(cols)

    def decode_having(self, state, enc, graph):
        col_query = self.column_pointer(state).unsqueeze(0)
        enc_ = enc[0]
        col_scores = torch.matmul(col_query, enc_.T).squeeze(0)
        col_idx = torch.argmax(col_scores).item()
        col_idx = min(col_idx, len(graph['column_names']) - 1)
        col = graph['column_names'][col_idx]
        val_logits = self.value_generator(state)
        val_id = torch.argmax(val_logits).item()
        val = graph.get("value_vocab_inv", {}).get(val_id, f"val{val_id}")
        return f"HAVING COUNT({col}) > '{val}'"

    def decode_order_by(self, state, enc, graph, threshold=0.2):
        col_query = self.column_pointer(state).unsqueeze(0)
        enc_ = enc[0]
        col_scores = torch.matmul(col_query, enc_.T).squeeze(0)
        col_probs = F.softmax(col_scores, dim=-1).view(-1)
        selected = [i for i, p in enumerate(col_probs.detach().cpu().tolist()) if p > threshold]
        if not selected:
            selected = [torch.argmax(col_scores).item()]

        dir_logits = self.order_direction_classifier(state)
        dir_probs = F.softmax(dir_logits, dim=-1)

        dir_idx = torch.argmax(dir_probs).item()
        dir_idx = min(dir_idx, len(['ASC', 'DESC']) - 1)
        direction = ['ASC', 'DESC'][dir_idx]

        return "ORDER BY " + ", ".join([
            f"{graph['column_names'][i]} {direction}" if i < len(graph['column_names']) else f"col{i} {direction}"
            for i in selected
        ])

    def decode_limit(self, state):
        return "LIMIT 10"

    def decode_set_op(self, constructor, state):
        return f"{constructor.upper()} SELECT ..."
    
__all__ = ["SQLTreeDecoder", "RULES"]
