# Build Schema Graph + Link Question to Schema

import os
import re
import sqlite3
import nltk
from typing import List, Dict, Tuple
from collections import defaultdict

nltk.download('punkt')

def tokenize(text: str) -> List[str]:
    return nltk.word_tokenize(text.lower())

def extract_schema_elements(schema: Dict) -> Tuple[List[str], List[str]]:
    table_names = [t.lower() for t in schema['table_names_original']]
    column_names = []
    for table_id, col_name in schema['column_names_original']:
        if col_name == "*":
            continue
        full_col = f"{table_names[table_id]}.{col_name.lower()}" if table_id >= 0 else col_name.lower()
        column_names.append(full_col)
    return column_names, table_names

def get_values_from_db(db_id: str, schema: Dict, db_dir: str) -> Dict[str, List[str]]:
    value_dict = defaultdict(list)
    try:
        conn = sqlite3.connect(os.path.join(db_dir, db_id, f"{db_id}.sqlite"))
        cursor = conn.cursor()
        table_names = [t.lower() for t in schema['table_names_original']]

        for table_id, col_name in schema['column_names_original']:
            if col_name == "*" or table_id < 0:
                continue
            table = table_names[table_id]
            col = col_name.lower()
            try:
                cursor.execute(f"SELECT DISTINCT {col} FROM {table} LIMIT 50")
                values = [str(row[0]).lower() for row in cursor.fetchall() if row[0] is not None]
                value_dict[f"{table}.{col}"] = values
            except Exception:
                continue
        conn.close()
    except Exception as e:
        print(f"[DB Error] Failed to connect for {db_id}: {e}")
    return value_dict

def get_relations(question: str, schema: Dict, db_dir: str) -> Dict:
    q_tokens = tokenize(question)
    col_names, tab_names = extract_schema_elements(schema)
    schema_tokens = col_names + tab_names
    all_nodes = q_tokens + schema_tokens
    edge_types = {}

    try:
        value_dict = get_values_from_db(schema['db_id'], schema, db_dir)
        for i, q_tok in enumerate(q_tokens):
            for full_col, values in value_dict.items():
                if q_tok in values and full_col in schema_tokens:
                    j = schema_tokens.index(full_col)
                    edge_types[(i, len(q_tokens) + j)] = "value_match"
                    edge_types[(len(q_tokens) + j, i)] = "value_match"
    except Exception as e:
        print(f"[Linking] Value linking failed: {e}")

    for i, q_tok in enumerate(q_tokens):
        for j, s_tok in enumerate(schema_tokens):
            if q_tok in s_tok.split("_") or s_tok in q_tok:
                edge_types[(i, len(q_tokens) + j)] = "match"
                edge_types[(len(q_tokens) + j, i)] = "match"

    for i, col in enumerate(col_names):
        for j, tab in enumerate(tab_names):
            if col.startswith(tab + "."):
                ci = len(q_tokens) + i
                tj = len(q_tokens) + len(col_names) + j
                edge_types[(ci, tj)] = "belongs_to"
                edge_types[(tj, ci)] = "belongs_to"

    for i, col_i in enumerate(col_names):
        for j, col_j in enumerate(col_names):
            if i != j and col_i.split('.')[0] == col_j.split('.')[0]:
                ci, cj = len(q_tokens) + i, len(q_tokens) + j
                edge_types[(ci, cj)] = "same_table"

    for fk_pair in schema.get('foreign_keys', []):
        col1_idx, col2_idx = fk_pair
        if col1_idx < len(col_names) and col2_idx < len(col_names):
            c1 = len(q_tokens) + col1_idx
            c2 = len(q_tokens) + col2_idx
            edge_types[(c1, c2)] = "foreign_key_forward"
            edge_types[(c2, c1)] = "foreign_key_backward"

    for pk_idx in schema.get('primary_keys', []):
        if pk_idx < len(col_names):
            col = len(q_tokens) + pk_idx
            table_name = col_names[pk_idx].split('.')[0]
            if table_name in tab_names:
                tab = len(q_tokens) + len(col_names) + tab_names.index(table_name)
                edge_types[(col, tab)] = "primary_key"
                edge_types[(tab, col)] = "primary_key"

    return {
        "tokens": all_nodes,
        "edges": edge_types,
        "q_len": len(q_tokens),
        "schema_len": len(schema_tokens),
        "column_names": col_names,
        "table_names": tab_names
    }

__all__ = ["get_relations", "get_values_from_db", "tokenize"]
