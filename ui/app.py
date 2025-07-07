# Gradio UI

import gradio as gr
import sqlite3
import os
import torch

from ratsql.graph import get_relations, build_relation_matrix
from ratsql.encoder import EmbeddingEncoder, RATEncoder
from ratsql.decoder import SQLTreeDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder_model = EmbeddingEncoder().to(device)
rat_encoder = RATEncoder(input_dim=encoder_model.output_dim).to(device)
decoder = SQLTreeDecoder(
    encoder_dim=256,
    hidden_dim=256,
    num_values=100
).to(device)

# Load trained weights 
checkpoint_path = "/content/drive/MyDrive/rat_sql_best.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    encoder_model.load_state_dict(checkpoint['encoder'])
    rat_encoder.load_state_dict(checkpoint['rat'])
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    print("Loaded checkpoint.")
else:
    print("No checkpoint found.")

value_vocab_inv = {i: str(i) for i in range(1, 101)}  

def extract_schema_from_sqlite(sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    schema = {
        'table_names_original': tables,
        'column_names_original': [],
        'db_id': 'custom'
    }

    for i, table in enumerate(tables):
        cursor.execute(f"PRAGMA table_info({table});")
        for col in cursor.fetchall():
            col_name = col[1]
            schema['column_names_original'].append((i, col_name))

    conn.close()
    return schema

def predict_sql_from_uploaded_db(question, db_file):
    schema = extract_schema_from_sqlite(db_file.name)
    graph = get_relations(question, schema, db_dir=os.path.dirname(db_file.name))
    graph['value_vocab_inv'] = value_vocab_inv
    tokens = graph['tokens']

    with torch.no_grad():
        x_embed, _ = encoder_model([tokens])
        rel_mat = build_relation_matrix(graph).unsqueeze(0).to(device)
        x = rat_encoder(x_embed, rel_mat)
        pred_sql = decoder(x, graph)

    return pred_sql

gr.Interface(
    fn=predict_sql_from_uploaded_db,
    inputs=[
        gr.Textbox(label="Enter your natural language question"),
        gr.File(label="Upload your SQLite (.sqlite) database")
    ],
    outputs=gr.Textbox(label="Generated SQL Query"),
    title="RAT-SQL Decoder (2025)",
    description="Upload a SQLite DB and ask a question. It will generate a SQL query using the RAT-SQL decoder."
).launch()
