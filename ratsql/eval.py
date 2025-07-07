import sqlite3
import os
from torch.utils.data import DataLoader
from ratsql.dataset import collate_fn


def normalize_sql(sql: str) -> str:
    return " ".join(sql.lower().strip().split())


def evaluate_execution_accuracy(decoder, encoder_model, rat_encoder, dataset, db_dir):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    correct = 0
    total = 0

    encoder_model.eval()
    rat_encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch in loader:
            try:
                x = rat_encoder(batch['x_embed'], batch['rel_mat'])
                graph = batch['graphs'][0]
                db_id = batch['db_ids'][0]

                graph['value_vocab_inv'] = getattr(decoder, 'value_vocab_inv', {})

                pred_sql = decoder(x, graph)
                gold_sql = batch['queries'][0]

                db_path = os.path.join(db_dir, db_id, f"{db_id}.sqlite")
                if not os.path.exists(db_path):
                    print(f"[Missing DB] {db_path}")
                    total += 1
                    continue

                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                cursor.execute(pred_sql)
                pred_result = cursor.fetchall()

                cursor.execute(gold_sql)
                gold_result = cursor.fetchall()

                if set(pred_result) == set(gold_result):
                    correct += 1
                conn.close()
            except Exception as e:
                print(f"[SQL Error] {e}")
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"Execution Accuracy: {correct}/{total} = {acc:.2%}")
    return acc


def evaluate_exact_match(decoder, encoder_model, rat_encoder, dataset, db_dir):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    exact = 0
    total = 0

    encoder_model.eval()
    rat_encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch in loader:
            try:
                x = rat_encoder(batch['x_embed'], batch['rel_mat'])
                graph = batch['graphs'][0]

                graph['value_vocab_inv'] = getattr(decoder, 'value_vocab_inv', {})
                pred_sql = decoder(x, graph)
                gold_sql = batch['queries'][0]

                if normalize_sql(pred_sql) == normalize_sql(gold_sql):
                    exact += 1
            except Exception as e:
                print(f"[Exact Match Error] {e}")
            total += 1

    score = exact / total if total > 0 else 0
    print(f"Exact Match Score: {exact}/{total} = {score:.2%}")
    return score
