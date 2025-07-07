# Training Loop

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import gc

from ratsql.encoder import EmbeddingEncoder, RATEncoder
from ratsql.decoder import SQLTreeDecoder
from ratsql.dataset import SpiderMiniDataset, collate_fn
from ratsql.graph import get_relations, extract_labels_from_sql, get_values_from_db
from ratsql.eval import evaluate_execution_accuracy, evaluate_exact_match
from ratsql.utils import load_json, build_value_vocab, schema_dict, db_dir, train_data, dev_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()

encoder_model = EmbeddingEncoder().to(device)
rat_encoder = RATEncoder(input_dim=encoder_model.output_dim).to(device)
decoder = SQLTreeDecoder(
    encoder_dim=256,
    hidden_dim=256,
    num_values=100
).to(device)

train_dataset = SpiderMiniDataset(train_data, schema_dict, db_dir)
dev_dataset = SpiderMiniDataset(dev_data, schema_dict, db_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

value_vocab = build_value_vocab(schema_dict, db_dir, max_size=100)
value_vocab_inv = {v: k for k, v in value_vocab.items()}

optimizer = torch.optim.AdamW(
    list(encoder_model.parameters()) +
    list(rat_encoder.parameters()) +
    list(decoder.parameters()), lr=2e-5
)
scaler = GradScaler()

best_acc = 0.0
patience = 3
epochs_no_improve = 0
max_epochs = 20

acc = evaluate_execution_accuracy(decoder, encoder_model, rat_encoder, dev_dataset, db_dir)
em = evaluate_exact_match(decoder, encoder_model, rat_encoder, dev_dataset, db_dir)
print(f"[Init] Dev Exec: {acc:.2%}, Exact Match: {em:.2%}")


for epoch in range(max_epochs):
    encoder_model.train(); rat_encoder.train(); decoder.train()
    print(f"\n Epoch {epoch + 1}/{max_epochs}")

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            batch['x_embed'] = batch['x_embed'].float()
            x_all = rat_encoder(batch['x_embed'], batch['rel_mat'])
            loss_sum = 0.0

            for i in range(x_all.size(0)):
                try:
                    graph = batch['graphs'][i]
                    db_id = batch['db_ids'][i]
                    schema = schema_dict[db_id]
                    graph['value_vocab_inv'] = value_vocab_inv

                    labels = extract_labels_from_sql(batch['queries'][i], schema, value_vocab)
                    x = x_all[i:i+1]
                    loss = decoder.forward_supervised(x, graph, labels)
                    if loss is not None and torch.isfinite(loss):
                        loss_sum += loss
                except Exception as e:
                    print(f"[Skip] Step {step}, Ex {i}: {e}")
                    continue

        if loss_sum == 0.0:
            continue

        scaler.scale(loss_sum).backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()

        if step % 10 == 0:
            print(f"🔹 Step {step} - Loss: {loss_sum.item():.4f}")

    acc = evaluate_execution_accuracy(decoder, encoder_model, rat_encoder, dev_dataset, db_dir)
    em = evaluate_exact_match(decoder, encoder_model, rat_encoder, dev_dataset, db_dir)
    print(f"Epoch {epoch + 1} — Dev Exec: {acc:.2%}, Exact Match: {em:.2%}")

    if acc > best_acc:
        best_acc = acc
        epochs_no_improve = 0
        print("✅ New best model — saving checkpoint.")
        torch.save({
            'encoder': encoder_model.state_dict(),
            'rat': rat_encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict()
        }, "./checkpoints/rat_sql_best.pt")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(" Early stopping in ", patience, " epochs.")
            break
