import os
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from datasets import load_from_disk
from utils import my_collate_fn

from config import MODEL_NAME, PREPROCESSED_DIR, DEVICE
from preprocess_data import process_data, SpanExtractionChunkedDataset, span_collate_fn
from models import GraphAugmentedNLIModel
from transformers import AutoConfig, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------
# 1) Define a BERT‐only baseline
# ---------------------
class BertOnlyNLIModel(nn.Module):
    def __init__(self, base_model_name: str, num_labels: int = 3):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        config.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(base_model_name, config=config)
        hidden_dim = config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_emb)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

# ---------------------
# 2) Training & evaluation routines
# ---------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, optimizer, scheduler):
    model.train()
    losses = []
    is_gnn = hasattr(model, "gnn_premise")  # True for GraphAugmentedNLIModel

    for batch in tqdm(loader, leave=False):
        optimizer.zero_grad()
        # Move all tensor fields to DEVICE
        batch = {
            k: v.to(DEVICE) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        if is_gnn:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                premise_graph_tokens=batch["premise_graph_tokens"],
                premise_graph_edges=batch["premise_graph_edges"],
                premise_node_indices=batch["premise_node_indices"],
                hypothesis_graph_tokens=batch["hypothesis_graph_tokens"],
                hypothesis_graph_edges=batch["hypothesis_graph_edges"],
                hypothesis_node_indices=batch["hypothesis_node_indices"],
                labels=batch["labels"],
            )
        else:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

        loss = out["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    preds, golds = [], []
    is_gnn = hasattr(model, "gnn_premise")

    for batch in loader:
        batch = {
            k: v.to(DEVICE) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        if is_gnn:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                premise_graph_tokens=batch["premise_graph_tokens"],
                premise_graph_edges=batch["premise_graph_edges"],
                premise_node_indices=batch["premise_node_indices"],
                hypothesis_graph_tokens=batch["hypothesis_graph_tokens"],
                hypothesis_graph_edges=batch["hypothesis_graph_edges"],
                hypothesis_node_indices=batch["hypothesis_node_indices"],
            )
        else:
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        logits = out["logits"].cpu().numpy()
        preds.extend(np.argmax(logits, axis=1).tolist())
        golds.extend(batch["labels"].cpu().tolist())

    acc = accuracy_score(golds, preds)
    f1  = f1_score(golds, preds, average="macro")
    return acc, f1



# ---------------------
# 3) Ablation runner
# ---------------------
def run_ablation(
    epochs=3,
    batch_size=16,
    lr=2e-5,
    sample_frac=0.05,    # ← fraction of data to use
):
    set_seed()
    process_data()

    # --- Load the preprocessed SNLI dataset from disk ---
    snli = load_from_disk(PREPROCESSED_DIR)
    full_train = snli["train"]
    full_val   = snli["validation"]

    # --- Sample 10% of each split ---
    num_train = len(full_train)
    num_val   = len(full_val)
    n_train   = max(1, int(sample_frac * num_train))
    n_val     = max(1, int(sample_frac * num_val))

    # reproducible shuffling
    train_indices = list(range(num_train))
    random.shuffle(train_indices)
    train_subset = Subset(full_train, train_indices[:n_train])

    val_indices = list(range(num_val))
    random.shuffle(val_indices)
    val_subset = Subset(full_val, val_indices[:n_val])

    # --- Build DataLoaders with the SNLI collate_fn ---
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=my_collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 4) Define models
    models = {
        "Baseline-BERT": BertOnlyNLIModel(MODEL_NAME).to(DEVICE),
        "GNN-Augmented": GraphAugmentedNLIModel(
            base_model_name=MODEL_NAME,
            num_labels=3,
            hidden_dim=768,
            gnn_dim=128
        ).to(DEVICE),
    }

    results = {}
    for name, model in models.items():
        logging.info(f"--- Training {name} on {sample_frac*100:.0f}% of data ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_steps = epochs * len(train_loader)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=total_steps
        )

        # training loop
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)
            logging.info(f"{name} Epoch {epoch}: train_loss={train_loss:.4f}")

        # evaluation
        acc, f1 = evaluate(model, val_loader)
        logging.info(f"{name} on {sample_frac*100:.0f}% val → acc={acc:.4f}, f1={f1:.4f}")
        results[name] = {"accuracy": acc, "f1": f1}

    # 5) Plot
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    f1s   = [results[n]["f1"] for n in names]

    plt.figure()
    plt.bar(names, accs)
    plt.xlabel("Model")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Ablation on {sample_frac*100:.0f}% Data: Accuracy")

    plt.figure()
    plt.bar(names, f1s)
    plt.xlabel("Model")
    plt.ylabel("Validation Macro-F1")
    plt.title(f"Ablation on {sample_frac*100:.0f}% Data: Macro-F1")

    plt.show()


if __name__ == "__main__":
    run_ablation(epochs=5, batch_size=16, lr=5e-3, sample_frac=0.1)
