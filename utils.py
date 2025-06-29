import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import sent_tokenize
from sklearn.metrics import accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict, load_from_disk
import spacy
import re
from tqdm.auto import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt
from torch.optim import AdamW
import pandas as pd
from typing import Optional, Tuple, List, Dict

from models import GraphAugmentedNLIModel, GraphAugmentedFinNLIModel
from preprocess_data import SpanExtractionChunkedDataset, process_data, chunk_transcript, span_collate_fn

# =============================
# Configuration Constants
# =============================
from config import MODEL_NAME, MAX_LENGTH, OVERLAP, PREPROCESSED_DIR, tokenizer, nlp

#MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
#MAX_LENGTH = 128
#OVERLAP = 32
LEARNING_RATE = 2e-5
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#PREPROCESSED_DIR = "preprocessed_snli"
MIXED_PRECISION = "fp16"

# label mapping
label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

# =============================
# Logging & Reproducibility
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================
# Tokenizer & NLP Model
# =============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
nlp = spacy.load("en_core_web_sm")

# =============================
# Dependency Graph Helpers
# =============================
def build_dependency_graph(sentence: str):
    doc = nlp(sentence)
    tokens = [token.text for token in doc]
    edges = []
    for token in doc:
        if token.head.i != token.i:
            edges.append((token.i, token.head.i))
            edges.append((token.head.i, token.i))
    return tokens, edges

# =============================
# Token Alignment
# =============================
def align_tokens(spacy_tokens, wp_tokens):
    node_indices = []
    wp_idx = 1  # after [CLS]
    for _ in spacy_tokens:
        if wp_idx >= len(wp_tokens) - 1:
            break
        node_indices.append(wp_idx)
        wp_idx += 1
        while wp_idx < len(wp_tokens) - 1 and wp_tokens[wp_idx].startswith("##"):
            wp_idx += 1
    return node_indices

# =============================
# Data Collation
# =============================
def my_collate_fn(batch):
    input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    attention_mask = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
    labels = [ex.get("labels", None) for ex in batch]

    premise_graph_tokens = [ex.get("premise_graph_tokens") for ex in batch]
    premise_graph_edges = [ex.get("premise_graph_edges") for ex in batch]
    premise_node_indices = [ex.get("premise_node_indices") for ex in batch]

    hypothesis_graph_tokens = [ex.get("hypothesis_graph_tokens") for ex in batch]
    hypothesis_graph_edges = [ex.get("hypothesis_graph_edges") for ex in batch]
    hypothesis_node_indices = [ex.get("hypothesis_node_indices") for ex in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long) if labels and labels[0] is not None else None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "premise_graph_tokens": premise_graph_tokens,
        "premise_graph_edges": premise_graph_edges,
        "premise_node_indices": premise_node_indices,
        "hypothesis_graph_tokens": hypothesis_graph_tokens,
        "hypothesis_graph_edges": hypothesis_graph_edges,
        "hypothesis_node_indices": hypothesis_node_indices,
    }

# =============================
# Training Loop
# =============================
def train_model(epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE,
                lr: float = LEARNING_RATE,
                save_model: bool = False,
                save_path: str = 'gnn_model_weights_3.pt'):
    set_seed()
    process_data()
    logging.info("Loading preprocessed dataset...")
    snli = load_from_disk(PREPROCESSED_DIR)
    snli.set_format("python", output_all_columns=True)

    train_loader = DataLoader(snli["train"], batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_loader   = DataLoader(snli["validation"], batch_size=batch_size, collate_fn=my_collate_fn)

    model = GraphAugmentedNLIModel(MODEL_NAME).to(DEVICE)

    if hasattr(model.bert, 'gradient_checkpointing_enable'):
        model.bert.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing on BERT.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)

    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    model.train()
    all_losses = []
    epoch_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            labels = batch["labels"].to(DEVICE) if batch.get("labels") is not None else None
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                premise_graph_tokens=batch["premise_graph_tokens"],
                premise_graph_edges=batch["premise_graph_edges"],
                premise_node_indices=batch["premise_node_indices"],
                hypothesis_graph_tokens=batch["hypothesis_graph_tokens"],
                hypothesis_graph_edges=batch["hypothesis_graph_edges"],
                hypothesis_node_indices=batch["hypothesis_node_indices"],
                labels=labels
            )
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            loss_val = loss.item()
            epoch_loss.append(loss_val)
            all_losses.append(loss_val)
            progress.set_postfix({"loss": f"{loss_val:.4f}"})

        avg_epoch_loss = np.mean(epoch_loss)
        epoch_losses.append(avg_epoch_loss)
        logging.info(f"Epoch {epoch} completed. Avg Loss: {avg_epoch_loss:.4f}")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].to(DEVICE) if batch.get("labels") is not None else None
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    premise_graph_tokens=batch["premise_graph_tokens"],
                    premise_graph_edges=batch["premise_graph_edges"],
                    premise_node_indices=batch["premise_node_indices"],
                    hypothesis_graph_tokens=batch["hypothesis_graph_tokens"],
                    hypothesis_graph_edges=batch["hypothesis_graph_edges"],
                    hypothesis_node_indices=batch["hypothesis_node_indices"],
                    labels=labels
                )
                loss_item = outputs.get("loss").item() if isinstance(outputs, dict) else outputs.item()
                val_losses.append(loss_item)
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        logging.info(f"Validation Loss after Epoch {epoch}: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            if save_model:
                logging.info(f"Saving best model at epoch {epoch} with val loss {avg_val_loss:.4f}")
                torch.save(model.state_dict(), save_path)
        model.train()

    # Plot losses
    plt.figure()
    plt.plot(all_losses)
    plt.xlabel('Training steps')
    plt.ylabel('Loss')
    plt.title('Step-wise Training Loss')
    plt.show()

    plt.figure()
    plt.plot(range(1, epochs+1), epoch_losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch-wise Training Loss')
    plt.show()

    logging.info(f"Training complete. Best validation loss {best_val_loss:.4f} at epoch {best_epoch}.")
    return model


def predict_nli(premise, hypothesis, tokenizer=tokenizer, model_path='gnn_model_checkpoint.pt'):
    # 1) instantiate the model exactly as you did during training
    model = GraphAugmentedNLIModel(MODEL_NAME).to(DEVICE)

    # 2) load the checkpoint, then hand only the model weights to load_state_dict
    ckpt = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    # 3) tokenize & build graphs (as before)…
    encoded = tokenizer(
        premise, hypothesis,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Build dependency graphs
    p_tokens, p_edges = build_dependency_graph(premise)
    h_tokens, h_edges = build_dependency_graph(hypothesis)

    # Convert ids back to tokens for alignment
    wp_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    p_node_indices = align_tokens(p_tokens, wp_tokens)
    h_node_indices = align_tokens(h_tokens, wp_tokens)

    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # Prepare inputs for the model: the model expects lists for graph fields
    # since we used a custom collate_fn logic.
    premise_graph_tokens = [p_tokens]
    premise_graph_edges = [p_edges]
    premise_node_indices = [p_node_indices]

    hypothesis_graph_tokens = [h_tokens]
    hypothesis_graph_edges = [h_edges]
    hypothesis_node_indices = [h_node_indices]

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            premise_graph_tokens=premise_graph_tokens,
            premise_graph_edges=premise_graph_edges,
            premise_node_indices=premise_node_indices,
            hypothesis_graph_tokens=hypothesis_graph_tokens,
            hypothesis_graph_edges=hypothesis_graph_edges,
            hypothesis_node_indices=hypothesis_node_indices
        )

    logits = outputs["logits"]
    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    # Get predicted label
    predicted_label_id = torch.argmax(logits, dim=-1).item()
    predicted_label = label_map[predicted_label_id]
    prob_map = dict()
    for i, cls_label in label_map.items():
        prob_map[cls_label] = probs[i]
    return predicted_label, prob_map


def predict_fin_nli(
        premise: str,
        hypothesis: str,
        tokenizer=tokenizer,
        model_path: str = 'gnn_model_checkpoint.pt',
        adapter_dir: str = './lora_finance_adapter',
) -> (str, list):
    # 1) Load base GraphAugmentedFinNLIModel and its checkpoint
    base_model = GraphAugmentedFinNLIModel(MODEL_NAME).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE)
    base_model.load_state_dict(ckpt['model_state_dict'])

    # 2) Wrap with the same LoRA config you used in training
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        task_type='SEQ_CLS',
        target_modules=['query', 'value']
    )
    model = get_peft_model(base_model, lora_cfg).to(DEVICE)

    # 3) Load your adapter checkpoint (the .pt under lora_finance_adapter/)
    adapter_ckpt = torch.load(os.path.join(adapter_dir, 'training_checkpoint.pt'), map_location=DEVICE)
    # This checkpoint contains the same 'model_state_dict' keys—so load it leniently:
    model.load_state_dict(adapter_ckpt['model_state_dict'], strict=False)
    model.eval()

    # 4) Tokenize
    enc = tokenizer(
        premise, hypothesis,
        truncation=True,
        padding='max_length',
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    # 5) Build & align your dependency graphs
    p_toks, p_edges = build_dependency_graph(premise)
    h_toks, h_edges = build_dependency_graph(hypothesis)
    wp = tokenizer.convert_ids_to_tokens(input_ids[0])
    p_idx = align_tokens(p_toks, wp)
    h_idx = align_tokens(h_toks, wp)

    premise_graph_tokens = [p_toks]
    premise_graph_edges = [p_edges]
    premise_node_indices = [p_idx]
    hypothesis_graph_tokens = [h_toks]
    hypothesis_graph_edges = [h_edges]
    hypothesis_node_indices = [h_idx]

    # 6) Forward
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            premise_graph_tokens=premise_graph_tokens,
            premise_graph_edges=premise_graph_edges,
            premise_node_indices=premise_node_indices,
            hypothesis_graph_tokens=hypothesis_graph_tokens,
            hypothesis_graph_edges=hypothesis_graph_edges,
            hypothesis_node_indices=hypothesis_node_indices
        )

    logits = out['logits'][0]  # shape [3]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # 7) Collapse to entailment vs. contradiction (ignore neutral)
    entail, neutral, contra = probs
    s = entail + contra + 1e-12
    scores = [entail / s, contra / s]
    label = 'entailment' if entail >= contra else 'contradiction'
    return label, scores


def train_model_with_chkpt(epochs: int = 5,
                batch_size: int = 16,
                lr: float = 2e-5,
                save_model: bool = False,
                save_path: str = 'gnn_model_checkpoint.pt',
                resume: bool = False):
    """
    Train with mixed precision, gradient checkpointing, and resume support.
    If resume=True and save_path exists, picks up from last epoch.
    """
    set_seed()
    process_data()
    logging.info("Loading preprocessed dataset…")
    snli = load_from_disk(PREPROCESSED_DIR)
    snli.set_format("python", output_all_columns=True)

    train_loader = DataLoader(snli["train"], batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)
    val_loader   = DataLoader(snli["validation"], batch_size=batch_size, collate_fn=my_collate_fn)

    model = GraphAugmentedNLIModel(MODEL_NAME).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    # --- Resume checkpoint if requested ---
    start_epoch = 1
    if resume and os.path.isfile(save_path):
        ckpt = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 1) + 1
        logging.info(f"Resuming from epoch {start_epoch}")

    # Mixed precision setup
    if hasattr(model.bert, "gradient_checkpointing_enable"):
        model.bert.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing on BERT.")
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val_loss = float("inf")
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                premise_graph_tokens=batch["premise_graph_tokens"],
                premise_graph_edges=batch["premise_graph_edges"],
                premise_node_indices=batch["premise_node_indices"],
                hypothesis_graph_tokens=batch["hypothesis_graph_tokens"],
                hypothesis_graph_edges=batch["hypothesis_graph_edges"],
                hypothesis_node_indices=batch["hypothesis_node_indices"],
                labels=batch.get("labels", None).to(DEVICE) if batch.get("labels") is not None else None
            )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())
        avg_train = np.mean(train_losses)
        logging.info(f"Epoch {epoch} train loss: {avg_train:.4f}")

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(DEVICE),
                    attention_mask=batch["attention_mask"].to(DEVICE),
                    premise_graph_tokens=batch["premise_graph_tokens"],
                    premise_graph_edges=batch["premise_graph_edges"],
                    premise_node_indices=batch["premise_node_indices"],
                    hypothesis_graph_tokens=batch["hypothesis_graph_tokens"],
                    hypothesis_graph_edges=batch["hypothesis_graph_edges"],
                    hypothesis_node_indices=batch["hypothesis_node_indices"],
                    labels=batch.get("labels", None).to(DEVICE) if batch.get("labels") is not None else None
                )
                v_loss = outputs["loss"].item() if isinstance(outputs, dict) else outputs.item()
                val_losses.append(v_loss)
        avg_val = np.mean(val_losses) if val_losses else float("inf")
        logging.info(f"Epoch {epoch} val loss: {avg_val:.4f}")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(ckpt, save_path)
        logging.info(f"Saved checkpoint: {save_path}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val

    logging.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return model


def extract_sentences_by_intent(
    text: str,
    intent: str,
    adapter_dir: str = "./lora_finance_adapter",
    threshold: float = 0.7,
    top_k: int = None,
    min_words: int = 4,
    convo_focus: str = None
):
    """
    Splits `text` into sentences, embeds them (and the `intent`) under your
    LoRA‐adapted BERT, and returns those whose cosine similarity ≥ `threshold`.
    Loads the adapter from the single `training_checkpoint.pt` in `adapter_dir`.
    """
    # 1) Sentence split & cleanup
    # 1) Only consider lines spoken by the customer

    if convo_focus is None:
        sentences = [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]

    elif convo_focus == "customer":
        customer_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip().lower().startswith("customer:")
        ]

        # 2) Sentence-split each customer line
        sentences = []
        for cust_line in customer_lines:
            for sent in nlp(cust_line).sents:
                s = sent.text.strip()
                if s and len(s.split(' '))>6:
                    sentences.append(s)

    else:
        customer_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip().lower().startswith("agent:")
        ]

        # 2) Sentence-split each customer line
        sentences = []
        for cust_line in customer_lines:
            for sent in nlp(cust_line).sents:
                s = sent.text.strip()
                if s and len(s.split(' '))>6:
                    sentences.append(s)

    # 2) Load base BERT + wrap in same LoRA config
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",        # must match your fine-tune setting
    )
    model = get_peft_model(base_model, lora_cfg).to(DEVICE)

    # 3) Load your adapter checkpoint
    chkpt_path = os.path.join(adapter_dir, "training_checkpoint.pt")
    if not os.path.isfile(chkpt_path):
        raise FileNotFoundError(f"No LoRA checkpoint at {chkpt_path}")
    ckpt = torch.load(chkpt_path, map_location=DEVICE)
    # ckpt["model_state_dict"] contains both base + LoRA weights; strict=False
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # helper: get [CLS] embedding under LoRA-BERT
    def embed(text_str):
        toks = tokenizer(
            text_str,
            truncation=True,
            padding="longest",
            return_tensors="pt"
        ).to(DEVICE)

        em_args = {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }
        if "token_type_ids" in toks:
            em_args["token_type_ids"] = toks["token_type_ids"]

        # unwrap PEFT to call only the base BertModel
        hf_model = getattr(model, "base_model", model)
        with torch.no_grad():
            last_hidden = hf_model(
                input_ids=em_args["input_ids"],
                attention_mask=em_args["attention_mask"],
                **({"token_type_ids": em_args["token_type_ids"]} if "token_type_ids" in em_args else {})
            ).last_hidden_state
        return last_hidden[:, 0, :]

    # now embed(intent) and each sentence using this safe helper
    intent_emb = embed(intent)

    results = []
    with torch.no_grad():
        for sent in sentences:
            clean = re.sub(r'^(Agent|Customer):\s*', "", sent)
            if len(clean.split()) < min_words:
                continue

            sent_emb = embed(clean)
            sim = F.cosine_similarity(sent_emb, intent_emb, dim=1).item()
            if sim >= threshold:
                results.append((clean, sim))

    # 5) sort & trim
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k] if top_k else results


def train_sentence_extractor(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    output_dir: str,
    val_split: float = 0.2,
    epochs: int      = 3,
    batch_size: int  = 16,
    lr: float        = 2e-5,
    device: str      = "cpu",
    unfreeze_after_epoch: int = 1,
    threshold: float = 0.5
):
    """
    Fine-tune `model` on `dataset`, hold out `val_split` for val,
    compute loss + acc + precision + F1 each epoch, save best checkpoint,
    and plot all four metrics at the end.
    """
    # Split
    total = len(dataset)
    val_n = int(total * val_split)
    train_n = total - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    # Oversample train
    train_labels = [train_ds[i]['label'].item() for i in range(len(train_ds))]
    counts = torch.bincount(torch.tensor(train_labels, dtype=torch.long))
    weights = (1.0 / counts.float()).tolist()
    sample_weights = [weights[int(l)] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model.to(device)
    # initially freeze backbone
    for p in model.bert.parameters(): p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.BCEWithLogitsLoss()

    # storage for metrics
    train_losses, val_losses = [], []
    train_accs,  val_accs  = [], []
    train_precs, val_precs = [], []
    train_f1s,   val_f1s   = [], []

    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        # —— TRAIN ——
        model.train()
        epoch_loss = 0.0
        preds, labels = [], []
        for batch in tqdm(train_loader, desc=f"Train {epoch}/{epochs}"):
            inputs = batch['input_ids'].to(device)
            masks  = batch['attention_mask'].to(device)
            labs   = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(inputs, masks)              # raw logits
            loss   = criterion(logits, labs)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            probs = torch.sigmoid(logits)
            batch_preds = (probs >= threshold).long()
            preds.extend(batch_preds.cpu().tolist())
            labels.extend(labs.cpu().long().tolist())

        avg_train = epoch_loss / len(train_loader)
        train_losses.append(avg_train)
        train_accs.append(  accuracy_score(labels, preds) )
        train_precs.append( precision_score(labels, preds, zero_division=0) )
        train_f1s.append(   f1_score(labels, preds, zero_division=0) )
        print(f"→ Epoch {epoch} Train — loss {avg_train:.4f}, acc {train_accs[-1]:.4f}, prec {train_precs[-1]:.4f}, f1 {train_f1s[-1]:.4f}")

        # unfreeze if needed
        if epoch == unfreeze_after_epoch:
            for p in model.bert.parameters(): p.requires_grad = True
            optimizer = AdamW([
                {"params": model.classifier.parameters(), "lr": 1e-3},
                {"params": model.bert.parameters(),       "lr": 1e-5},
            ], weight_decay=1e-2)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )

        # —— VALIDATION ——
        model.eval()
        epoch_loss = 0.0
        preds, labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f" Val   {epoch}/{epochs}"):
                inputs = batch['input_ids'].to(device)
                masks  = batch['attention_mask'].to(device)
                labs   = batch['label'].to(device)

                logits = model(inputs, masks)
                loss   = criterion(logits, labs)
                epoch_loss += loss.item()

                probs = torch.sigmoid(logits)
                batch_preds = (probs >= threshold).long()
                preds.extend(batch_preds.cpu().tolist())
                labels.extend(labs.cpu().long().tolist())

        avg_val = epoch_loss / len(val_loader)
        val_losses.append(avg_val)
        val_accs.append(  accuracy_score(labels, preds) )
        val_precs.append( precision_score(labels, preds, zero_division=0) )
        val_f1s.append(   f1_score(labels, preds, zero_division=0) )
        print(f"→ Epoch {epoch}   Val — loss {avg_val:.4f}, acc {val_accs[-1]:.4f}, prec {val_precs[-1]:.4f}, f1 {val_f1s[-1]:.4f}")

        # checkpoints
        os.makedirs(output_dir, exist_ok=True)
        ckpt = os.path.join(output_dir, f"epo{epoch}_val{avg_val:.4f}.pth")
        torch.save(model.state_dict(), ckpt)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print(f"🎉 New best model saved (val loss {best_val_loss:.4f})")

    print(f"✔️ Training complete — best val loss: {best_val_loss:.4f}")

    # —— PLOT METRICS ——
    epochs = list(range(1, epochs+1))

    save_metric_plot(
        epochs,
        train_losses,
        val_losses,
        metric_name="Loss",
        output_path="results/Loss_Plot.png"
    )

    save_metric_plot(
        epochs,
        train_accs,
        val_accs,
        metric_name="Accuracy",
        output_path="results/Accuracy_Plot.png",
        threshold=0.5
    )

    save_metric_plot(
        epochs,
        train_precs,
        val_precs,
        metric_name="Precision",
        output_path="results/Precision_Plot.png",
        threshold=0.5
    )

    save_metric_plot(
        epochs,
        train_f1s,
        val_f1s,
        metric_name="F1 Score",
        output_path="results/F1Score_Plot.png",
        threshold=0.5
    )


def save_metric_plot(
    epochs,
    train_vals,
    val_vals,
    metric_name: str,
    output_path: str,
    threshold: float = None
):
    """
    epochs      – list of epoch indices
    train_vals  – list of train metric values
    val_vals    – list of validation metric values
    metric_name – e.g. "Loss", "Accuracy", "Precision", "F1 Score"
    output_path – where to save the PNG
    threshold   – optional horizontal line to draw, e.g. 0.5
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_vals, marker='o', linewidth=2, label=f'Train {metric_name}')
    ax.plot(epochs, val_vals,   marker='s', linewidth=2, label=f'Val {metric_name}')

    if threshold is not None:
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=1, label=f'Threshold = {threshold}')

    ax.set_title(f'{metric_name} over Epochs', fontsize=14, pad=10)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def demo_on_random_val(
    model,
    tokenizer,
    excel_path: str,
    ckpt_path: str,
    max_length: int = 128,
    device: str    = "cpu",
    temperature: float = 1.0
):
    """
    Like demo_on_random_val, but instead of a fixed threshold:
      1) Compute sigmoid(logits / temperature) for each sentence
      2) Sort probabilities descending
      3) Find the largest gap between adjacent probs
      4) Set dynamic_threshold = midpoint of that gap
      5) Extract all sentences with prob >= dynamic_threshold
    """
    # load model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # sample one from validation split
    df = pd.read_excel(excel_path)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    row = val_df.sample(n=1, random_state=random.randint(0,999)).iloc[0]
    transcript = str(row['Claude_Call'])
    print(f"\n── Transcript (val sample idx={row['idx']}):\n{transcript}\n")

    # split into sentences & run inference
    sentences, probs = [], []
    for sent in sent_tokenize(transcript):
        enc = tokenizer.encode_plus(
            sent,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        logits = model(enc['input_ids'].to(device),
                       enc['attention_mask'].to(device))
        prob   = torch.sigmoid(logits / temperature).item()
        sentences.append(sent)
        probs.append(prob)

    # print all
    print("Sentence probabilities:")
    for s,p in zip(sentences, probs):
        print(f"  → {p:.4f} → {s}")

    # if no variation, fall back to 0.5
    if len(probs) < 2 or max(probs) - min(probs) < 1e-3:
        dynamic_thr = 0.5
    else:
        # find elbow in sorted probabilities
        sorted_probs = sorted(probs, reverse=True)
        diffs = [sorted_probs[i] - sorted_probs[i+1] for i in range(len(sorted_probs)-1)]
        idx = max(range(len(diffs)), key=lambda i: diffs[i])
        # threshold is midpoint between the two
        dynamic_thr = (sorted_probs[idx] + sorted_probs[idx+1]) / 2.0

    print(f"\nDynamic threshold = {dynamic_thr:.4f}\n")
    print("Extracted sentences:")
    for s,p in zip(sentences, probs):
        if p >= dynamic_thr:
            print(f"  • {p:.4f} → {s}")
    print()


def batch_predict_and_save(
    model,
    tokenizer,
    excel_path: str,
    ckpt_path: str,
    output_path: str,
    n_samples: int     = 40,
    max_length: int    = 128,
    device: str        = "cpu",
    temperature: float = 1.0,
    random_state: int  = None
):
    """
    1) Loads best checkpoint
    2) Samples `n_samples` rows
    3) For each transcript:
         - tokenize into sentences
         - compute p = sigmoid(logits/temperature)
         - compute elbow threshold on sorted p’s
         - extract all sentences with p >= elbow
         - if none, pick the highest-p sentence
    4) Save new Excel with columns:
         - 'Claude_Call'
         - 'Predicted Sel_K' (list of extracted sentences)
    """
    # load model
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    # sample rows
    df = pd.read_excel(excel_path)
    sampled = df.sample(n=n_samples, random_state=random_state) \
               if random_state is not None else df.sample(n=n_samples)

    records = []
    for _, row in tqdm(sampled.iterrows(),
                       total=len(sampled),
                       desc="Running Predictions"):
        transcript = str(row['Claude_Call'])
        sentences  = sent_tokenize(transcript)

        # compute probabilities
        probs = []
        for sent in sentences:
            enc = tokenizer.encode_plus(
                sent,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                logits = model(enc['input_ids'].to(device),
                               enc['attention_mask'].to(device))
                p = torch.sigmoid(logits / temperature).item()
            probs.append(p)

        # dynamic threshold via elbow detection
        if len(probs) >= 2 and max(probs) - min(probs) > 1e-3:
            sp = sorted(probs, reverse=True)
            diffs = [sp[i] - sp[i+1] for i in range(len(sp)-1)]
            idx  = max(range(len(diffs)), key=lambda i: diffs[i])
            thr  = (sp[idx] + sp[idx+1]) / 2.0
        else:
            thr = 0.5  # fallback

        # collect all above threshold, else top-1
        extracted = [s for s,p in zip(sentences, probs) if p >= thr]
        if not extracted and sentences:
            best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
            extracted = [sentences[best_idx]]

        records.append({
            'Claude_Call':     transcript,
            'Predicted Sel_K': extracted
        })

    # save
    out_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    out_df.to_excel(output_path, index=False)
    print(f"➡️ Saved {len(out_df)} rows to {output_path}")