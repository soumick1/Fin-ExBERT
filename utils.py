import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_linear_schedule_with_warmup
from peft import PeftModel, LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict, load_from_disk
import spacy
import re
import networkx as nx
from tqdm.auto import tqdm
from accelerate import Accelerator
import matplotlib.pyplot as plt

from models import GraphAugmentedNLIModel, GraphAugmentedFinNLIModel, FrozenGNNBertSpanModel
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


def train_span_extraction(
    train_data,
    val_data,
    tokenizer=tokenizer,
    epochs=15,
    batch_size=5,
    lr=5e-4,
    base_model_name="bert-base-uncased",
    device="cuda",
    save_model=False,
    save_path='span_extraction_model.pt',
    freeze=True,
    lora_enabled=True,
    show_graph=False,
    few_shot=False,
    k_shot=2
):
    # Build datasets
    if few_shot:
        few = create_few_shot_data(train_data, k=k_shot)
        train_dataset = SpanExtractionChunkedDataset(few)
        val_dataset   = SpanExtractionChunkedDataset(val_data)
    else:
        train_dataset = SpanExtractionChunkedDataset(train_data)
        val_dataset   = SpanExtractionChunkedDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=span_collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=span_collate_fn)

    do_validation = len(val_dataset) > 0

    # Model & optimizer
    model = FrozenGNNBertSpanModel(
        base_model_name=base_model_name,
        freeze=freeze,
        lora_enabled=lora_enabled
    ).to(device)
    optimizer = torch.optim.AdamW(model.span_head.parameters(), lr=lr)
    ce_loss   = nn.CrossEntropyLoss()
    bce_loss  = nn.BCEWithLogitsLoss()

    epoch_loss_train = []
    epoch_loss_val   = []
    best_val_loss    = float('inf')
    best_epoch       = -1

    for epoch in range(1, epochs+1):
        # — train —
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} — Training"):
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos      = batch["start_positions"].to(device)
            end_pos        = batch["end_positions"].to(device)
            no_span_label  = batch["no_span_label"].float().to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits, end_logits, no_span_logit = out["start_logits"], out["end_logits"], out["no_span_logit"]

            # only compute span losses where no_span==0
            valid = (start_pos >= 0) & (end_pos >= 0) & (no_span_label == 0)
            if valid.any():
                loss_start = ce_loss(start_logits[valid], start_pos[valid])
                loss_end   = ce_loss(end_logits[valid],   end_pos[valid])
            else:
                loss_start = torch.tensor(0.0, device=device)
                loss_end   = torch.tensor(0.0, device=device)

            loss_nospan = bce_loss(no_span_logit, no_span_label)
            loss = loss_start + loss_end + loss_nospan

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = float(np.mean(train_losses))
        epoch_loss_train.append(avg_train)
        print(f"Epoch {epoch} — Train Loss: {avg_train:.4f}")

        # — validate —
        if do_validation:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    start_pos      = batch["start_positions"].to(device)
                    end_pos        = batch["end_positions"].to(device)
                    no_span_label  = batch["no_span_label"].float().to(device)

                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                    start_logits, end_logits, no_span_logit = out["start_logits"], out["end_logits"], out["no_span_logit"]

                    valid = (start_pos >= 0) & (end_pos >= 0) & (no_span_label == 0)
                    if valid.any():
                        loss_start = ce_loss(start_logits[valid], start_pos[valid])
                        loss_end   = ce_loss(end_logits[valid],   end_pos[valid])
                    else:
                        loss_start = torch.tensor(0.0, device=device)
                        loss_end   = torch.tensor(0.0, device=device)
                    loss_nospan = bce_loss(no_span_logit, no_span_label)

                    val_losses.append((loss_start + loss_end + loss_nospan).item())

            avg_val = float(np.mean(val_losses)) if val_losses else float('inf')
            epoch_loss_val.append(avg_val)
            print(f"Epoch {epoch} — Val   Loss: {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch    = epoch
                if save_model:
                    torch.save(model.state_dict(), save_path)
        else:
            print(f"Epoch {epoch} — no validation data, skipping.")

    print(f"\nTraining complete. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")

    if show_graph and do_validation:
        plt.plot(epoch_loss_train, label="Train Loss", marker='o')
        plt.plot(epoch_loss_val,   label="Val   Loss", marker='o')
        plt.title("Span‐Extraction Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig("span_extraction_loss.png")

    return model


def create_few_shot_data(full_data, k=5, random_seed=42):
    random.seed(random_seed)
    if len(full_data) <= k:
        return full_data
    else:
        return random.sample(full_data, k)


def predict_span_in_transcript(transcript_text, tokenizer=tokenizer, device="cuda", threshold_no_span=0.0, model_path='span_extraction_model.pt', freeze=True, lora_enabled=False, show_logits=False, score_thresh=0.8):
    """
    1) Chunks transcript
    2) For each chunk, run model, pick best start/end if no_span_logit <= threshold_no_span
    3) Return the best overall span text or None
    """
    all_spans = []
    model = FrozenGNNBertSpanModel(base_model_name=MODEL_NAME, freeze=freeze, lora_enabled=lora_enabled).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    chunked = chunk_transcript(transcript_text, -1, -1, tokenizer)  # -1 => no known gold
    best_score = float("-inf")
    best_span_text = None

    logits = []

    with torch.no_grad():
        for chunk_dict in chunked:
            ids = chunk_dict["input_ids"].unsqueeze(0).to(device)
            mask = chunk_dict["attention_mask"].unsqueeze(0).to(device)
            outputs = model(input_ids=ids, attention_mask=mask)
            start_logits = outputs["start_logits"][0].cpu().numpy()  # [seq_len]
            start_logits = np.exp(start_logits)/np.sum(np.exp(start_logits))
            end_logits   = outputs["end_logits"][0].cpu().numpy()
            end_logits = np.exp(end_logits)/np.sum(np.exp(end_logits))
            no_span_val  = outputs["no_span_logit"][0].item()

            logits.append(no_span_val)

            if no_span_val > threshold_no_span:
                # Model indicates no span for this chunk
                continue

            # find best start
            best_start_idx = int(np.argmax(start_logits))
            best_end_idx   = int(np.argmax(end_logits))
            if best_end_idx < best_start_idx:
                continue

            score = (start_logits[best_start_idx] + end_logits[best_end_idx])/2
            if score > best_score:
                best_score = score
                # decode
                local_ids = chunk_dict["input_ids"][best_start_idx : best_end_idx+1]
                buffer = tokenizer.decode(local_ids, skip_special_tokens=True)
                #print(buffer)
                if len(buffer.split()) > 8:
                    best_span_text = buffer

            if score>=score_thresh:
                local_ids = chunk_dict["input_ids"][best_start_idx : best_end_idx+1]
                span_text = tokenizer.decode(local_ids, skip_special_tokens=True)
                all_spans.append((score, span_text))

    if show_logits:
        #print('Logits from the model for each chunk: {}'.format(logits))
        #print(sorted(all_spans, key=lambda x: x[0], reverse=True))
        return best_span_text, sorted(all_spans, key=lambda x: x[0], reverse=True)
    return best_span_text


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
