import os
import torch
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

# Configuration constants
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 5e-4
EPOCHS = 20
SEED = 42
ADAPTER_SAVE_DIR = "./lora_finance_adapter"
CHECKPOINT_PATH = os.path.join(ADAPTER_SAVE_DIR, "training_checkpoint.pt")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fine_tune_lora(dataset_name: str = "FinGPT/fingpt-fiqa_qa", split: str = "train"):
    """
    Fine-tune BERT with LoRA on an MLM objective.
    Supports checkpointing and resuming, and plots loss, perplexity, and MLM accuracy per epoch.
    Saves the LoRA adapter and checkpoint in ADAPTER_SAVE_DIR.
    """
    set_seed()

    # Prepare save directory
    os.makedirs(ADAPTER_SAVE_DIR, exist_ok=True)

    # Load and prepare dataset
    dataset = load_dataset(dataset_name, split=split)
    def combine_fields(example):
        text = ' '.join([example.get(k, '').strip() for k in ['instruction', 'input', 'output'] if example.get(k)])
        return {"text": text}
    dataset = dataset.map(combine_fields, remove_columns=[c for c in dataset.column_names if c != 'text'])

    # Tokenization and DataLoader
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    def tokenize_fn(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=[c for c in dataset.column_names if c != 'text'])
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True,
    )

    # Model, LoRA, optimizer, scheduler
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    lora_cfg = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, lora_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Accelerator
    accelerator = Accelerator()
    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, scheduler)
    device = accelerator.device

    # Metrics storage and resume state
    start_epoch = 1
    epoch_losses = []
    epoch_ppls = []
    epoch_accs = []

    # Load checkpoint if exists
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        epoch_losses = ckpt.get('epoch_losses', [])
        epoch_ppls = ckpt.get('epoch_ppls', [])
        epoch_accs = ckpt.get('epoch_accs', [])
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    model.train()
    for epoch in range(start_epoch, EPOCHS + 1):
        total_loss, total_masked, correct_masked = 0.0, 0, 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            # Accumulate
            step_loss = loss.item()
            total_loss += step_loss
            preds = torch.argmax(logits, dim=-1)
            mask = labels.ne(-100)
            correct_masked += preds.eq(labels).masked_select(mask).sum().item()
            total_masked += mask.sum().item()
            progress.set_postfix({'loss': f"{step_loss:.4f}"})

        # Epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_ppl = math.exp(avg_loss)
        avg_acc = correct_masked / total_masked if total_masked > 0 else 0
        epoch_losses.append(avg_loss)
        epoch_ppls.append(avg_ppl)
        epoch_accs.append(avg_acc)
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, PPL={avg_ppl:.2f}, MLM Acc={avg_acc:.4%}")

        # Save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch_losses': epoch_losses,
            'epoch_ppls': epoch_ppls,
            'epoch_accs': epoch_accs,
        }
        torch.save(ckpt, CHECKPOINT_PATH)

    # Final plots
    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
    epochs_list = list(range(1, len(epoch_losses) + 1))
    axes[0].plot(epochs_list, epoch_losses, marker='o'); axes[0].set_ylabel('Loss'); axes[0].set_title('Training Loss'); axes[0].grid(True)
    axes[1].plot(epochs_list, epoch_ppls, marker='o'); axes[1].set_ylabel('Perplexity'); axes[1].set_title('Training Perplexity'); axes[1].grid(True)
    axes[2].plot(epochs_list, epoch_accs, marker='o'); axes[2].set_ylabel('MLM Accuracy'); axes[2].set_xlabel('Epoch'); axes[2].set_title('Masked LM Accuracy'); axes[2].grid(True)
    plt.tight_layout(); plt.show()

    # Save LoRA adapter
    model.save_pretrained(ADAPTER_SAVE_DIR)
    print(f"LoRA adapter saved to {ADAPTER_SAVE_DIR}")


if __name__ == '__main__':
    fine_tune_lora()