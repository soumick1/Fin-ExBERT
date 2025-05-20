import os
import logging
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

from config import MODEL_NAME, MAX_LENGTH, OVERLAP, PREPROCESSED_DIR, tokenizer, nlp

# =============================
# Logging Setup
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# =============================
# One-Time Preprocessing
# =============================
def process_data():
    if not os.path.exists(PREPROCESSED_DIR):
        logging.info("Preprocessing data... This may take a while.")
        # Load and filter SNLI
        snli = load_dataset("snli")
        snli = snli.filter(lambda x: x["label"] != -1)

        def build_dependency_graph(sentence):
            doc = nlp(sentence)
            tokens = [tok.text for tok in doc]
            edges = []
            for tok in doc:
                if tok.head.i != tok.i:
                    edges.extend([(tok.i, tok.head.i), (tok.head.i, tok.i)])
            return tokens, edges

        def preprocess(examples):
            premises = examples["premise"]
            hypotheses = examples["hypothesis"]
            labels = examples["label"]
            tokenized = tokenizer(premises, hypotheses,
                                  truncation=True, padding="max_length",
                                  max_length=MAX_LENGTH)
            tokenized["labels"] = labels

            p_tokens_list, p_edges_list, p_idx_list = [], [], []
            h_tokens_list, h_edges_list, h_idx_list = [], [], []

            for p, h, input_ids in zip(premises, hypotheses, tokenized["input_ids"]):
                p_toks, p_edges = build_dependency_graph(p)
                h_toks, h_edges = build_dependency_graph(h)
                wp_tokens = tokenizer.convert_ids_to_tokens(input_ids)

                def align_tokens(spacy_tokens, wp_tokens):
                    node_indices, wp_idx = [], 1
                    for _ in spacy_tokens:
                        if wp_idx >= len(wp_tokens) - 1: break
                        node_indices.append(wp_idx)
                        wp_idx += 1
                        while wp_idx < len(wp_tokens) - 1 and wp_tokens[wp_idx].startswith("##"):
                            wp_idx += 1
                    return node_indices

                p_idx = align_tokens(p_toks, wp_tokens)
                h_idx = align_tokens(h_toks, wp_tokens)

                p_tokens_list.append(p_toks)
                p_edges_list.append(p_edges)
                p_idx_list.append(p_idx)

                h_tokens_list.append(h_toks)
                h_edges_list.append(h_edges)
                h_idx_list.append(h_idx)

            tokenized.update({
                "premise_graph_tokens": p_tokens_list,
                "premise_graph_edges": p_edges_list,
                "premise_node_indices": p_idx_list,
                "hypothesis_graph_tokens": h_tokens_list,
                "hypothesis_graph_edges": h_edges_list,
                "hypothesis_node_indices": h_idx_list,
            })
            return tokenized

        snli = snli.map(preprocess, batched=True)
        snli.save_to_disk(PREPROCESSED_DIR)
        logging.info(f"Preprocessing complete. Saved to {PREPROCESSED_DIR}")
    else:
        logging.info("Using existing preprocessed data at %s", PREPROCESSED_DIR)


def chunk_transcript(transcript_text, start_idx, end_idx, tokenizer):
    encoded = tokenizer(transcript_text,
                        return_offsets_mapping=True,
                        add_special_tokens=True,
                        return_tensors=None,
                        max_length=1024,
                        padding=False,
                        truncation=False)
    all_input_ids = encoded["input_ids"]
    all_offsets   = encoded["offset_mapping"]

    chunks = []
    i = 0
    while i < len(all_input_ids):
        chunk_ids = all_input_ids[i : i + MAX_LENGTH]
        chunk_offsets = all_offsets[i : i + MAX_LENGTH]
        attention_mask = [1] * len(chunk_ids)

        no_span = 1
        start_token, end_token = -1, -1
        if start_idx >= 0 and end_idx >= 0:
            for j, (off_s, off_e) in enumerate(chunk_offsets):
                if off_s <= start_idx < off_e:
                    start_token = j
                if off_s < end_idx <= off_e:
                    end_token = j
                    break
            if 0 <= start_token <= end_token:
                no_span = 0
            else:
                start_token, end_token = -1, -1

        chunks.append({
            "input_ids": torch.tensor(chunk_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "start_label": start_token,
            "end_label": end_token,
            "no_span_label": no_span,
        })
        i += (MAX_LENGTH - OVERLAP)
    return chunks


class SpanExtractionChunkedDataset(Dataset):
    def __init__(self, data):
        self.samples = []
        for item in data:
            chunks = chunk_transcript(
                item.get("transcript", ""),
                item.get("start_idx", -1),
                item.get("end_idx", -1),
                tokenizer)
            self.samples.extend(chunks)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def span_collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    inputs, masks, starts, ends, nos = [], [], [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        inputs.append(torch.cat([x["input_ids"], torch.zeros(pad, dtype=torch.long)]).unsqueeze(0))
        masks.append(torch.cat([x["attention_mask"], torch.zeros(pad, dtype=torch.long)]).unsqueeze(0))
        starts.append(x["start_label"])
        ends.append(x["end_label"])
        nos.append(x["no_span_label"])
    return {
        "input_ids": torch.cat(inputs, dim=0),
        "attention_mask": torch.cat(masks, dim=0),
        "start_positions": torch.tensor(starts, dtype=torch.long),
        "end_positions": torch.tensor(ends, dtype=torch.long),
        "no_span_label": torch.tensor(nos, dtype=torch.long),
    }
