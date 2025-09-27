import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import pipeline
from utils import extract_sentences_by_intent, nlp

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TOP_K       = 3          # number of spans to extract per example
N_PER_DS    = 200        # how many *valid* examples per dataset
BATCH_SIZE  = 16         # batch size for judge pipelines
DEVICE      = 0          # GPU id, or -1 for CPU
OUTPUT_PATH = "results/combined_results.xlsx"
# ────────────────────────────────────────────────────────────────────────────────

def flatten_to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "text" in x and isinstance(x["text"], str):
            return x["text"]
        return "\n".join(flatten_to_text(v) for v in x.values())
    if isinstance(x, list):
        return "\n".join(flatten_to_text(v) for v in x)
    return str(x)

def label_to_int(lbl: str) -> int:
    # handles both variants A and B
    m = re.search(r"([1-5])", lbl)
    if m:                       # digits present -> easy
        return int(m.group(1))
    # descriptive version -> map by order
    mapping = {
        "very bad answer": 1,
        "bad answer": 2,
        "acceptable answer": 3,
        "good answer": 4,
        "perfect answer": 5
    }
    return mapping.get(lbl.lower(), 1)

# ─── 1) choose these two datasets ──────────────────────────────────────────────
datasets_info = [
    ("FinQA-10K", "virattt/financial-qa-10K", "train", False, {}),
    ("SQuAD",     "rajpurkar/squad",          "validation", False, {}),
]

# ─── 2) spin up zero‐shot classification judges ────────────────────────────────
candidate_labels = [
    "very bad answer",   # 1
    "bad answer",        # 2
    "acceptable answer", # 3
    "good answer",       # 4
    "perfect answer"     # 5
]

judge1 = pipeline(
    "zero-shot-classification",
    model="roberta-large-mnli",
    device=DEVICE,
    batch_size=BATCH_SIZE
)
judge2 = pipeline(
    "zero-shot-classification",
    model="microsoft/deberta-base-mnli",
    tokenizer="microsoft/deberta-base-mnli",
    device=DEVICE,
    batch_size=BATCH_SIZE
)
judge3 = pipeline(
    "zero-shot-classification",
    model="valhalla/distilbart-mnli-12-3",
    tokenizer="valhalla/distilbart-mnli-12-3",
    device=DEVICE,
    batch_size=BATCH_SIZE
)

all_rows = []

for ds_name, hf_id, split, trust_code, extra_kwargs in datasets_info:
    print(f"\n→ Loading {ds_name} ({hf_id}#{split}), gathering {N_PER_DS} valid examples…")
    # load full split and shuffle
    ds = load_dataset(hf_id, split=split, trust_remote_code=trust_code, **extra_kwargs)
    ds = ds.shuffle(seed=42)

    collected = 0
    pbar = tqdm(total=N_PER_DS, desc=f"{ds_name} valid examples")
    for ex in ds:
        # unify question & context
        question = (
            ex.get("question")
            or ex.get("question_text")
            or ex.get("query")
            or ""
        )
        raw_ctx = (
            ex.get("context")
            or ex.get("document_text")
            or ex.get("story")
            or ex.get("text")
            or ""
        )
        context = flatten_to_text(raw_ctx)

        # only keep examples whose context has at least 2 sentences
        if len(list(nlp(context).sents)) < 2:
            continue

        # extract top-K spans
        hits = extract_sentences_by_intent(
            text        = context,
            intent      = question,
            threshold   = -1.0,
            top_k       = TOP_K,
            convo_focus = None
        )
        spans = [s for s,_ in hits]
        if not spans:
            # record defaults if no span found
            all_rows.append({
                "dataset":   ds_name,
                "question":  question,
                "context":   context,
                "span":      "",
                "score1":    5.0,
                "score2":    5.0,
                "score3":    5.0,
                "score_avg": 5.0
            })
            collected += 1
            pbar.update(1)
            if collected >= N_PER_DS:
                break
            continue

        # build prompts
        prompts = [
            f"Question: {question}\nCandidate answer: {span}\n\n"
            "On a scale from 1 (completely wrong) to 5 (perfect), "
            "reply with a single digit."
            for span in spans
        ]

        # run judges
        out1 = judge1(prompts, candidate_labels=candidate_labels, multi_label=False)
        out2 = judge2(prompts, candidate_labels=candidate_labels, multi_label=False)
        out3 = judge3(prompts, candidate_labels=candidate_labels, multi_label=False)

        # parse their top‐chosen labels
        j1 = [int(o["labels"][0]) for o in out1]
        j2 = [int(o["labels"][0]) for o in out2]
        j3 = [int(o["labels"][0]) for o in out3]

        # average per span, pick best
        avg_scores = [(a+b+c)/3.0 for a,b,c in zip(j1,j2,j3)]
        best_idx   = int(np.argmax(avg_scores))

        all_rows.append({
            "dataset":   ds_name,
            "question":  question,
            "context":   context,
            "span":      spans[best_idx],
            "score1":    float(j1[best_idx]),
            "score2":    float(j2[best_idx]),
            "score3":    float(j3[best_idx]),
            "score_avg": float(avg_scores[best_idx])
        })
        collected += 1
        pbar.update(1)
        if collected >= N_PER_DS:
            break

    pbar.close()
    if collected < N_PER_DS:
        print(f"⚠️ Only found {collected} valid examples for {ds_name}.")

# ─── dump to Excel ─────────────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_excel(OUTPUT_PATH, index=False)
print(f"\n✔️  Saved combined results to ./{OUTPUT_PATH}")

# ─── per‐dataset summary ───────────────────────────────────────────────────────
print("\n▶︎ Per‐dataset judge averages:")
grouped = df.groupby("dataset")[["score1","score2","score3","score_avg"]].mean()
for ds, row in grouped.iterrows():
    print(f"  {ds}: "
          f"Judge1={row['score1']:.2f}, "
          f"Judge2={row['score2']:.2f}, "
          f"Judge3={row['score3']:.2f}, "
          f"Combined={row['score_avg']:.2f}")
