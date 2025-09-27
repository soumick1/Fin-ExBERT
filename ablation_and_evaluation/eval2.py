import os, re, numpy as np, pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import pipeline
from utils import extract_sentences_by_intent, nlp   # <- your spaCy model

# ─── CONFIG ──────────────────────────────────────────────────────────
TOP_K       = 3          # candidate spans per example
N_PER_DS    = 200        # keep *valid* examples per dataset
BATCH_SIZE  = 16
DEVICE      = 0          # GPU id (-1 = CPU)
OUTPUT_PATH = "results/combined_results.xlsx"
# ────────────────────────────────────────────────────────────────────

# ── helper: flatten arbitrary json-ish field to plain text ──────────
def flatten_to_text(x):
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        if "text" in x and isinstance(x["text"], str):
            return x["text"]
        return "\n".join(flatten_to_text(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return "\n".join(flatten_to_text(v) for v in x)
    return str(x)

# ── helper: map any label (“score 3” or “good answer”) → int 1-5 ────
LABEL_STRINGS = [
    "very bad answer",   # 1
    "bad answer",        # 2
    "acceptable answer", # 3
    "good answer",       # 4
    "perfect answer"     # 5
]
def label_to_int(lbl: str) -> int:
    m = re.search(r"([1-5])", lbl)
    if m:                       # the label already contains a digit
        return int(m.group(1))
    for i, s in enumerate(LABEL_STRINGS, 1):
        if s in lbl.lower():
            return i
    return 1                    # fallback

# ── datasets – full splits will be shuffled, then filtered ──────────
datasets_info = [
    ("FinQA-10K", "virattt/financial-qa-10K", "train",      False, {}),
    ("SQuAD",     "rajpurkar/squad",          "validation", False, {}),
]

# ── zero-shot classification judges (all ~125-140 M params) ─────────
candidate_labels = LABEL_STRINGS           # same list for every judge

judge1 = pipeline("zero-shot-classification",
                  model="roberta-large-mnli",
                  device=DEVICE, batch_size=BATCH_SIZE)

judge2 = pipeline("zero-shot-classification",
                  model="microsoft/deberta-base-mnli",
                  device=DEVICE, batch_size=BATCH_SIZE)

judge3 = pipeline("zero-shot-classification",
                  model="valhalla/distilbart-mnli-12-3",
                  device=DEVICE, batch_size=BATCH_SIZE)

# ── main loop ───────────────────────────────────────────────────────
rows = []

for ds_name, hf_id, split, trust_code, extra_kwargs in datasets_info:
    print(f"\n→ Loading {ds_name} ({hf_id}#{split}) and collecting {N_PER_DS} examples…")
    ds = load_dataset(hf_id, split=split, trust_remote_code=trust_code, **extra_kwargs)
    ds = ds.shuffle(seed=42)

    collected, bar = 0, tqdm(total=N_PER_DS, desc=f"{ds_name} valid")
    for ex in ds:
        # unified fields -------------------------------------------------------
        question = ex.get("question") or ex.get("question_text") or ex.get("query") or ""
        context  = flatten_to_text(
            ex.get("context") or ex.get("document_text") or ex.get("story") or ex.get("text") or ""
        )

        # keep only if context has ≥ 2 sentences -------------------------------
        if len(list(nlp(context).sents)) < 2:
            continue

        # candidate spans ------------------------------------------------------
        spans = [s for s, _ in extract_sentences_by_intent(
            text=context, intent=question, threshold=-1.0, top_k=TOP_K)]

        if not spans:                       # no hit – fill with defaults
            rows.append({
                "dataset": ds_name, "question": question, "context": context,
                "span": "", "score1": 5.0, "score2": 5.0, "score3": 5.0, "score_avg": 5.0
            })
            collected += 1; bar.update(1)
            if collected >= N_PER_DS: break
            continue

        prompts = [
            f"Question: {question}\nCandidate answer: {span}\n\n"
            "On a scale from 1 (completely wrong) to 5 (perfect), reply with a single digit."
            for span in spans
        ]

        # run judges -----------------------------------------------------------
        out1 = judge1(prompts, candidate_labels=candidate_labels, multi_label=False)
        out2 = judge2(prompts, candidate_labels=candidate_labels, multi_label=False)
        out3 = judge3(prompts, candidate_labels=candidate_labels, multi_label=False)

        j1 = [label_to_int(o["labels"][0]) for o in out1]
        j2 = [label_to_int(o["labels"][0]) for o in out2]
        j3 = [label_to_int(o["labels"][0]) for o in out3]

        avg_scores = [(a+b+c)/3.0 for a, b, c in zip(j1, j2, j3)]
        best = int(np.argmax(avg_scores))

        rows.append({
            "dataset": ds_name, "question": question, "context": context,
            "span": spans[best],
            "score1": float(j1[best]), "score2": float(j2[best]), "score3": float(j3[best]),
            "score_avg": float(avg_scores[best])
        })
        collected += 1; bar.update(1)
        if collected >= N_PER_DS:
            break
    bar.close()
    if collected < N_PER_DS:
        print(f"⚠️  Only {collected} qualifying examples found for {ds_name}")

# ── save & report ───────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
pd.DataFrame(rows).to_excel(OUTPUT_PATH, index=False)
print(f"\n✔️  Saved combined results →  {OUTPUT_PATH}")

print("\n▶︎ Per-dataset judge averages:")
summary = (pd.DataFrame(rows)
           .groupby("dataset")[["score1", "score2", "score3", "score_avg"]]
           .mean())
for ds, row in summary.iterrows():
    print(f"  {ds:12s} | "
          f"Judge1 {row['score1']:.2f}  "
          f"Judge2 {row['score2']:.2f}  "
          f"Judge3 {row['score3']:.2f}  "
          f"Combined {row['score_avg']:.2f}")
