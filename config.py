# config.py
import torch
import numpy as np
from transformers import AutoTokenizer
import spacy

MODEL_NAME      = "bert-base-uncased"
MAX_LENGTH      = 128
OVERLAP         = 32
PREPROCESSED_DIR= "preprocessed_snli"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
nlp       = spacy.load("en_core_web_sm")
