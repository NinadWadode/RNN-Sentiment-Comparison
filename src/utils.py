## utils.py

import re, random, json, time
import numpy as np
import torch

def set_seeds(seed: int = 42):
    ## Fixed seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clean_text(s: str) -> str:
    ## IMDb contains HTML breaks and punctuation
    s = s.lower()
    s = re.sub(r"<br\\s*/?>", " ", s)
    s = re.sub(r"[^a-z0-9\\s']", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def accuracy_from_probs(y_prob, y_true, thresh=0.5):
    y_hat = (y_prob >= thresh).astype(np.int64)
    return (y_hat == y_true).mean()

def confusion_counts(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
