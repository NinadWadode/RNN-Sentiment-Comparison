## preprocess.py

import os, argparse, numpy as np, pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
from utils import clean_text, save_json, set_seeds

PAD = "<pad>"
UNK = "<unk>"

def build_vocab(tokenized_texts, vocab_size=10000):
    counter = Counter()
    for toks in tokenized_texts:
        counter.update(toks)
    most_common = [w for w, _ in counter.most_common(vocab_size - 2)]
    itos = [PAD, UNK] + most_common
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def tokens_to_ids(tokens, stoi):
    unk = stoi[UNK]
    return [stoi.get(t, unk) for t in tokens]

def pad_or_truncate(seq, max_len, pad_idx):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_idx] * (max_len - len(seq))

def label_to_int(y):
    return 1 if y.strip().lower() == "positive" else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default=os.path.join("data", "IMDB_Dataset.csv"))
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--vocab_size", type=int, default=10000)
    ap.add_argument("--seq_lengths", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seeds(args.seed)
    nltk.download("punkt", quiet=True)

    assert os.path.exists(args.data_csv), f"Missing file: {args.data_csv}"
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    assert {"review", "sentiment"}.issubset(df.columns), "Expected columns: review, sentiment"

    ## Deterministic 25k/25k split by row order
    df_train = df.iloc[:25000].copy()
    df_test  = df.iloc[25000:].copy()

    df_train["clean"] = df_train["review"].apply(clean_text)
    df_test["clean"]  = df_test["review"].apply(clean_text)

    df_train["tokens"] = df_train["clean"].apply(word_tokenize)
    df_test["tokens"]  = df_test["clean"].apply(word_tokenize)

    ## Build vocab from train only; use top 10k (including PAD/UNK)
    stoi, itos = build_vocab(df_train["tokens"].tolist(), vocab_size=args.vocab_size)

    y_train = df_train["sentiment"].apply(label_to_int).to_numpy(dtype=np.int64)
    y_test  = df_test["sentiment"].apply(label_to_int).to_numpy(dtype=np.int64)
    save_json({"stoi": stoi, "itos": itos}, os.path.join(args.out_dir, "vocab.json"))

    pad_idx = stoi[PAD]
    for L in args.seq_lengths:
        X_train = np.stack([np.array(pad_or_truncate(tokens_to_ids(toks, stoi), L, pad_idx), dtype=np.int64)
                            for toks in df_train["tokens"]])
        X_test  = np.stack([np.array(pad_or_truncate(tokens_to_ids(toks, stoi), L, pad_idx), dtype=np.int64)
                            for toks in df_test["tokens"]])
        np.save(os.path.join(args.out_dir, f"X_train_len{L}.npy"), X_train)
        np.save(os.path.join(args.out_dir, f"X_test_len{L}.npy"),  X_test)

    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "y_test.npy"),  y_test)

    lens = df_train["tokens"].apply(len).tolist() + df_test["tokens"].apply(len).tolist()

    stats = {
        "vocab_size": len(itos),
        "train_size": int(len(df_train)),
        "test_size": int(len(df_test)),
        "avg_len": float(np.mean(lens)),
        "median_len": float(np.median(lens)),
        "p90_len": float(np.percentile(lens, 90)),
        "max_len": int(np.max(lens)),
    }
    save_json(stats, os.path.join(args.out_dir, "dataset_stats.json"))
    print(f"[preprocess] saved stats to {os.path.join(args.out_dir, 'dataset_stats.json')}")

    print(f"[preprocess] vocab size: {len(itos)} (top 10k with PAD/UNK)")
    print(f"[preprocess] dataset sizes: train={len(df_train)}, test={len(df_test)}")
    print(f"[preprocess] avg len={np.mean(lens):.1f}, median={np.median(lens):.1f}, p90={np.percentile(lens,90):.1f}, max={np.max(lens)}")

if __name__ == "__main__":
    main()
