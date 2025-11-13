## train.py
## Sweep the full experimental grid 
##   Architecture: RNN, LSTM, BiLSTM
##   Activation: relu, tanh, sigmoid
##   Optimizer: Adam, SGD, RMSprop
##   Sequence Length: 25, 50, 100
##   Gradient clipping: off/on (max_norm=1.0)

import os, argparse, itertools, time
import numpy as np, pandas as pd
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from utils import set_seeds, load_json, accuracy_from_probs, save_json, confusion_counts
from models import make_model

OPTIMIZERS = {
    "adam":    lambda p, lr: torch.optim.Adam(p, lr=lr),
    "sgd":     lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "rmsprop": lambda p, lr: torch.optim.RMSprop(p, lr=lr, momentum=0.9),
}

def get_optimizer(name, params, lr):
    name = name.lower()
    if name not in OPTIMIZERS: raise ValueError(f"Unknown optimizer: {name}")
    return OPTIMIZERS[name](params, lr)

def train_one_epoch(model, loader, criterion, optimizer, device, clip_val=None):
    model.train(); epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.float().to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        if clip_val is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    return epoch_loss / len(loader.dataset)

@torch.no_grad()
def evaluate_with_outputs(model, loader, device, thresh=0.5):
    model.eval(); probs, ys = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p); ys.append(yb.numpy())
    probs = np.concatenate(probs, axis=0)
    ys = np.concatenate(ys, axis=0)
    preds = (probs >= thresh).astype(int)
    acc = accuracy_from_probs(probs, ys, thresh=thresh)
    f1  = f1_score(ys, preds, average="macro")
    return acc, f1, probs, ys, preds

def build_loaders(data_dir, seq_len, batch_size):
    X_train = np.load(os.path.join(data_dir, f"X_train_len{seq_len}.npy"))
    X_test  = np.load(os.path.join(data_dir, f"X_test_len{seq_len}.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"))
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds  = TensorDataset(torch.from_numpy(X_test),  torch.from_numpy(y_test))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--clip_val", type=float, default=1.0)
    ap.add_argument("--limit_runs", type=int, default=None)  
    args = ap.parse_args()

    set_seeds(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "preds"), exist_ok=True)


    vocab = load_json(os.path.join(args.data_dir, "vocab.json"))
    vocab_size = len(vocab["itos"])
    device = torch.device(args.device)

    architectures = ["rnn", "lstm", "bilstm"]
    activations   = ["relu", "tanh", "sigmoid"]
    optimizers    = ["adam", "sgd", "rmsprop"]
    seq_lengths   = [25, 50, 100]
    clip_options  = [False, True]

    grid = list(itertools.product(architectures, activations, optimizers, seq_lengths, clip_options))
    if args.limit_runs: grid = grid[:args.limit_runs]

    metrics_csv = os.path.join(args.results_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        pd.DataFrame(columns=[
            "model","activation","optimizer","seq_len","grad_clipping",
            "accuracy","f1","epoch_time_sec","epochs","batch_size","lr","dropout","seed","device"
        ]).to_csv(metrics_csv, index=False)

    for (arch, act, opt_name, L, do_clip) in tqdm(grid, desc="grid"):
        train_loader, test_loader = build_loaders(args.data_dir, L, args.batch_size)
        model = make_model(arch, vocab_size, activation=act, dropout=args.dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = get_optimizer(opt_name, model.parameters(), args.lr)
        clip_val = args.clip_val if do_clip else None

        run_id = f"{arch}-{act}-{opt_name}-len{L}-clip{int(do_clip)}-seed{args.seed}"
        log_path = os.path.join(args.results_dir, "logs", f"{run_id}.csv")
        preds_path = os.path.join(args.results_dir, "preds", f"{run_id}.csv")
        conf_path = os.path.join(args.results_dir, "preds", f"{run_id}.json")


        loss_log, epoch_times = [], []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, clip_val)
            et = time.time() - t0
            epoch_times.append(et)
            loss_log.append({"epoch": epoch, "train_loss": train_loss, "epoch_time_sec": et})

        acc, f1, probs, ys, preds = evaluate_with_outputs(model, test_loader, device, thresh=0.5)

        pd.DataFrame(loss_log).to_csv(log_path, index=False)

        pd.DataFrame({
            "y_true": ys.astype(int),
            "y_prob": probs.astype(float),
            "y_pred": preds.astype(int),
        }).to_csv(preds_path, index=False)

        conf = confusion_counts(ys, preds)
        conf.update({
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "n_samples": int(len(ys)),
            "run_id": run_id,
        })
        save_json(conf, conf_path)

        pd.DataFrame([{
            "model": arch, "activation": act, "optimizer": opt_name, "seq_len": L,
            "grad_clipping": do_clip, "accuracy": float(acc), "f1": float(f1),
            "epoch_time_sec": float(np.mean(epoch_times)), "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "dropout": args.dropout,
            "seed": args.seed, "device": str(device)
        }]).to_csv(metrics_csv, mode="a", header=False, index=False)

        print(f"[run] {run_id} -> acc={acc:.4f} f1={f1:.4f} epoch_time={np.mean(epoch_times):.2f}s")

if __name__ == "__main__":
    main()
