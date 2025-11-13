## evaluate.py

import os, argparse, json
import pandas as pd
import matplotlib.pyplot as plt

def save_plot(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    assert os.path.exists(metrics_path), "metrics.csv not found"
    df = pd.read_csv(metrics_path)

    show_cols = ["model","activation","optimizer","seq_len","grad_clipping","accuracy","f1","epoch_time_sec"]
    df_round = df[show_cols].copy()
    df_round[["accuracy","f1","epoch_time_sec"]] = df_round[["accuracy","f1","epoch_time_sec"]].round(4)
    summary_path = os.path.join(args.results_dir, "summary_rounded.csv")
    df_round.to_csv(summary_path, index=False)
    
    print("\nSummary (first 20 rows)")
    print(df_round.head(20).to_string(index=False))

    agg_dir = os.path.join(args.results_dir, "aggregates")
    os.makedirs(agg_dir, exist_ok=True)

    for metric in ["accuracy", "f1"]:
        pivot = (df.groupby(["model","seq_len"])[metric]
                   .mean().reset_index()
                   .pivot(index="seq_len", columns="model", values=metric)
                   .sort_index())
        pivot.to_csv(os.path.join(agg_dir, f"{metric}_vs_len_by_model.csv"))

        for arch in sorted(df["model"].unique()):
            sub = df[df["model"] == arch].groupby("seq_len")[metric].mean().reset_index()
            plt.figure()
            plt.plot(sub["seq_len"], sub[metric], marker="o")
            plt.xlabel("Sequence length"); plt.ylabel(metric.upper())
            plt.title(f"{metric.upper()} vs. sequence length — {arch}")
            save_plot(os.path.join(args.results_dir, "plots", f"{metric}_vs_len_{arch}.png"))

    
    for metric in ["accuracy", "f1"]:
        by_act = (df.groupby(["activation","seq_len"])[metric]
                    .mean().reset_index()
                    .pivot(index="seq_len", columns="activation", values=metric)
                    .sort_index())
        by_opt = (df.groupby(["optimizer","seq_len"])[metric]
                    .mean().reset_index()
                    .pivot(index="seq_len", columns="optimizer", values=metric)
                    .sort_index())
        by_act.to_csv(os.path.join(agg_dir, f"{metric}_vs_len_by_activation.csv"))
        by_opt.to_csv(os.path.join(agg_dir, f"{metric}_vs_len_by_optimizer.csv"))


    
    best = df.sort_values("f1", ascending=False).iloc[0]
    worst = df.sort_values("f1", ascending=True).iloc[0]
    print("\nBest by F1")
    print(best[show_cols].to_string())
    print("\nWorst by F1")
    print(worst[show_cols].to_string())

    best_worst = {
        "best": {k: (float(v) if isinstance(v, (int,float)) else v) for k, v in best.to_dict().items()},
        "worst": {k: (float(v) if isinstance(v, (int,float)) else v) for k, v in worst.to_dict().items()},
    }
    with open(os.path.join(args.results_dir, "best_worst.json"), "w") as f:
        json.dump(best_worst, f, indent=2)

    ## Training loss curves for best/worst
    for tag, row in [("best", best), ("worst", worst)]:
        run_id = f"{row['model']}-{row['activation']}-{row['optimizer']}-len{int(row['seq_len'])}-clip{int(row['grad_clipping'])}-seed{int(row['seed'])}"
        log_path = os.path.join(args.results_dir, "logs", f"{run_id}.csv")
        if os.path.exists(log_path):
            log = pd.read_csv(log_path)
            plt.figure()
            plt.plot(log["epoch"], log["train_loss"], marker="o")
            plt.xlabel("Epoch"); plt.ylabel("Training Loss")
            plt.title(f"Training Loss — {tag.upper()} ({run_id})")
            save_plot(os.path.join(args.results_dir, "plots", f"train_loss_{tag}.png"))

if __name__ == "__main__":
    main()
