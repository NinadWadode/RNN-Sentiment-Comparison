## visualize.py

import os, json, glob, argparse, base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image  

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_plot(path):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def run(args):
    BASE = args.base
    RES  = os.path.join(BASE, "results")
    PLOT_DIR = os.path.join(RES, "plots")
    AGG_DIR  = os.path.join(RES, "aggregates")
    PREDS_DIR= os.path.join(RES, "preds")
    LOG_DIR  = os.path.join(RES, "logs")
    DATA_DIR = os.path.join(BASE, "data")

    assert os.path.exists(os.path.join(RES, "metrics.csv")), "metrics.csv not found"
    assert os.path.exists(os.path.join(RES, "summary_rounded.csv")), "summary_rounded.csv not found"

    ## Dataset stats 
    stats_path = os.path.join(DATA_DIR, "dataset_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            ds_stats = json.load(f)
        print("Dataset stats:")
        display(pd.DataFrame([ds_stats]))  
    else:
        print("dataset_stats.json not found")

    ## Full metrics + rounded summary
    metrics = pd.read_csv(os.path.join(RES, "metrics.csv"))
    summary = pd.read_csv(os.path.join(RES, "summary_rounded.csv"))

    print("\nSummary (first 20 rows):")
    print(summary.head(20).to_string(index=False))

    print("\nTop 15 runs by F1:")
    top15 = metrics.sort_values("f1", ascending=False).head(15)
    print(top15[["model","activation","optimizer","seq_len","grad_clipping","accuracy","f1","epoch_time_sec"]]
          .to_string(index=False))

    ## Aggregates (accuracy/F1 vs seq length by model/activation/optimizer)
    if os.path.isdir(AGG_DIR):
        agg_files = sorted(glob.glob(os.path.join(AGG_DIR, "*.csv")))
        if agg_files:
            print("\nAggregates available:")
            for path in agg_files:
                print(" -", os.path.basename(path))
                display(pd.read_csv(path))
        else:
            print("No aggregates found in results/aggregates/")
    else:
        print("Aggregates dir not found")

    ## Best and worst metadata
    bw_path = os.path.join(RES, "best_worst.json")
    bw = None
    if os.path.exists(bw_path):
        with open(bw_path) as f:
            bw = json.load(f)
        print("\nBest (by F1):")
        print(pd.DataFrame([bw["best"]])[["model","activation","optimizer","seq_len","grad_clipping","accuracy","f1","epoch_time_sec"]]
              .to_string(index=False))
        print("\nWorst (by F1):")
        print(pd.DataFrame([bw["worst"]])[["model","activation","optimizer","seq_len","grad_clipping","accuracy","f1","epoch_time_sec"]]
              .to_string(index=False))
    else:
        print("best_worst.json not found")

    ## Pareto trade-off: F1 vs epoch time (marker=arch, color=optimizer, edge=clipping)
    print("\nPlot: pareto_f1_time.png")
    plt.figure(figsize=(7,5))
    markers = {"rnn":"o","lstm":"s","bilstm":"^"}
    colors  = {"adam":"tab:blue","sgd":"tab:orange","rmsprop":"tab:green"}
    for _, r in metrics.iterrows():
        m = markers.get(r.model, "o")
        c = colors.get(r.optimizer, "tab:gray")
        edge = "black" if bool(r.grad_clipping) else None
        plt.scatter(r.epoch_time_sec, r.f1, marker=m, c=c, edgecolors=edge, alpha=0.7, s=45)
    plt.xlabel("Avg epoch time (s)")
    plt.ylabel("F1 (macro)")
    plt.title("F1 vs Epoch Time — marker=arch, color=optimizer, edge=clipping")
    plt.grid(True, alpha=0.3)
    save_plot(os.path.join(PLOT_DIR, "pareto_f1_time.png"))

    ## Heatmaps: mean F1 across seq_len × optimizer (one per architecture)
    print("Plots: heatmap_f1_<arch>.png")
    for arch in ["rnn","lstm","bilstm"]:
        sub = metrics[metrics.model==arch].groupby(["seq_len","optimizer"])["f1"].mean().unstack()
        plt.figure(figsize=(5,3.2))
        im = plt.imshow(sub.values, aspect="auto")
        plt.xticks(range(sub.shape[1]), sub.columns)
        plt.yticks(range(sub.shape[0]), sub.index)
        plt.colorbar(im, fraction=0.046, pad=0.04, label="mean F1")
        plt.title(f"{arch.upper()}: mean F1 by seq_len × optimizer")
        plt.xlabel("optimizer"); plt.ylabel("seq_len")
        save_plot(os.path.join(PLOT_DIR, f"heatmap_f1_{arch}.png"))

    ## ΔF1 with vs without gradient clipping (paired by model/activation/optimizer/seq_len)
    print("Plot: delta_f1_clipping.png")
    keys = ["model","activation","optimizer","seq_len"]
    base = metrics[metrics.grad_clipping==False].set_index(keys)["f1"]
    clip = metrics[metrics.grad_clipping==True ].set_index(keys)["f1"]

    pair = pd.DataFrame({"f1_base": base}).join(pd.DataFrame({"f1_clip": clip}), how="inner")
    if len(pair) == 0:
        print("No paired rows for clipping delta")
    else:
        pair = pair.reset_index()
        pair["delta"] = pair["f1_clip"] - pair["f1_base"]
        pair["cond"] = pair["model"] + "-" + pair["optimizer"] + "-L" + pair["seq_len"].astype(str)
        g = pair.groupby("cond")["delta"].mean().sort_values()

        plt.figure(figsize=(max(8, 0.35*len(g)), 4))
        plt.bar(range(len(g)), g.values)
        plt.axhline(0, color="k", lw=1)
        plt.xticks(range(len(g)), g.index, rotation=90)
        plt.ylabel("ΔF1 (clip1 - clip0)")
        plt.title("Effect of gradient clipping by model/optimizer/seq_len (mean over activations)")
        save_plot(os.path.join(PLOT_DIR, "delta_f1_clipping.png"))

    ## Overall behavior bar charts (means across runs)
    print("Plots: overall_mean_f1_by_arch.png, overall_mean_f1_by_optimizer.png, overall_mean_f1_by_activation.png")
    for col, out in [("model","overall_mean_f1_by_arch.png"),
                     ("optimizer","overall_mean_f1_by_optimizer.png"),
                     ("activation","overall_mean_f1_by_activation.png")]:
        g = metrics.groupby(col)["f1"].mean().sort_values(ascending=False)
        plt.figure(figsize=(5,3))
        plt.bar(g.index.astype(str), g.values)
        plt.ylabel("mean F1")
        plt.title(f"Mean F1 by {col}")
        save_plot(os.path.join(PLOT_DIR, out))

    ## ROC + PR for best and worst runs (requires preds)
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
    def run_id_from_row(row):
        return f"{row['model']}-{row['activation']}-{row['optimizer']}-len{int(row['seq_len'])}-clip{int(row['grad_clipping'])}-seed{int(row['seed'])}"

    if bw is not None:
        for tag in ["best","worst"]:
            row = bw[tag]
            run_id = run_id_from_row(row)
            preds_path = os.path.join(PREDS_DIR, f"{run_id}.csv")
            if os.path.exists(preds_path):
                dfp = pd.read_csv(preds_path)
                y, p = dfp.y_true.values, dfp.y_prob.values

                ## ROC
                fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
                plt.figure(); plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
                plt.plot([0,1],[0,1],'--',lw=1,color='gray'); plt.xlabel("FPR"); plt.ylabel("TPR")
                plt.title(f"ROC — {tag.upper()} ({run_id})"); plt.legend()
                save_plot(os.path.join(PLOT_DIR, f"roc_{tag}.png"))

                ## PR
                pr, rc, _ = precision_recall_curve(y, p)
                ap = average_precision_score(y, p)
                plt.figure(); plt.plot(rc, pr, label=f"AP={ap:.3f}")
                plt.xlabel("Recall"); plt.ylabel("Precision")
                plt.title(f"PR — {tag.upper()} ({run_id})"); plt.legend()
                save_plot(os.path.join(PLOT_DIR, f"pr_{tag}.png"))

                ## Calibration (reliability) for best only
                if tag == "best":
                    bins = np.linspace(0,1,11)
                    idx = np.digitize(p, bins) - 1
                    bin_conf = []; bin_acc = []; centers = []
                    for b in range(10):
                        mask = idx==b
                        centers.append((bins[b]+bins[b+1])/2)
                        if mask.sum()==0:
                            bin_conf.append(np.nan); bin_acc.append(np.nan)
                        else:
                            bin_conf.append(p[mask].mean()); bin_acc.append(y[mask].mean())
                    plt.figure(figsize=(5,4))
                    plt.plot([0,1],[0,1],'--',color='gray',lw=1)
                    plt.plot(bin_conf, bin_acc, marker='o')
                    plt.xlabel("Predicted probability (bin mean)")
                    plt.ylabel("Empirical accuracy")
                    plt.title(f"Reliability — best run ({run_id})")
                    save_plot(os.path.join(PLOT_DIR, "calibration_best.png"))

                    ## Threshold sweep for F1
                    ths = np.linspace(0.1, 0.9, 33)
                    f1s = [f1_score(y, (p>=t).astype(int), average="macro") for t in ths]
                    t_star = ths[int(np.argmax(f1s))]
                    plt.figure(figsize=(6,3.5))
                    plt.plot(ths, f1s, marker='o')
                    plt.axvline(0.5, color='gray', lw=1, ls='--', label='0.5')
                    plt.axvline(t_star, color='tab:red', lw=1, ls='--', label=f'best≈{t_star:.2f}')
                    plt.xlabel("Threshold"); plt.ylabel("F1 (macro)")
                    plt.title(f"Threshold sweep — {run_id}")
                    plt.legend()
                    save_plot(os.path.join(PLOT_DIR, "threshold_sweep_best.png"))
            else:
                print(f"Predictions not found for {tag} run: {preds_path}")

        ## Confusion matrices for best and worst
        for tag in ["best","worst"]:
            row = bw[tag]
            run_id = run_id_from_row(row)
            preds_path = os.path.join(PREDS_DIR, f"{run_id}.csv")
            if os.path.exists(preds_path):
                dfp = pd.read_csv(preds_path)
                y, yhat = dfp.y_true.values, dfp.y_pred.values
                cm = np.zeros((2,2), dtype=int)
                for yt, yp in zip(y,yhat): cm[int(yt), int(yp)] += 1
                plt.figure(figsize=(3.6,3.2))
                plt.imshow(cm, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        plt.text(j,i,str(cm[i,j]), ha="center", va="center", fontsize=11)
                plt.xticks([0,1],["Neg","Pos"]); plt.yticks([0,1],["Neg","Pos"])
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.title(f"Confusion — {tag.upper()} ({run_id})")
                save_plot(os.path.join(PLOT_DIR, f"cm_{tag}.png"))

    ## Interaction line plot: mean F1 vs seq_len per optimizer (for each architecture)
    print("Plots: f1_vs_len_by_optimizer_<arch>.png")
    for arch in ["rnn","lstm","bilstm"]:
        sub = metrics[metrics.model==arch]
        g = sub.groupby(["seq_len","optimizer"])["f1"].mean().reset_index()
        plt.figure(figsize=(6,4))
        for opt, dfopt in g.groupby("optimizer"):
            dfopt = dfopt.sort_values("seq_len")
            plt.plot(dfopt["seq_len"], dfopt["f1"], marker="o", label=opt)
        plt.xlabel("Sequence length")
        plt.ylabel("mean F1")
        plt.title(f"{arch.upper()}: mean F1 vs length by optimizer")
        plt.legend()
        save_plot(os.path.join(PLOT_DIR, f"f1_vs_len_by_optimizer_{arch}.png"))

    ## Interaction line plot: mean F1 vs seq_len per activation (for each architecture)
    print("Plots: f1_vs_len_by_activation_<arch>.png")
    for arch in ["rnn","lstm","bilstm"]:
        sub = metrics[metrics.model==arch]
        g = sub.groupby(["seq_len","activation"])["f1"].mean().reset_index()
        plt.figure(figsize=(6,4))
        for act, dfact in g.groupby("activation"):
            dfact = dfact.sort_values("seq_len")
            plt.plot(dfact["seq_len"], dfact["f1"], marker="o", label=act)
        plt.xlabel("Sequence length")
        plt.ylabel("mean F1")
        plt.title(f"{arch.upper()}: mean F1 vs length by activation")
        plt.legend()
        save_plot(os.path.join(PLOT_DIR, f"f1_vs_len_by_activation_{arch}.png"))

    
    pngs = sorted(glob.glob(os.path.join(PLOT_DIR, "*.png")))
    print("Found", len(pngs), "plots in", PLOT_DIR)
    for p in pngs:
        print(os.path.basename(p))
        display(Image(filename=p))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate summary tables and plots for the RNN sentiment grid")
    ap.add_argument("--base", default="/content/rnn-sentiment-comparison", help="Project base path")
    args = ap.parse_args()
    run(args)
