import os
import csv
import json


def load_step_metrics(log_dir):
    path = os.path.join(log_dir, "metrics.jsonl")
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a half-written last line after a crash
    return recs


def load_epoch_metrics(log_dir):
    path = os.path.join(log_dir, "epoch_log.csv")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({"epoch": int(row["epoch"]),
                         "avg_loss": float(row["avg_loss"]),
                         "time_sec": float(row["time_sec"])})
    return rows


def _smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    out, acc = [], 0.0
    from collections import deque
    q = deque()
    for v in values:
        q.append(v)
        acc += v
        if len(q) > window:
            acc -= q.popleft()
        out.append(acc / len(q))
    return out


def make_curves_pdf(log_dir, out_path=None, smooth_window=50):
    """Render learning curves to a PDF. Returns the output path, or None if
    matplotlib is unavailable or there's nothing to plot."""
    steps = load_step_metrics(log_dir)
    epochs = load_epoch_metrics(log_dir)
    if not steps and not epochs:
        print(f"[plot] no metrics found in {log_dir}")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — metadata is saved; run later:\n"
              "       pip install matplotlib && python plot_curves.py --log-dir "
              f"{log_dir}")
        return None

    out_path = out_path or os.path.join(log_dir, "learning_curve.pdf")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # 1) per-step training loss (raw + smoothed)
    if steps:
        gstep = [r["gstep"] for r in steps]
        loss = [r["loss"] for r in steps]
        axes[0].plot(gstep, loss, lw=0.6, alpha=0.35, label="loss (raw)")
        axes[0].plot(gstep, _smooth(loss, smooth_window), lw=1.6,
                     label=f"loss (smoothed, w={smooth_window})")
        axes[0].set_xlabel("global step")
        axes[0].set_ylabel("reconstruction loss (MSE)")
        axes[0].set_title("Training loss per step")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

    # 2) learning-rate schedule
    if steps:
        axes[1].plot([r["gstep"] for r in steps], [r["lr"] for r in steps], lw=1.4)
        axes[1].set_xlabel("global step")
        axes[1].set_ylabel("learning rate")
        axes[1].set_title("Learning-rate schedule")
        axes[1].grid(alpha=0.3)

    # 3) per-epoch average loss
    if epochs:
        axes[2].plot([e["epoch"] for e in epochs], [e["avg_loss"] for e in epochs],
                     marker="o", ms=3, lw=1.4)
        axes[2].set_xlabel("epoch")
        axes[2].set_ylabel("avg reconstruction loss")
        axes[2].set_title("Average training loss per epoch")
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "no epoch summaries yet", ha="center", va="center")
        axes[2].axis("off")

    # annotate with run config if present
    meta_path = os.path.join(log_dir, "run_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        cap = (f"model={meta.get('model')}  img={meta.get('img_size')}  "
               f"frames={meta.get('num_frames')}  bs={meta.get('batch_size')}  "
               f"lr={meta.get('lr')}  enc_mask={meta.get('enc_mask_ratio')}  "
               f"dec_mask={meta.get('dec_mask_ratio')}")
        fig.suptitle("VideoMAE v2 pretraining — cataract", fontsize=12)
        fig.text(0.5, 0.005, cap, ha="center", fontsize=7, color="0.35")

    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path
