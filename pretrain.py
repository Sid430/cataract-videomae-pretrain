import os
import sys
import math
import json
import time
import argparse
 
import torch
from torch.utils.data import DataLoader
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.videomae_v2 import videomae_v2_base, videomae_v2_tiny
from data.datasets import DummyVideoDataset, CataractVideoDataset
from plot_utils import make_curves_pdf
 
 
def build_args():
    p = argparse.ArgumentParser("VideoMAE v2 pretraining")
    # data
    p.add_argument("--dummy", action="store_true", help="use synthetic clips")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-mode", choices=["video", "frames"], default="video")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--sampling-stride", type=int, default=4)
    # model
    p.add_argument("--model", choices=["base", "tiny"], default="base")
    p.add_argument("--enc-mask-ratio", type=float, default=0.9)
    p.add_argument("--dec-mask-ratio", type=float, default=0.5,
                   help="dual masking: fraction of masked tokens dropped from decoder")
    # optim
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="cap steps/epoch (handy for dummy runs)")
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=40)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    # io
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    # early stopping (on the training reconstruction loss, which is the only
    # signal in self-supervised pretraining). 0 = disabled.
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop if avg epoch loss doesn't improve for this many "
                        "epochs (0 disables early stopping)")
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4,
                   help="minimum loss improvement to count as progress")
    p.add_argument("--no-plot", action="store_true",
                   help="skip PDF generation (metadata is still saved)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()
 
 
def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))
 
 
class MetricLogger:
    """Streams per-step and per-epoch metrics to disk (crash-safe append)."""
 
    def __init__(self, log_dir, args):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        # record the run config so figures/analysis are self-describing
        with open(os.path.join(log_dir, "run_meta.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        self.step_f = open(os.path.join(log_dir, "metrics.jsonl"), "a")
        self.epoch_path = os.path.join(log_dir, "epoch_log.csv")
        if not os.path.exists(self.epoch_path) or os.path.getsize(self.epoch_path) == 0:
            with open(self.epoch_path, "w") as f:
                f.write("epoch,avg_loss,time_sec\n")
 
    def log_step(self, gstep, epoch, step, loss, lr):
        self.step_f.write(json.dumps({"gstep": gstep, "epoch": epoch,
                                      "step": step, "loss": loss, "lr": lr}) + "\n")
        self.step_f.flush()  # so metadata survives an interruption
 
    def log_epoch(self, epoch, avg_loss, time_sec):
        with open(self.epoch_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{time_sec:.2f}\n")
 
    def close(self):
        self.step_f.close()
 
 
def main():
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    logger = MetricLogger(args.out_dir, args)
 
    # ---- data ----
    if args.dummy:
        ds = DummyVideoDataset(length=max(args.batch_size * 8, 64),
                               num_frames=args.num_frames, img_size=args.img_size)
    else:
        assert args.data_root, "--data-root required unless --dummy"
        ds = CataractVideoDataset(args.data_root, mode=args.data_mode,
                                  num_frames=args.num_frames, img_size=args.img_size,
                                  sampling_stride=args.sampling_stride)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0 if args.dummy else args.num_workers,
                        pin_memory=(device.type == "cuda"), drop_last=True)
 
    # ---- model ----
    model_fn = videomae_v2_tiny if args.model == "tiny" else videomae_v2_base
    model = model_fn(num_frames=args.num_frames, img_size=args.img_size,
                     enc_mask_ratio=args.enc_mask_ratio,
                     dec_mask_ratio=args.dec_mask_ratio).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] VideoMAE-v2-{args.model} | {n_params:.1f}M params | "
          f"{model.num_patches} tokens | device={device}")
 
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
 
    steps_per_epoch = args.steps_per_epoch or len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
 
    def maybe_plot():
        if not args.no_plot:
            try:
                make_curves_pdf(args.out_dir)
            except Exception as e:
                print(f"[plot] skipped ({e}); metadata is saved, use plot_curves.py")
 
    # ---- train ----
    model.train()
    gstep = 0
    best_loss = float("inf")   # best avg epoch loss so far (for early stopping)
    epochs_no_improve = 0
    try:
        for epoch in range(args.epochs):
            t0 = time.time()
            running = 0.0
            for i, clip in enumerate(loader):
                if i >= steps_per_epoch:
                    break
                clip = clip.to(device, non_blocking=True)
                lr = cosine_lr(gstep, total_steps, warmup_steps, args.lr, args.min_lr)
                for g in opt.param_groups:
                    g["lr"] = lr
 
                opt.zero_grad()
                with torch.autocast(device_type=device.type,
                                    enabled=args.amp and device.type == "cuda"):
                    loss, _ = model(clip)
                scaler.scale(loss).backward()
                if args.clip_grad:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()
 
                loss_val = loss.item()
                running += loss_val
                logger.log_step(gstep, epoch, i + 1, loss_val, lr)
                gstep += 1
                if gstep % args.log_every == 0:
                    print(f"  epoch {epoch} step {i+1}/{steps_per_epoch} "
                          f"loss {loss_val:.4f} lr {lr:.2e}")
 
            avg = running / min(steps_per_epoch, len(loader))
            dt = time.time() - t0
            logger.log_epoch(epoch, avg, dt)
            print(f"[epoch {epoch}] avg_loss {avg:.4f} time {dt:.1f}s")
 
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                ckpt = os.path.join(args.out_dir,
                                    f"videomae_v2_{args.model}_ep{epoch+1}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "args": vars(args)}, ckpt)
                print(f"[ckpt] saved {ckpt}")
                maybe_plot()  # refresh the curve PDF as training progresses
 
            # ---- early stopping (on avg training loss) ----
            if avg < best_loss - args.early_stop_min_delta:
                best_loss = avg
                epochs_no_improve = 0
                # always keep the best model separately
                best_path = os.path.join(args.out_dir,
                                         f"videomae_v2_{args.model}_best.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "avg_loss": avg, "args": vars(args)}, best_path)
            else:
                epochs_no_improve += 1
 
            if args.early_stop_patience and epochs_no_improve >= args.early_stop_patience:
                print(f"[early-stop] no improvement > {args.early_stop_min_delta} "
                      f"for {args.early_stop_patience} epochs "
                      f"(best avg_loss {best_loss:.4f}). Stopping at epoch {epoch}.")
                # save a final checkpoint at the stopping point
                stop_ckpt = os.path.join(args.out_dir,
                                         f"videomae_v2_{args.model}_ep{epoch+1}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "args": vars(args)}, stop_ckpt)
                print(f"[ckpt] saved {stop_ckpt}")
                break
    finally:
        logger.close()
        maybe_plot()  # always leave a final figure, even on Ctrl-C / crash
 
    print("done.")
 
 
if __name__ == "__main__":
    main()
