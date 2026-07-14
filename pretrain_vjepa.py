import os
import sys
import math
import json
import time
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vjepa2 import vjepa2_base, vjepa2_tiny
from data.datasets import DummyVideoDataset, CataractVideoDataset
from plot_utils import make_curves_pdf


def build_args():
    p = argparse.ArgumentParser("V-JEPA 2 pretraining")
    p.add_argument("--dummy", action="store_true")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--data-mode", choices=["video", "frames"], default="video")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--sampling-stride", type=int, default=4)
    # model
    p.add_argument("--model", choices=["base", "tiny"], default="base")
    p.add_argument("--mask-ratio", type=float, default=0.9)
    # optim
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps-per-epoch", type=int, default=None)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=15)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="gradient accumulation steps (effective batch = batch * accum)")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="use activation checkpointing on encoder+predictor blocks "
                        "(cuts memory ~2x at ~30%% extra compute)")
    # EMA schedule for the target encoder
    p.add_argument("--ema-start", type=float, default=0.996)
    p.add_argument("--ema-end", type=float, default=1.0)
    # io
    p.add_argument("--out-dir", type=str, default="./checkpoints_vjepa")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))


def ema_momentum(step, total_steps, m0, m1):
    # increase momentum from m0 -> m1 over training (cosine)
    prog = min(1.0, step / max(1, total_steps))
    return m1 - 0.5 * (m1 - m0) * (1 + math.cos(math.pi * prog))


class MetricLogger:
    def __init__(self, log_dir, args):
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "run_meta.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        self.step_f = open(os.path.join(log_dir, "metrics.jsonl"), "a")
        self.epoch_path = os.path.join(log_dir, "epoch_log.csv")
        if not os.path.exists(self.epoch_path) or os.path.getsize(self.epoch_path) == 0:
            with open(self.epoch_path, "w") as f:
                f.write("epoch,avg_loss,time_sec\n")

    def log_step(self, gstep, epoch, step, loss, lr):
        self.step_f.write(json.dumps({"gstep": gstep, "epoch": epoch, "step": step,
                                      "loss": loss, "lr": lr}) + "\n")
        self.step_f.flush()

    def log_epoch(self, epoch, avg_loss, time_sec):
        with open(self.epoch_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f},{time_sec:.2f}\n")

    def close(self):
        self.step_f.close()


def main():
    args = build_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    logger = MetricLogger(args.out_dir, args)

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

    model_fn = vjepa2_tiny if args.model == "tiny" else vjepa2_base
    model = model_fn(num_frames=args.num_frames, img_size=args.img_size,
                     mask_ratio=args.mask_ratio).to(device)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[model] V-JEPA2-{args.model} | {n_tr:.1f}M trainable | "
          f"{model.num_patches} tokens | device={device}")

    if args.grad_checkpoint:
        # wrap RoPEBlock.forward to use checkpoint(). Trades ~30% compute for
        # ~half the activation memory. Applied to online encoder + predictor;
        # target encoder is under no_grad so checkpointing wouldn't help there.
        from torch.utils.checkpoint import checkpoint
        def wrap(block):
            orig = block.forward
            block.forward = lambda x, cos, sin: checkpoint(
                orig, x, cos, sin, use_reentrant=False)
        for b in model.encoder.blocks:
            wrap(b)
        for b in model.predictor_blocks:
            wrap(b)
        print("[mem] gradient checkpointing enabled on encoder + predictor")

    # only the online encoder + predictor are optimized (target is EMA)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    steps_per_epoch = args.steps_per_epoch or len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    print(f"[optim] batch={args.batch_size} accum={args.accum_steps} "
          f"effective_batch={args.batch_size * args.accum_steps}")

    def maybe_plot():
        if not args.no_plot:
            try:
                make_curves_pdf(args.out_dir)
            except Exception as e:
                print(f"[plot] skipped ({e}); metadata saved, use plot_curves.py")

    model.train()
    gstep = 0
    best_loss = float("inf")
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

                accum = max(1, args.accum_steps)
                is_accum_step = ((i + 1) % accum == 0) or (i + 1 == steps_per_epoch)

                with torch.autocast(device_type=device.type,
                                    enabled=args.amp and device.type == "cuda"):
                    loss, _ = model(clip)
                # scale so summed grads == mean grads of an effective batch
                scaler.scale(loss / accum).backward()

                if is_accum_step:
                    if args.clip_grad:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    # EMA update on optimizer steps only (once per effective batch)
                    m = ema_momentum(gstep, total_steps, args.ema_start, args.ema_end)
                    model.update_target(m)
                    gstep += 1
                else:
                    m = ema_momentum(gstep, total_steps, args.ema_start, args.ema_end)

                loss_val = loss.item()
                running += loss_val
                logger.log_step(gstep, epoch, i + 1, loss_val, lr)
                if is_accum_step and gstep % args.log_every == 0:
                    print(f"  epoch {epoch} step {i+1}/{steps_per_epoch} "
                          f"loss {loss_val:.4f} lr {lr:.2e} ema {m:.4f}")

            avg = running / min(steps_per_epoch, len(loader))
            dt = time.time() - t0
            logger.log_epoch(epoch, avg, dt)
            print(f"[epoch {epoch}] avg_loss {avg:.4f} time {dt:.1f}s")

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                ckpt = os.path.join(args.out_dir, f"vjepa2_{args.model}_ep{epoch+1}.pth")
                torch.save({"model": model.state_dict(),
                            "encoder": model.encoder.state_dict(),
                            "epoch": epoch, "args": vars(args)}, ckpt)
                print(f"[ckpt] saved {ckpt}")
                maybe_plot()

            if avg < best_loss - args.early_stop_min_delta:
                best_loss = avg
                epochs_no_improve = 0
                best_path = os.path.join(args.out_dir, f"vjepa2_{args.model}_best.pth")
                torch.save({"model": model.state_dict(),
                            "encoder": model.encoder.state_dict(),
                            "epoch": epoch, "avg_loss": avg, "args": vars(args)}, best_path)
            else:
                epochs_no_improve += 1

            if args.early_stop_patience and epochs_no_improve >= args.early_stop_patience:
                print(f"[early-stop] no improvement > {args.early_stop_min_delta} for "
                      f"{args.early_stop_patience} epochs (best {best_loss:.4f}). "
                      f"Stopping at epoch {epoch}.")
                stop_ckpt = os.path.join(args.out_dir, f"vjepa2_{args.model}_ep{epoch+1}.pth")
                torch.save({"model": model.state_dict(),
                            "encoder": model.encoder.state_dict(),
                            "epoch": epoch, "args": vars(args)}, stop_ckpt)
                print(f"[ckpt] saved {stop_ckpt}")
                break
    finally:
        logger.close()
        maybe_plot()

    print("done.")


if __name__ == "__main__":
    main()
