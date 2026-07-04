

import os
import sys
import time
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.datasets import CataractVideoDataset


def main():
    p = argparse.ArgumentParser("cataract data check")
    p.add_argument("--data-root", required=True)
    p.add_argument("--data-mode", choices=["video", "frames"], default="video")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--sampling-stride", type=int, default=4)
    p.add_argument("--num-check", type=int, default=8,
                   help="how many clips to actually load-test")
    args = p.parse_args()

    print(f"[check] scanning {args.data_root} (mode={args.data_mode}) ...")
    ds = CataractVideoDataset(
        args.data_root, mode=args.data_mode,
        num_frames=args.num_frames, img_size=args.img_size,
        sampling_stride=args.sampling_stride,
    )
    n = len(ds)
    print(f"[check] found {n} clips.")
    if n == 0:
        print("[FAIL] no videos found — check the path and file extensions.")
        return

    # try loading a sample of clips
    k = min(args.num_check, n)
    print(f"[check] load-testing {k} clips ...")
    ok, failed, times = 0, [], []
    for i in range(k):
        try:
            t0 = time.time()
            clip = ds[i]
            dt = time.time() - t0
            times.append(dt)
            if i == 0:
                print(f"        clip shape={tuple(clip.shape)} dtype={clip.dtype} "
                      f"min={clip.min():.2f} max={clip.max():.2f}")
            ok += 1
        except Exception as e:
            failed.append((ds.items[i], repr(e)))

    print(f"[check] {ok}/{k} clips loaded OK.")
    if times:
        avg = sum(times) / len(times)
        print(f"[check] avg decode time {avg*1000:.0f} ms/clip "
              f"({'fast enough' if avg < 0.3 else 'consider pre-extracting frames'})")
    if failed:
        print(f"[WARN] {len(failed)} clip(s) failed to load:")
        for path, err in failed[:10]:
            print(f"        {os.path.basename(path)} -> {err}")
    else:
        print("[check] all sampled clips decoded cleanly. Ready to pretrain.")


if __name__ == "__main__":
    main()