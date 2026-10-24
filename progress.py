"""Live progress table for a running suite (reads runs/<tag>_*.stdout.log; touches nothing).

    python progress.py syn1m                 # one snapshot
    watch -n 30 python progress.py syn1m     # auto-refresh
"""
from __future__ import annotations

import glob
import os
import re
import sys

PAT = re.compile(r"\[(?:ppo|random)\] step (\d+) (?:ep=\d+ )?obj=([-\d.eE+]+) valid_ic=([-\d.eE+na]+)(?:.*?t=([\d.]+)s)?")


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    rows = []
    for p in sorted(glob.glob(os.path.join("runs", f"{tag}_*.stdout.log"))):
        name = os.path.basename(p)[: -len(".stdout.log")]
        txt = open(p, errors="ignore").read()
        m = None
        for m in PAT.finditer(txt):
            pass
        done = os.path.exists(os.path.join("runs", name, "pool_best_valid.json")) and "done" in txt[-200:]
        if "Traceback" in txt:
            rows.append((name, "FAILED", "", "", "", ""))
        elif m:
            step = int(m.group(1))
            t = float(m.group(4)) if m.group(4) else None
            eta = f"~{(total - step) / max(step / t, 1e-9) / 60:.0f}m left" if t else ""
            rows.append((name, f"{step:>8}/{total} ({100*step/total:4.1f}%)", m.group(2)[:7], m.group(3)[:7],
                         f"{t/60:.0f}m" if t else "", "done" if done else eta))
        else:
            rows.append((name, "starting", "", "", "", ""))
    print(f"{'run':32} {'progress':26} {'train':>7} {'valid':>7} {'elapsed':>8}  eta")
    for r in rows:
        print(f"{r[0]:32} {r[1]:26} {r[2]:>7} {r[3]:>7} {r[4]:>8}  {r[5]}")
    if not rows:
        print("no logs found")


if __name__ == "__main__":
    main()
