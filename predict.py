#!/usr/bin/env python3
"""Run one checkpoint on corpus episodes and print its decisions.

    python3 predict.py
    python3 predict.py --checkpoint checkpoints/futian/fine_tuned/B1024/film/best.pt --corpus demo_data/futian --episodes 0 1 --windows 0 5 11
    python3 predict.py --budget 2048 --increment 1

Without arguments, the General B=1024 FiLM model runs on bundled demo data for episodes 11 and 53 and windows 0, 5, and 11. For ``--checkpoint`` and ``--corpus``, a relative path already present in the current working directory is used there; otherwise it is resolved relative to this file. ``--out`` is passed unchanged to ``numpy.savez_compressed`` and is therefore resolved by the current working directory. The historical ``futian`` paths denote the Leipzig demo. ``--corpus`` must contain observations.h5, such as demo data or a corpus rebuilt with ``release_corpus.py derive``. CUDA is selected when available, otherwise CPU. If targets.h5 and budget_labels.h5 are also present, the script scores each completed decision against stored future data and prints its BLER, target comparison, feasibility, and label. Future data are used only for this scoring and do not change the decision.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from osbs_infer import (EPS_TARGET, INCREMENT_MAX, REPETITION_BUDGETS, decide, load_budget_labels, load_checkpoint,
                        load_future, load_observations, served_bler_curve)

HERE = Path(__file__).resolve().parent          # Fallback for relative paths absent from the current working directory.


def here(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() or p.exists() else HERE / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/general/B1024/film/best.pt", help="best.pt of one variant and budget")
    ap.add_argument("--corpus", default="demo_data/general", help="directory with observations.h5")
    ap.add_argument("--episodes", type=int, nargs="+", default=[11, 53], help="episode indices in that file")
    ap.add_argument("--windows", type=int, nargs="+", default=[0, 5, 11], help="window indices 0..11")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--budget", type=int, choices=REPETITION_BUDGETS, default=None,
                    help="actual per-window budget; defaults to the checkpoint training budget")
    ap.add_argument("--increment", type=int, choices=range(INCREMENT_MAX + 1), default=0,
                    help="K increment selected on validation data for this exact checkpoint and setting (default: 0)")
    ap.add_argument("--out", default=None,
                    help="optional .npz with serve, base_K, final K, level, pairs, actual budget and increment")
    args = ap.parse_args()

    corpus = here(args.corpus)
    net, ck = load_checkpoint(here(args.checkpoint), args.device)
    checkpoint_budget = int(ck["budget_provenance"]["budget"])
    budget = checkpoint_budget if args.budget is None else int(args.budget)
    obs = load_observations(corpus, args.episodes)
    pairs = np.asarray([(i, w) for i in range(len(args.episodes)) for w in args.windows], dtype=np.int64)
    serve, K, level, base_K = decide(net, ck, obs, pairs[:, 0], pairs[:, 1], args.device, budget=budget,
                                     increment=args.increment, return_base_k=True)
    episodes = np.asarray(args.episodes)[pairs[:, 0]]
    scored = (corpus / "targets.h5").exists() and (corpus / "budget_labels.h5").exists()
    if scored:
        horizon, _ = load_future(corpus, episodes, pairs[:, 1])
        curve = served_bler_curve(horizon, serve, obs["mu_p"][pairs[:, 0]])                # [P, 8]
        bler = np.where(K > 0, curve[np.arange(len(K)), np.maximum(K, 1) - 1], np.nan)
        label = load_budget_labels(corpus, episodes, pairs[:, 1], budget)
    print(f"checkpoint {args.checkpoint}: variant {ck['variant']}, checkpoint budget {checkpoint_budget}, "
          f"actual budget {budget}, increment {args.increment}, rule offset {float(ck['rule_offset_db']):.2f} dB")
    if args.increment:
        print("increment provenance: caller supplied; it must be selected on validation data for this exact checkpoint "
              "and setting. Released checkpoint copies are not asserted byte-identical to the paper-run checkpoints, "
              "so this command alone does not reproduce paper metrics.")
    head = f"{'episode':>8} {'window':>6} {'mu_p':>4} {'served':>6} {'base K':>6} {'K':>2} {'median dB':>10} {'q0.2 dB':>8}"
    if scored:
        head += f" {'BLER at K':>10} {'target':>8} {'feasible':>8} {'label served':>12} {'label K':>8} {'label usage':>11}"
    print(head)
    for n, ((i, w), s, base_k, k, lv) in enumerate(zip(pairs, serve, base_K, K, level)):
        line = (f"{args.episodes[i]:>8} {w:>6} {int(obs['mu_p'][i]):>4} {int(s.sum()):>6} "
                f"{int(base_k):>6} {int(k):>2} {lv[0]:>10.2f} {lv[1]:>8.2f}")
        if scored:
            b = "skipped" if k == 0 else f"{bler[n]:.4f}"
            met = "-" if k == 0 else ("met" if bler[n] <= EPS_TARGET else "violated")
            if label["feasible"][n]:
                kb = label["K_bin"][n]
                served_n, ks, usage = str(int((kb > 0).sum())), ",".join(str(int(k)) for k in np.unique(kb[kb > 0])), str(int(label["usage"][n]))
            else:
                served_n, ks, usage = "-", "-", "-"
            line += f" {b:>10} {met:>8} {str(bool(label['feasible'][n])):>8} {served_n:>12} {ks:>8} {usage:>11}"
        print(line)
    if scored:
        print("feasible: the label fits the budget; label served: bins the label serves; label K: the repetition factors the "
              "label uses, one per served class of the future repetition map; label usage: its bin transmissions")
    if args.out:
        np.savez_compressed(args.out, serve=serve, base_K=base_K, K=K, level=level,
                            pairs=np.stack((episodes, pairs[:, 1]), 1),
                            checkpoint_budget=np.asarray(checkpoint_budget, dtype=np.int64),
                            actual_budget=np.asarray(budget, dtype=np.int64),
                            increment=np.asarray(args.increment, dtype=np.int64))
        print("written", args.out)


if __name__ == "__main__":
    main()
