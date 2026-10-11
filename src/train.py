"""
train.py - train one model with one seed and save checkpoint + predictions.

    python src/train.py --model tebif --seed 0
    python src/train.py --model all            (runs every model x seed 0,1,2, 3 processes at a time)

Same data, optimiser (Adam), batch size, class weight (pos_weight = n_neg/n_pos from TRAIN) and
early stopping (validation PR-AUC) for every model.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(__file__))
from data import EventData
from models import make_model

RUNS = "results/runs"


@torch.no_grad()
def predict(model, data, idx, bs=4096):
    """Probability p_t for every event index in idx."""
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        logit, _ = model(data.batch(idx[i:i + bs]))
        out.append(torch.sigmoid(logit).numpy())
    return np.concatenate(out)


def train(model_name, seed, feats="safe", max_epochs=20, patience=4, bs=512, lr=1e-3, threads=1):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = EventData(feats)
    model = make_model(model_name, data.n_cont)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr, va = data.idx(0), data.idx(1)
    rng = np.random.default_rng(seed)
    # Class imbalance (~1% positives): every epoch uses ALL train positives + a fresh random
    # NEG_RATIO x as many negatives, and positives get class weight = NEG_RATIO (n_neg/n_pos of
    # the epoch sample). Statistically this equals weighting by the full-train n_neg/n_pos (~100)
    # but is ~10x cheaper on CPU. Same recipe for every model.
    NEG_RATIO = 10
    pos_i, neg_i = tr[data.label[tr] == 1], tr[data.label[tr] == 0]
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(NEG_RATIO)))
    best, best_state, bad, hist, t0 = -1, None, 0, [], time.time()
    for ep in range(max_epochs):
        model.train()
        perm = rng.permutation(np.r_[pos_i, rng.choice(neg_i, NEG_RATIO * len(pos_i), replace=False)])
        tot = 0
        for i in range(0, len(perm), bs):
            ii = perm[i:i + bs]
            logit, _ = model(data.batch(ii))
            loss = loss_fn(logit, torch.from_numpy(data.label[ii]).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(ii)
        pv = predict(model, data, va)
        ap = average_precision_score(data.label[va], pv)
        hist.append((ep, tot / len(perm), ap))
        print(f"[{model_name}/{feats}/s{seed}] epoch {ep} loss {tot/len(perm):.4f} val PR-AUC {ap:.4f}",
              flush=True)
        if ap > best:
            best, bad = ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:                                # early stopping on validation PR-AUC
                break
    model.load_state_dict(best_state)
    # predictions for EVERY val/test event (labelled or not) - needed for risk scores later
    ev = np.where(data.split >= 1)[0]
    p = predict(model, data, ev)
    name = f"{model_name}_{feats}_s{seed}"
    os.makedirs(RUNS, exist_ok=True)
    torch.save(best_state, f"{RUNS}/{name}.pt")
    np.savez(f"{RUNS}/{name}.npz", idx=ev, p=p)
    json.dump(dict(model=model_name, feats=feats, seed=seed, best_val_prauc=best, history=hist,
                   epochs=len(hist), train_seconds=time.time() - t0,
                   params=sum(x.numel() for x in model.parameters())),
              open(f"{RUNS}/{name}.json", "w"))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tebif")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feats", default="safe")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--redo", action="store_true", help="re-run even if results exist")
    a = ap.parse_args()
    if a.model != "all":
        train(a.model, a.seed, a.feats, a.epochs)
    else:
        # main comparison (3 seeds) + ablations + feature-set (leakage) checks (seed 0 only)
        jobs = [(m, s, "safe") for s in (0, 1, 2)
                for m in ("lstm", "vanilla", "tebif", "tebif_dt", "tebif_ctime")]
        jobs += [("tebif", 0, "nobal"), ("tebif", 0, "leaky"), ("vanilla", 0, "leaky")]
        jobs = [j for j in jobs if a.redo or not os.path.exists(f"{RUNS}/{j[0]}_{j[2]}_s{j[1]}.json")]
        procs = []
        for m, s, f in jobs:
            while len([p for p in procs if p.poll() is None]) >= 4:
                time.sleep(2)
            procs.append(subprocess.Popen([sys.executable, __file__, "--model", m, "--seed", str(s),
                                           "--feats", f, "--epochs", str(a.epochs)]))
        for p in procs:
            p.wait()
