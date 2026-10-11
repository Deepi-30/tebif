"""
data.py - PaySim -> per-account event streams -> 30-event windows with 24-step labels.

Key ideas (see README for the reasoning):
  * An "event" is one transaction seen from ONE account's point of view. Every PaySim row
    gives two events: one for the sender (role 0) and one for the receiver (role 1).
  * An account's stream = all its events ordered by step. Streams are cut at the
    train/val/test boundaries, so no window or label ever crosses a split.
  * Label at event t = 1 if a LATER event of the same account is a confirmed fraud and
    happens at most 24 steps after t. Features only use events up to t.
  * Only the compact event table is cached; windows are gathered on the fly (cheap).
"""
import glob
import os
import numpy as np
import pandas as pd
import torch

TYPES = ["CASH_IN", "CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT"]
# last step of train / val / test. Cut at 60% / 80% of all PaySim TRANSACTIONS (not of time):
# normal traffic is bursty and 95% of it happens before step ~445, while fraud is spread evenly.
SPLIT_END = np.array([281, 355, 743])
SPLIT_NAMES = ["train", "val", "test"]
HORIZON = 24                               # label horizon in steps (1 step = 1 simulated hour)
WINDOW = 30                                # events in a window
UNK = 10                                   # token id for padding / "unknown" (type*2+role in 0..9)
CACHE = "cache/events.npz"

# columns of the stored continuous matrix
CONT_NAMES = ["log_amount", "amt_over_old", "rel_change", "log_old_bal", "log_new_bal", "drained"]
# feature sets: 'safe' = amount + balance-change ratios (default); 'nobal' = amount only;
# 'leaky' = also raw balances + the "account drained to zero" flag (used only for the leakage test)
FEATURE_SETS = {"safe": [0, 1, 2], "nobal": [0], "leaky": [0, 1, 2, 3, 4, 5]}


def build_cache(csv_path=None, target_events=420_000, seed=0):
    """Read the full CSV, build per-account streams, keep a subset, label it, save the cache."""
    csv_path = csv_path or glob.glob("data/*.csv")[0]
    df = pd.read_csv(csv_path)
    n = len(df)

    # --- long table: row L < n is the sender event of tx L, row L >= n is the receiver event
    names = pd.concat([df.nameOrig, df.nameDest], ignore_index=True)
    codes, uniques = pd.factorize(names)
    codes = codes.astype(np.int64)
    is_merchant = np.array([u[0] == "M" for u in uniques])
    step_all = np.tile(df.step.to_numpy(np.int32), 2)
    fraud_all = np.tile(df.isFraud.to_numpy(np.int8), 2)
    split_all = np.searchsorted(SPLIT_END, step_all)          # 0 train, 1 val, 2 test

    # --- stream = (account, split). Keep streams with >= 2 events, non-merchant accounts
    key = codes * 3 + split_all
    counts = np.bincount(key)
    group_ok = (counts[key] >= 2) & ~is_merchant[codes]

    # --- subset: ALL streams of accounts that touch a fraud, plus a random sample of normal
    #     streams. The sample is drawn per split (60/20/20 of the budget) so val/test are usable.
    fraud_acc = np.zeros(len(uniques), bool)
    fraud_acc[codes[fraud_all == 1]] = True
    fraud_acc &= ~is_merchant
    take_ev = group_ok & fraud_acc[codes]
    rng = np.random.default_rng(seed)
    normal_ev = np.where(group_ok & ~fraud_acc[codes])[0]
    ug, inv = np.unique(key[normal_ev], return_inverse=True)       # one entry per normal stream
    keep_g = np.zeros(len(ug), bool)
    for s_, share in enumerate([0.6, 0.2, 0.2]):
        budget = share * target_events - np.sum(take_ev & (split_all == s_))
        g = rng.permutation(np.where(ug % 3 == s_)[0])
        keep_g[g[np.cumsum(counts[ug[g]]) <= budget]] = True
    take_ev[normal_ev[keep_g[inv]]] = True
    sel = np.where(take_ev)[0]

    # --- sort events by account, then step, then transaction id
    tid = sel % n
    role = (sel // n).astype(np.int8)
    order = np.lexsort((tid, step_all[sel], codes[sel]))
    sel, tid, role = sel[order], tid[order], role[order]
    acc = codes[sel]
    step = step_all[sel]
    split = split_all[sel].astype(np.int8)
    fraud = fraud_all[sel]

    # --- event features (raw; normalised later using train rows only)
    typ = df.type.map({t: i for i, t in enumerate(TYPES)}).to_numpy()[tid]
    tok = (typ * 2 + role).astype(np.int8)
    amount = df.amount.to_numpy()[tid]
    old = np.where(role == 0, df.oldbalanceOrg.to_numpy()[tid], df.oldbalanceDest.to_numpy()[tid])
    new = np.where(role == 0, df.newbalanceOrig.to_numpy()[tid], df.newbalanceDest.to_numpy()[tid])
    cont = np.stack([
        np.log1p(amount),
        np.minimum(amount / (old + 1), 5),                   # how much of the balance this tx is
        np.clip((new - old) / (old + 1), -5, 5),             # relative balance change
        np.log1p(old), np.log1p(new),
        ((new == 0) & (old > 0)).astype(float),              # leaky: "account emptied"
    ], 1).astype(np.float32)

    # --- stream start index for every event
    gkey = acc * 3 + split
    newgrp = np.r_[True, gkey[1:] != gkey[:-1]]
    start = np.maximum.accumulate(np.where(newgrp, np.arange(len(gkey)), 0)).astype(np.int32)

    # --- label: a LATER event in the same stream is fraud and within HORIZON steps
    nxt = np.full(len(acc), np.inf)
    cur = np.inf
    for i in range(len(acc) - 1, -1, -1):
        if i == len(acc) - 1 or start[i] != start[i + 1]:
            cur = np.inf
        nxt[i] = cur
        if fraud[i]:
            cur = step[i]                       # steps are sorted, so this is the nearest fraud
    label = ((nxt - step) <= HORIZON).astype(np.int8)
    # a label is only trusted if its whole horizon lies inside the same split
    labelled = (step + HORIZON <= SPLIT_END[split])

    os.makedirs("cache", exist_ok=True)
    np.savez_compressed(CACHE, tok=tok, cont=cont, step=step, start=start, acc=acc, tid=tid,
                        fraud=fraud, label=label, labelled=labelled, split=split,
                        acc_name=np.array(uniques)[acc].astype("U16"))
    print("cache built:", len(acc), "events;", len(np.unique(tid)), "unique PaySim rows;",
          len(np.unique(acc)), "accounts")


class EventData:
    """Loads the cache, normalises with TRAIN statistics only, and gathers windows."""

    def __init__(self, feats="safe", cache=CACHE):
        if not os.path.exists(cache):
            build_cache()
        z = np.load(cache)
        for k in z.files:
            setattr(self, k, z[k])
        self.N = len(self.tok)
        self.cols = FEATURE_SETS[feats]
        tr = self.split == 0
        self.mu = self.cont[tr].mean(0)
        self.sd = self.cont[tr].std(0) + 1e-6
        self.cont = ((self.cont - self.mu) / self.sd).astype(np.float32)   # train-fitted z-score
        self.n_cont = len(self.cols)

    # indices of prediction points
    def idx(self, split, labelled_only=True):
        m = self.split == split
        if labelled_only:
            m &= self.labelled
        return np.where(m)[0]

    def pos_weight(self):
        """Class weight fitted on TRAIN only: n_negative / n_positive."""
        i = self.idx(0)
        p = self.label[i].sum()
        return float((len(i) - p) / max(p, 1))

    def batch(self, idx):
        """Gather the 30-event window ending at every index in idx (left-padded, most recent last)."""
        idx = np.asarray(idx)
        gi = idx[:, None] - np.arange(WINDOW - 1, -1, -1)[None, :]
        mask = gi >= self.start[idx][:, None]            # never reach before the stream start
        gi = np.where(mask, gi, 0)
        tok = np.where(mask, self.tok[gi], UNK)
        cont = self.cont[gi][:, :, self.cols] * mask[..., None]
        return dict(tok=torch.from_numpy(tok).long(),
                    cont=torch.from_numpy(cont.astype(np.float32)),
                    step=torch.from_numpy(self.step[gi]).float(),
                    mask=torch.from_numpy(mask))


if __name__ == "__main__":
    build_cache()
    d = EventData()
    L = np.bincount(d.start)[np.unique(d.start)]
    print("stream length: mean %.2f median %.1f max %d" % (L.mean(), np.median(L), L.max()))
    for s in range(3):
        i = d.idx(s)
        print(SPLIT_NAMES[s], "events", (d.split == s).sum(), "| labelled points", len(i),
              "| positives", d.label[i].sum(), "(%.2f%%)" % (100 * d.label[i].mean()))
    print("pos_weight (train):", d.pos_weight())
