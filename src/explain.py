"""
explain.py - Step 5: explanations for the TEBIF model.

Three views of "why was this account flagged?":
  1. ATTENTION   what the model attended to (last layer, head-average, from the current event to
                 every event of the window). A description of the model's focus, NOT a cause.
  2. OCCLUSION   SHAP-style, post-hoc: remove one event (or blank one feature of one event) and
                 measure how much the predicted probability drops. (We use occlusion instead of the
                 shap library because it is exact for 30-event windows and cheap on CPU.)
  3. FAITHFULNESS  delete the top-k events in attribution order vs. random events and compare the
                 probability drop ("comprehensiveness"), plus insertion/deletion curves.

Every deletion keeps the CURRENT event (the last one) - the question is which PAST events matter.
Deleting an event re-derives the time gaps from the remaining events (see models.time_features).

    python src/explain.py        -> results/faithfulness.json, results/fig_faithfulness.png
"""
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data import EventData, TYPES, CONT_NAMES, UNK, WINDOW
from models import make_model
from evaluate import (load_P, risk_scores, split_events, stream_table, alert_threshold, RUNS, ALPHA)

FEATURE_LABEL = {"log_amount": "amount", "amt_over_old": "amount ÷ balance",
                 "rel_change": "balance change", "type": "type/role", "timing": "time gap"}
RNG = np.random.default_rng(0)


def load_tebif(data, seed=0):
    m = make_model("tebif", data.n_cont)
    m.load_state_dict(torch.load(f"{RUNS}/tebif_safe_s{seed}.pt"))
    return m.eval()


@torch.no_grad()
def prob(m, b):
    return torch.sigmoid(m(b)[0]).numpy()


def variants(b1, n):
    """n copies of a single-window batch (to edit one copy per perturbation)."""
    return {k: v.repeat(n, *([1] * (v.dim() - 1))).clone() for k, v in b1.items()}


# ----------------------------------------------------------------------------- 1. attention
@torch.no_grad()
def attention(m, b):
    """(B, L) attention of the last event onto every window position (rows sum to 1)."""
    _, a = m(b, return_attn=True)
    return (a * b["mask"]).numpy()


# ----------------------------------------------------------------------------- 2. occlusion
def event_drops(m, b1):
    """drop[j] = p(full window) - p(window without event j) for every PAST event j."""
    mask = b1["mask"][0]
    pos = [j for j in range(mask.shape[0] - 1) if mask[j]]
    p0 = prob(m, b1)[0]
    drop = np.zeros(mask.shape[0])
    if pos:
        v = variants(b1, len(pos))
        for r, j in enumerate(pos):
            v["mask"][r, j] = False
        drop[pos] = p0 - prob(m, v)
    return drop, p0


def cell_drops(m, data, b1):
    """Blank ONE feature of ONE event and record the drop. Returns [(position, feature, drop)]."""
    mask = b1["mask"][0]
    p0 = prob(m, b1)[0]
    cells, edits = [], []
    prev = None
    for j in range(mask.shape[0]):
        if not mask[j]:
            continue
        edits.append((j, "type"))
        for ci, c in enumerate(data.cols):
            edits.append((j, CONT_NAMES[c]))
        if prev is not None:
            edits.append((j, "timing"))
        prev = j
    v = variants(b1, len(edits))
    for r, (j, f) in enumerate(edits):
        if f == "type":
            v["tok"][r, j] = UNK
        elif f == "timing":                                   # collapse the gap before this event
            pj = max(q for q in range(j) if mask[q])
            v["step"][r, j] = v["step"][r, pj]
        else:
            v["cont"][r, j, [CONT_NAMES[c] for c in data.cols].index(f)] = 0.0
    d = p0 - prob(m, v)
    return [(j, f, float(x)) for (j, f), x in zip(edits, d)], p0


def group_drops(m, data, b1):
    """Blank a feature in EVERY event of the window at once -> importance per feature type."""
    p0 = prob(m, b1)[0]
    names = ["type"] + [CONT_NAMES[c] for c in data.cols] + ["timing"]
    v = variants(b1, len(names))
    for r, f in enumerate(names):
        if f == "type":
            v["tok"][r][v["mask"][r]] = UNK
        elif f == "timing":
            v["step"][r] = v["step"][r][-1]                   # all gaps -> 0
        else:
            v["cont"][r, :, [CONT_NAMES[c] for c in data.cols].index(f)] = 0.0
    d = p0 - prob(m, v)
    return {n: float(x) for n, x in zip(names, d)}


# ----------------------------------------------------------------------------- 3. faithfulness
def p_after_deleting(m, b1, sets):
    """Probability after deleting each set of positions (list of lists)."""
    v = variants(b1, len(sets))
    for r, s in enumerate(sets):
        v["mask"][r, list(s)] = False
    return prob(m, v)


def p_with_only(m, b1, sets):
    """Probability when ONLY the current event + the listed past positions are kept (insertion)."""
    v = variants(b1, len(sets))
    keep = torch.zeros_like(v["mask"])
    keep[:, -1] = True
    for r, s in enumerate(sets):
        keep[r, list(s)] = True
    v["mask"] &= keep
    return prob(m, v)


def faithfulness_window(m, b1, drop, attn, ks=(1, 3, 5), n_rand=20):
    """Comprehensiveness (p_full - p_after_deleting_top_k) for occlusion order, attention order
    and random order; plus deletion/insertion curves over all past events."""
    mask = b1["mask"][0]
    past = [j for j in range(mask.shape[0] - 1) if mask[j]]
    n = len(past)
    p0 = prob(m, b1)[0]
    orders = {"occlusion": sorted(past, key=lambda j: -drop[j]),
              "attention": sorted(past, key=lambda j: -attn[j])}
    res = {"p_full": float(p0), "n_past": n, "comp": {}, "del_curve": {}, "ins_curve": {}}
    for k in ks:
        kk = min(k, n)
        row = {}
        for name, o in orders.items():
            row[name] = float(p0 - p_after_deleting(m, b1, [o[:kk]])[0])
        rnd = [list(RNG.choice(past, kk, replace=False)) for _ in range(n_rand)]
        row["random"] = float(p0 - p_after_deleting(m, b1, rnd).mean())
        res["comp"][str(k)] = row
    rnd_o = [list(RNG.permutation(past)) for _ in range(n_rand)]
    for name, o in list(orders.items()) + [("random", None)]:
        if o is not None:
            res["del_curve"][name] = [float(x) for x in
                                      np.r_[p0, p_after_deleting(m, b1, [o[:i] for i in range(1, n + 1)])]]
            res["ins_curve"][name] = [float(x) for x in
                                      np.r_[p_with_only(m, b1, [[]])[0],
                                            p_with_only(m, b1, [o[:i] for i in range(1, n + 1)])]]
        else:   # average over random orders
            d = np.mean([np.r_[p0, p_after_deleting(m, b1, [o_[:i] for i in range(1, n + 1)])]
                         for o_ in rnd_o], 0)
            ins = np.mean([np.r_[p_with_only(m, b1, [[]])[0],
                                 p_with_only(m, b1, [o_[:i] for i in range(1, n + 1)])] for o_ in rnd_o], 0)
            res["del_curve"][name], res["ins_curve"][name] = d.tolist(), ins.tolist()
    return res


# ----------------------------------------------------------------------------- plain language
def describe_event(data, g, amt_q):
    t = TYPES[data.tok[g] // 2]
    role = "received" if data.tok[g] % 2 else "sent"
    a = np.expm1(data.cont[g, 0] * data.sd[0] + data.mu[0])          # un-normalise log amount
    size = "large " if a >= amt_q[0] else ("small " if a <= amt_q[1] else "")
    return f"{size}{t} ({role}, {a:,.0f})", a


def gap_text(dt):
    return "in the same hour" if dt == 0 else (f"{int(dt)} hour after" if dt == 1 else f"{int(dt)} hours after")


def summarise(data, focus, window_pos_to_global, drop, p, R, alerted, groups):
    """Short plain-language sentence(s) for an account at its focus event."""
    amt_q = np.expm1(np.percentile(data.cont[data.split == 0][:, 0], [90, 25]) * data.sd[0] + data.mu[0])
    pos = [q for q in window_pos_to_global if q is not None]
    k = pos.index(focus) + 1
    cur, _ = describe_event(data, focus, amt_q)
    s = f"{'Alert' if alerted else 'Highest-risk point'} at event {k} (step {data.step[focus]}): {cur}"
    if k > 1:
        prev = pos[k - 2]
        pd_, _ = describe_event(data, prev, amt_q)
        s += f", {gap_text(data.step[focus] - data.step[prev])} a {pd_}"
    s += f". Risk p={p:.2f}, R={R:.2f}."
    recent = [q for q in pos if data.step[focus] - data.step[q] <= 3]
    if len(recent) >= 3:
        s += f" Burst: {len(recent)} events within 3 hours."
    j = int(np.argmax(drop))
    if drop[j] > 0.005 and window_pos_to_global[j] is not None:
        gj = window_pos_to_global[j]
        ev, _ = describe_event(data, gj, amt_q)
        s += f" Removing event {pos.index(gj) + 1} ({ev}) lowers the risk most (-{drop[j]:.2f})."
    top = max(groups, key=lambda f: groups[f])
    if groups[top] > 0.005:
        s += f" Most influential feature: {FEATURE_LABEL.get(top, top)}."
    return s


def explain_focus(m, data, focus, p, R, alerted):
    """Everything the UI needs for one account at one focus event."""
    b1 = data.batch([focus])
    mask = b1["mask"][0].numpy()
    g_of = [(focus - (WINDOW - 1 - j)) if mask[j] else None for j in range(WINDOW)]
    attn = attention(m, b1)[0]
    drop, _ = event_drops(m, b1)
    cells, _ = cell_drops(m, data, b1)
    groups = group_drops(m, data, b1)
    fw = faithfulness_window(m, b1, drop, attn) if mask.sum() > 1 else None
    top = sorted(cells, key=lambda c: -abs(c[2]))[:6]
    return dict(
        window_global=g_of, attention=attn.tolist(), event_drop=drop.tolist(),
        top_features=[dict(event=g_of[j], feature=FEATURE_LABEL.get(f, f), drop=d) for j, f, d in top],
        feature_groups={FEATURE_LABEL.get(k, k): v for k, v in groups.items()},
        faithfulness=fw["comp"] if fw else None,
        summary=summarise(data, focus, g_of, drop, p, R, alerted, groups))


# ----------------------------------------------------------------------------- aggregate test
def main():
    data = EventData("safe")
    m = load_tebif(data)
    P = load_P("tebif", 0, "safe", data.N)
    ev_v, ev_t = split_events(data, 1), split_events(data, 2)
    thr, _ = alert_threshold(risk_scores(P, data, ev_v), data, ev_v)
    Rt = risk_scores(P, data, ev_t)
    # focus windows = the first alert event of every alerted test stream (>= 6 events so k=5 works)
    foc = []
    for a, b, f in stream_table(data, ev_t):
        end = b if f < 0 else a + max(f, 0)
        hit = np.where(Rt[a:end] >= thr)[0] if end > a else []
        if len(hit) and a + hit[0] - a >= 5:
            foc.append(ev_t[a + hit[0]])
    foc = np.array(foc, dtype=np.int64)
    if len(foc) > 300:
        foc = RNG.choice(foc, 300, replace=False)
    pool = "alert events"
    if len(foc) < 60:                                         # too few alerts: add top-scored events
        cand = ev_t[(ev_t - data.start[ev_t] >= 5)]
        extra = cand[np.argsort(-P[cand])[: 200 - len(foc)]]
        foc, pool = np.unique(np.r_[foc, extra]), "alert events + top-scored test events"
    print(f"faithfulness pool: {len(foc)} windows ({pool}), alert threshold {thr:.4f}")
    allres = []
    for i, g in enumerate(foc):
        b1 = data.batch([g])
        attn = attention(m, b1)[0]
        drop, _ = event_drops(m, b1)
        allres.append(faithfulness_window(m, b1, drop, attn))
    out = dict(pool=pool, n_windows=len(allres), comp={}, del_curve={}, ins_curve={})
    for k in ("1", "3", "5"):
        out["comp"][k] = {name: dict(mean=float(np.mean([r["comp"][k][name] for r in allres])),
                                     sem=float(np.std([r["comp"][k][name] for r in allres]) / np.sqrt(len(allres))))
                          for name in ("occlusion", "attention", "random")}
    grid = np.linspace(0, 1, 11)
    for kind in ("del_curve", "ins_curve"):
        for name in ("occlusion", "attention", "random"):
            ys = [np.interp(grid, np.linspace(0, 1, len(r[kind][name])), r[kind][name]) for r in allres]
            out[kind][name] = np.mean(ys, 0).tolist()
    out["grid"] = grid.tolist()
    json.dump(out, open("results/faithfulness.json", "w"), indent=1)
    print("comprehensiveness (mean drop in p):")
    for k, v in out["comp"].items():
        print(" k=%s " % k, {n: round(x["mean"], 4) for n, x in v.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, col in (("occlusion", "#d62728"), ("attention", "#1f77b4"), ("random", "#7f7f7f")):
        ax[0].plot(grid, out["del_curve"][name], color=col, label=name)
        ax[1].plot(grid, out["ins_curve"][name], color=col, label=name)
    ax[0].set_title("Deletion curve (lower = better ranking)")
    ax[1].set_title("Insertion curve (higher = better ranking)")
    for a in ax:
        a.set_xlabel("fraction of past events removed / kept")
        a.set_ylabel("mean predicted probability")
        a.legend()
    plt.tight_layout()
    plt.savefig("results/fig_faithfulness.png", dpi=130)


if __name__ == "__main__":
    main()
