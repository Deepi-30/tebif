"""
evaluate.py - Step 3 (metrics, ablation, latency) and Step 4 (dynamic risk score + alerts).

    python src/evaluate.py

Reads results/runs/*.npz written by train.py, writes results/metrics.json, results/risk.json
and the PNG figures. Importable helpers (load_P, risk_scores, stream_table ...) are reused by
explain.py and export.py.
"""
import glob
import json
import os
import sys
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.insert(0, os.path.dirname(__file__))
from data import EventData
from models import make_model

RUNS = "results/runs"
SEEDS = (0,)          # short-budget demo: 1 seed per model (see README); use (0, 1, 2) with full runs
ALPHA = 0.5
LABELS = {"lstm": "LSTM", "vanilla": "Vanilla Transformer", "tebif": "TEBIF (Transformer + Δt + time)",
          "tebif_dt": "TEBIF, Δt encoding only", "tebif_ctime": "TEBIF, cumulative-time only"}


# ----------------------------------------------------------------------------- helpers
def load_P(model, seed, feats="safe", n=None):
    """Full-length array of p_t (NaN where the event is in the train split)."""
    z = np.load(f"{RUNS}/{model}_{feats}_s{seed}.npz")
    P = np.full(n, np.nan, np.float32)
    P[z["idx"]] = z["p"]
    return P


def best_f1_threshold(y, p):
    """Threshold maximising F1 (used with VALIDATION data only)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    k = np.argmax(f1[:-1])
    return float(th[k])


def metrics_for(P, data):
    """AUC-ROC, PR-AUC, F1 (val-tuned threshold), recall@85% precision, recall@1% alert rate."""
    iv, it = data.idx(1), data.idx(2)
    yv, pv, yt, pt = data.label[iv], P[iv], data.label[it], P[it]
    thr = best_f1_threshold(yv, pv)
    pred = pt >= thr
    tp = (pred & (yt == 1)).sum()
    f1 = 2 * tp / max(pred.sum() + yt.sum(), 1)
    pr, rc, _ = precision_recall_curve(yt, pt)
    r85 = rc[pr >= 0.85].max() if (pr >= 0.85).any() else 0.0
    top = np.argsort(-pt)[: max(1, len(pt) // 100)]          # flag the 1% highest-scored events
    return dict(auc=roc_auc_score(yt, pt), prauc=average_precision_score(yt, pt), f1=f1,
                recall_at_85p=float(r85), recall_at_1pct=float(yt[top].sum() / yt.sum()),
                val_prauc=average_precision_score(yv, pv), threshold=thr)


def latency_ms(model_name, seed, data, feats="safe", n=300):
    """Mean time to score ONE event (one 30-event window) on a single CPU thread."""
    torch.set_num_threads(1)
    m = make_model(model_name, data.n_cont)
    m.load_state_dict(torch.load(f"{RUNS}/{model_name}_{feats}_s{seed}.pt"))
    m.eval()
    idx = data.idx(2)[:n]
    bs = [data.batch(idx[i:i + 1]) for i in range(n)]
    with torch.no_grad():
        for b in bs[:20]:
            m(b)
        t0 = time.perf_counter()
        for b in bs:
            m(b)
    return (time.perf_counter() - t0) / n * 1000


# ----------------------------------------------------------------------------- risk score
def split_events(data, split):
    return np.where(data.split == split)[0]


def risk_scores(P, data, ev, alpha=ALPHA):
    """R_t = alpha*p_t + (1-alpha)*R_(t-1) inside every stream; R_0 = p_0.  ev = sorted event ids."""
    pos = ev - data.start[ev]
    p = P[ev].astype(np.float64)
    R = np.zeros(len(ev))
    for j in range(pos.max() + 1):
        sel = np.where(pos == j)[0]
        R[sel] = p[sel] if j == 0 else alpha * p[sel] + (1 - alpha) * R[sel - 1]
    return R


def stream_table(data, ev):
    """One row per stream: [first local index, end local index, first-fraud position (-1 if none)]."""
    st = data.start[ev]
    bounds = np.r_[0, np.where(st[1:] != st[:-1])[0] + 1, len(ev)]
    rows = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        fr = np.where(data.fraud[ev[a:b]] == 1)[0]
        rows.append((a, b, int(fr[0]) if len(fr) else -1))
    return rows


def stream_scores(R, rows):
    """Alert score of a stream = max R_t BEFORE the first fraud (whole stream if no fraud).
    Streams whose first event is already fraud are skipped: nothing precedes it."""
    out = []
    for a, b, f in rows:
        if f == 0:
            continue
        end = b if f < 0 else a + f
        out.append((a, b, f, R[a:end].max()))
    return out


def alert_threshold(R, data, ev_val):
    """Pick R_t threshold on VALIDATION: maximise stream-level F1 (alert vs. fraud stream)."""
    sc = stream_scores(R, stream_table(data, ev_val))
    s = np.array([x[3] for x in sc])
    y = np.array([x[2] > 0 for x in sc])
    best = (-1, None)
    for t in np.unique(np.quantile(s, np.linspace(0.5, 0.9995, 400))):
        pred = s >= t
        tp = (pred & y).sum()
        f1 = 2 * tp / max(pred.sum() + y.sum(), 1)
        if f1 > best[0]:
            best = (f1, float(t))
    return best[1], dict(val_f1=best[0], val_streams=len(s), val_fraud_streams=int(y.sum()))


def alert_report(R, data, ev, thr):
    """Stream-level alert statistics + lead time (events and hours) on the given split."""
    rows = stream_table(data, ev)
    fraud_lead, fraud_hours, n_fraud, caught = [], [], 0, 0
    n_norm, false_alarm = 0, 0
    for a, b, f in rows:
        if f == 0:
            continue
        if f < 0:
            n_norm += 1
            false_alarm += bool((R[a:b] >= thr).any())
        else:
            n_fraud += 1
            hit = np.where(R[a:a + f] >= thr)[0]               # alerts strictly before the fraud
            if len(hit):
                caught += 1
                fraud_lead.append(f - hit[0])
                fraud_hours.append(float(data.step[ev[a + f]] - data.step[ev[a + hit[0]]]))
    return dict(fraud_streams=n_fraud, caught_early=caught, recall=caught / max(n_fraud, 1),
                normal_streams=n_norm, false_alarms=false_alarm,
                false_alarm_rate=false_alarm / max(n_norm, 1),
                precision=caught / max(caught + false_alarm, 1),
                lead_events_median=float(np.median(fraud_lead)) if fraud_lead else None,
                lead_events_mean=float(np.mean(fraud_lead)) if fraud_lead else None,
                lead_hours_median=float(np.median(fraud_hours)) if fraud_hours else None)


def average_curves(R, P, data, ev, K=10, seed=0):
    """Mean R_t and p_t aligned to the event just before fraud (offset 0) for fraud streams;
    normal streams use one random event per stream as the anchor."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("fraud", "normal"):
        sumR, sumP, cnt = np.zeros(K + 1), np.zeros(K + 1), np.zeros(K + 1)
        for a, b, f in stream_table(data, ev):
            if f == 0 or (name == "fraud") != (f > 0):
                continue
            anchor = a + f - 1 if f > 0 else a + rng.integers(0, b - a)
            for k in range(K + 1):                            # offset -k
                j = anchor - k
                if j >= a:
                    sumR[K - k] += R[j]
                    sumP[K - k] += P[ev[j]]
                    cnt[K - k] += 1
        out[name] = dict(offsets=list(range(-K, 1)), R=(sumR / np.maximum(cnt, 1)).tolist(),
                         p=(sumP / np.maximum(cnt, 1)).tolist(), n=cnt.astype(int).tolist())
    return out


# ----------------------------------------------------------------------------- main
def main():
    data = EventData("safe")
    summary, runs = {}, {}
    names = sorted({os.path.basename(f)[:-4] for f in glob.glob(f"{RUNS}/*.npz")})
    for nm in names:
        model, feats, s = nm.rsplit("_", 2)
        seed = int(s[1:])
        runs[nm] = metrics_for(load_P(model, seed, feats, data.N), data)
        if max(runs[nm]["auc"], 0) > 0.99:
            print("WARNING: AUC > 0.99 for", nm, "- suspect label leakage")
    # mean +- std over seeds for the safe feature set
    for model in LABELS:
        rs = [runs[f"{model}_safe_s{s}"] for s in SEEDS if f"{model}_safe_s{s}" in runs]
        if not rs:
            continue
        agg = {k: dict(mean=float(np.mean([r[k] for r in rs])), std=float(np.std([r[k] for r in rs])) if len(rs) > 1 else None)
               for k in ("auc", "prauc", "f1", "recall_at_85p", "recall_at_1pct")}
        agg["n_seeds"] = len(rs)
        agg["latency_ms"] = latency_ms(model, 0, data)
        agg["params"] = json.load(open(f"{RUNS}/{model}_safe_s0.json"))["params"]
        agg["epochs"] = float(np.mean([json.load(open(f"{RUNS}/{model}_safe_s{s}.json"))["epochs"]
                                       for s in SEEDS if f"{model}_safe_s{s}" in runs]))
        summary[model] = agg
    # feature-set (leakage) checks, single seed
    featcheck = {nm: runs[nm] for nm in runs if "_safe_" not in nm}
    it = data.idx(2)
    base = dict(test_points=len(it), test_positives=int(data.label[it].sum()),
                prevalence=float(data.label[it].mean()))

    # ---- Step 4: risk score for the primary model (TEBIF, seed 0 - fixed in advance, not cherry-picked)
    P = load_P("tebif", 0, "safe", data.N)
    ev_v, ev_t = split_events(data, 1), split_events(data, 2)
    Rv, Rt = risk_scores(P, data, ev_v), risk_scores(P, data, ev_t)
    thr, vinfo = alert_threshold(Rv, data, ev_v)
    risk = dict(alpha=ALPHA, threshold=thr, validation=vinfo,
                val_report=alert_report(Rv, data, ev_v, thr),
                test_report=alert_report(Rt, data, ev_t, thr),
                curves=average_curves(Rt, P, data, ev_t))
    json.dump(dict(base=base, models=summary, feature_checks=featcheck, runs=runs),
              open("results/metrics.json", "w"), indent=1)
    json.dump(risk, open("results/risk.json", "w"), indent=1)

    print("\nTest set:", base)
    print("%-34s %-14s %-14s %-14s %-14s %-9s" % ("model", "AUC-ROC", "PR-AUC", "F1", "R@85%P", "ms/event"))
    for m, a in summary.items():
        f = lambda k: "%.3f" % a[k]["mean"] + ("±%.3f" % a[k]["std"] if a[k]["std"] is not None else "")
        print("%-34s %-14s %-14s %-14s %-14s %.2f" % (LABELS[m], f("auc"), f("prauc"), f("f1"),
                                                      f("recall_at_85p"), a["latency_ms"]))
    print("\nFeature-set checks (seed 0):")
    for k, v in featcheck.items():
        print("  %-22s AUC %.3f PR-AUC %.4f" % (k, v["auc"], v["prauc"]))
    print("\nRisk: alpha", ALPHA, "threshold %.4f" % thr, vinfo)
    print("val ", risk["val_report"])
    print("test", risk["test_report"])
    figures(summary, risk)


def figures(summary, risk):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, k, t in zip(ax, ("auc", "prauc"), ("AUC-ROC", "PR-AUC")):
        ms = list(summary)
        a.bar(range(len(ms)), [summary[m][k]["mean"] for m in ms],
              yerr=[summary[m][k]["std"] or 0 for m in ms], capsize=3, color="#4c78a8")
        a.set_xticks(range(len(ms)))
        a.set_xticklabels([m for m in ms], rotation=30, ha="right")
        a.set_title(t + " (test, 1 seed - no error bars)")
    plt.tight_layout()
    plt.savefig("results/fig_metrics.png", dpi=130)
    plt.close()
    c = risk["curves"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, col in (("fraud", "#d62728"), ("normal", "#2ca02c")):
        ax.plot(c[name]["offsets"], c[name]["R"], color=col, marker="o",
                label=f"{name} accounts (n≈{c[name]['n'][-1]})")
    ax.axhline(risk["threshold"], color="k", ls="--", label="alert threshold (chosen on validation)")
    ax.set_xlabel("events relative to the event just before fraud (0)")
    ax.set_ylabel(f"mean risk score R_t (α={risk['alpha']})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/fig_risk_curve.png", dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
