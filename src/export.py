"""
export.py - Step 6/7: write results/results.json and embed it into frontend/index.html.

    python src/export.py
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import EventData, TYPES, WINDOW, SPLIT_END, HORIZON
from evaluate import load_P, risk_scores, split_events, stream_table, ALPHA, LABELS
from explain import load_tebif, attention, explain_focus

N_DEMO = {"fraud_caught": 8, "fraud_missed": 8, "normal": 7, "false_alarm": 7}


def event_amount(data, g):
    return float(np.expm1(data.cont[g, 0] * data.sd[0] + data.mu[0]))


def main():
    data = EventData("safe")
    m = load_tebif(data)
    P = load_P("tebif", 0, "safe", data.N)
    ev = split_events(data, 2)
    risk = json.load(open("results/risk.json"))
    metrics = json.load(open("results/metrics.json"))
    faith = json.load(open("results/faithfulness.json"))
    thr = risk["threshold"]
    R = risk_scores(P, data, ev, ALPHA)

    # ---- categorise test streams (display only streams of 3..30 events that start with history)
    cats = {k: [] for k in N_DEMO}
    for a, b, f in stream_table(data, ev):
        if f == 0 or not (3 <= b - a <= WINDOW):
            continue
        end = b if f < 0 else a + f
        hit = np.where(R[a:end] >= thr)[0]
        if f > 0:
            cats["fraud_caught" if len(hit) else "fraud_missed"].append((a, b, f, hit[0] if len(hit) else None))
        else:
            cats["false_alarm" if len(hit) else "normal"].append((a, b, f, hit[0] if len(hit) else None))
    rng = np.random.default_rng(1)
    accounts = []
    for cat, k in N_DEMO.items():
        pick = rng.permutation(len(cats[cat]))[:k]
        for pi in pick:
            a, b, f, h = cats[cat][pi]
            g = ev[a:b]
            n = b - a
            if cat in ("fraud_caught", "false_alarm"):
                fpos = int(h)                                   # first alert event
            elif cat == "fraud_missed":
                fpos = f - 1                                    # event just before the fraud
            else:
                fpos = int(np.argmax(R[a:b]))                   # most risky moment
            # per-event attention at every prediction time t (rows) over events i <= t
            att = attention(m, data.batch(g))
            rows = np.zeros((n, n))
            for t in range(n):
                for j in range(WINDOW):
                    i = t - (WINDOW - 1 - j)
                    if 0 <= i <= t:
                        rows[t, i] = att[t, j]
            ex = explain_focus(m, data, g[fpos], float(P[g[fpos]]), float(R[a + fpos]),
                               alerted=cat in ("fraud_caught", "false_alarm"))
            base = int(g[0])
            ex["window_stream_pos"] = [None if q is None else int(q - base) for q in ex.pop("window_global")]
            for tf in ex["top_features"]:
                tf["event"] = None if tf["event"] is None else int(tf["event"] - base)
            events = [dict(type=TYPES[data.tok[x] // 2], role="receiver" if data.tok[x] % 2 else "sender",
                           amount=round(event_amount(data, x), 2), step=int(data.step[x]),
                           dt=int(data.step[x] - data.step[g[i - 1]]) if i else 0,
                           p=round(float(P[x]), 5), is_fraud=bool(data.fraud[x]))
                      for i, x in enumerate(g)]
            accounts.append(dict(
                id=str(data.acc_name[g[0]]), category=cat,
                true_label="fraud" if f > 0 else "normal",
                fraud_event=int(f) if f > 0 else None, focus_event=int(fpos),
                events=events, attention_by_time=np.round(rows, 4).tolist(), **ex))

    out = dict(
        meta=dict(title="TEBIF - Transformer-Based Explainable Behavioural Intelligence Framework",
                  note="All data is synthetic (PaySim). No real customer data.",
                  window=WINDOW, horizon_steps=HORIZON, split_end_steps=SPLIT_END.tolist(),
                  test=metrics["base"]),
        metrics={LABELS[k]: v for k, v in metrics["models"].items()},
        ablation=[dict(name=LABELS[k], **{q: metrics["models"][k][q] for q in ("auc", "prauc")})
                  for k in ("vanilla", "tebif_dt", "tebif_ctime", "tebif")],
        feature_checks=metrics["feature_checks"],
        risk=dict(alpha=ALPHA, threshold=thr, curves=risk["curves"], validation=risk["validation"],
                  val_report=risk["val_report"], test_report=risk["test_report"]),
        faithfulness=faith, accounts=accounts)
    json.dump(out, open("results/results.json", "w"), indent=1)
    print("results.json:", len(accounts), "demo accounts;", {c: sum(a['category'] == c for a in accounts) for c in N_DEMO})

    # ---- embed into the single-file frontend
    tpl = open("frontend/template.html", encoding="utf8").read()
    blob = json.dumps(out).replace("</", "<\\/")
    open("frontend/index.html", "w", encoding="utf8").write(tpl.replace("__RESULTS_JSON__", blob))
    print("frontend/index.html written (%.0f KB)" % (os.path.getsize("frontend/index.html") / 1024))


if __name__ == "__main__":
    main()
