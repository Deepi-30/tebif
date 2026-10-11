"""Step 0: quick data check of the PaySim CSV (columns, fraud rate, sequence lengths, leakage)."""
import glob
import pandas as pd
import numpy as np

path = glob.glob("data/*.csv")[0]
df = pd.read_csv(path)
print("file:", path)
print("columns:", list(df.columns))
print("rows:", len(df))
print("fraud rate: %.4f%% (%d frauds)" % (100 * df.isFraud.mean(), df.isFraud.sum()))
print("step range:", df.step.min(), "-", df.step.max())
print("types:\n", df.groupby("type").isFraud.agg(["count", "sum"]))

# sender-only sequence lengths
s = df.nameOrig.value_counts()
print("\nSENDER-ONLY events/account: n=%d mean=%.3f median=%.1f max=%d, share with >=2 events=%.4f%%"
      % (len(s), s.mean(), s.median(), s.max(), 100 * (s >= 2).mean()))

# sender OR receiver sequence lengths
both = pd.concat([df.nameOrig, df.nameDest]).value_counts()
print("SENDER+RECEIVER events/account: n=%d mean=%.3f median=%.1f max=%d, share >=2: %.2f%%, >=5: %.2f%%"
      % (len(both), both.mean(), both.median(), both.max(), 100 * (both >= 2).mean(), 100 * (both >= 5).mean()))

# merchants (M...) only ever receive PAYMENT; look at non-merchant accounts too
cust = both[~both.index.str.startswith("M")]
print("non-merchant accounts: n=%d mean=%.3f median=%.1f" % (len(cust), cust.mean(), cust.median()))

# fraud accounts: which role do they play, and how long are their streams
fr = df[df.isFraud == 1]
fa = pd.unique(pd.concat([fr.nameOrig, fr.nameDest]))
print("accounts touching a fraud tx:", len(fa), " mean stream len: %.2f median %.1f" % (both[fa].mean(), both[fa].median()))
print("fraud tx types:", fr.type.value_counts().to_dict())
print("isFlaggedFraud sum:", df.isFlaggedFraud.sum())

# leakage probe: balance columns on fraud vs normal TRANSFER/CASH_OUT
x = df[df.type.isin(["TRANSFER", "CASH_OUT"])].copy()
x["err_orig"] = x.oldbalanceOrg - x.amount - x.newbalanceOrig
x["drained"] = (x.newbalanceOrig == 0) & (x.oldbalanceOrg > 0)
print("\nLeak probe (TRANSFER/CASH_OUT): share where origin account fully drained")
print(x.groupby("isFraud")[["drained"]].mean())
print("share oldbalanceDest==0 :", x.groupby("isFraud").apply(lambda g: (g.oldbalanceDest == 0).mean()).to_dict())
