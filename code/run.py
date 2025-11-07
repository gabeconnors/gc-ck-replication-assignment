import pandas as pd
import statsmodels.api as sm
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "minimal.csv"
OUT  = Path(__file__).resolve().parents[1] / "output"
OUT.mkdir(exist_ok=True, parents=True)

# Load minimal data
df = pd.read_csv(DATA)

# Core sample: stores with both waves
df_core = df.dropna(subset=["NJ","fte1","fte2"]).copy()
df_core["dFTE"] = df_core["fte2"] - df_core["fte1"]

# DiD (means)
did_means = df_core.groupby("NJ")["dFTE"].mean()
did_effect = did_means.loc[1] - did_means.loc[0]

# OLS: ΔFTE ~ NJ (HC1 robust SEs)
X1 = sm.add_constant(df_core[["NJ"]])
m1 = sm.OLS(df_core["dFTE"], X1).fit(cov_type="HC1")

# OLS: ΔFTE ~ GAP (uses only obs with GAP)
df_gap = df_core.dropna(subset=["GAP"]).copy()
X2 = sm.add_constant(df_gap[["GAP"]])
m2 = sm.OLS(df_gap["dFTE"], X2).fit(cov_type="HC1")

# Write a simple report
with open(OUT / "results.txt","w") as f:
    f.write("=== Difference-in-Differences (means of ΔFTE) ===\n")
    f.write(did_means.rename({0:"PA",1:"NJ"}).to_string()+"\n")
    f.write(f"\nDiD (NJ - PA) = {did_effect:.3f}\n\n")
    f.write("=== OLS: ΔFTE ~ NJ (HC1) ===\n")
    f.write(str(m1.summary())+"\n\n")
    f.write("=== OLS: ΔFTE ~ GAP (HC1) ===\n")
    f.write(str(m2.summary())+"\n")

print("Done. See output/results.txt")
