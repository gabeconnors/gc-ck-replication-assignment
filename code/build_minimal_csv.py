import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "public.dat"
OUT = Path(__file__).resolve().parents[1] / "data" / "minimal.csv"

names = [
    "SHEET","CHAINr","CO_OWNED","STATEr","SOUTHJ","CENTRALJ","NORTHJ","PA1","PA2","SHORE",
    "NCALLS","EMPFT","EMPPT","NMGRS","WAGE_ST","INCTIME","FIRSTINC","BONUS","PCTAFF","MEAL",
    "OPEN","HRSOPEN","PSODA","PFRY","PENTREE","NREGS","NREGS11","TYPE2","STATUS2","DATE2",
    "NCALLS2","EMPFT2","EMPPT2","NMGRS2","WAGE_ST2","INCTIME2","FIRSTIN2","SPECIAL2",
    "MEALS2","OPEN2R","HRSOPEN2","PSODA2","PFRY2","PENTREE2","NREGS2","NREGS112"
]

# Read flat file
df = pd.read_csv(RAW, sep=r"\s+", header=None, names=names, engine="python")

# Convert used columns to numeric
num_cols = ["STATEr","EMPFT","EMPPT","NMGRS","EMPFT2","EMPPT2","NMGRS2","WAGE_ST"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# NJ indicator
df["NJ"] = (df["STATEr"] == 1).astype(int)

# FTEs
df["fte1"] = df["NMGRS"] + df["EMPFT"] + 0.5 * df["EMPPT"]
df["fte2"] = df["NMGRS2"] + df["EMPFT2"] + 0.5 * df["EMPPT2"]

# Wage GAP: proportional raise needed to reach $5.05 (NJ only)
df["GAP"] = 0.0
mask_nj = df["NJ"] == 1
df.loc[mask_nj, "GAP"] = ((5.05 - df.loc[mask_nj, "WAGE_ST"]).clip(lower=0)
                          / df.loc[mask_nj, "WAGE_ST"])

# Keep minimal schema
out = df[["NJ","fte1","fte2","WAGE_ST","GAP"]].rename(columns={"WAGE_ST":"w1"})
OUT.parent.mkdir(exist_ok=True, parents=True)
out.to_csv(OUT, index=False)

print(out.head())
print("Saved minimal CSV to", OUT, "with shape", out.shape)
