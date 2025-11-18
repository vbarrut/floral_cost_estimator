# retrain_model.py
import pandas as pd, numpy as np, re, unicodedata, pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DATA = BASE / "data"
VIS  = BASE / "visualizations"
MODEL_PATH = BASE / "trained_model.pkl"
SRC = DATA / "sales_raw.csv"

def money_to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float)): return float(x)
    s = str(x).replace("$","").replace(",","").strip()
    try: return float(s)
    except: return np.nan

def slugify(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+","_", s).strip("_").lower()
    return s[:80]

if not SRC.exists():
    raise FileNotFoundError(f"Missing {SRC}")

df = pd.read_csv(SRC, encoding="utf-8", engine="python")
rename = {
    "Arrangement Type":"arrangement_type",
    "Unit Price":"unit_price",
    "Quantity":"quantity",
    "Total Price":"total_price",
    "Cost":"materials_cost",
    "Labor":"labor_cost",
}
df = df.rename(columns=rename)

for c in ["unit_price","total_price","materials_cost","labor_cost","quantity"]:
    df[c] = df[c].apply(money_to_float)
df["quantity"] = df["quantity"].fillna(1).astype(int)
df = df.dropna(subset=["arrangement_type","total_price","unit_price","quantity"])

# per-unit features
df["unit_materials"] = (df["materials_cost"]/df["quantity"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
df["unit_labor"]     = (df["labor_cost"]/df["quantity"]).replace([np.inf,-np.inf], np.nan).fillna(0.0)
df["unit_margin"]    = (df["unit_price"] - df["unit_materials"] - df["unit_labor"]).clip(lower=0)
df["style_key"]      = df["arrangement_type"].apply(slugify)

ml = df[["style_key","quantity","unit_materials","unit_labor","unit_margin","total_price"]].dropna(subset=["total_price"])
X = ml[["style_key","quantity","unit_materials","unit_labor","unit_margin"]]
y = ml["total_price"]

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["style_key"]),
        ("num", "passthrough", ["quantity","unit_materials","unit_labor","unit_margin"]),
    ]
)

rf = Pipeline([("pre", pre), ("rf", RandomForestRegressor(n_estimators=500, random_state=7))])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7)
rf.fit(Xtr, ytr)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(rf, f)
print(f"Saved model to {MODEL_PATH}. Train rows: {len(Xtr)}, Test rows: {len(Xte)}")

# Visualizations 
VIS.mkdir(exist_ok=True, parents=True)
yp = rf.predict(Xte)

mae  = mean_absolute_error(yte, yp)
rmse = mean_squared_error(yte, yp, squared=False)
r2   = r2_score(yte, yp)


pre_fitted = rf.named_steps["pre"]
rf_est     = rf.named_steps["rf"]
feat_names = list(pre_fitted.get_feature_names_out())
imps = rf_est.feature_importances_

imp = pd.DataFrame({"feature": feat_names, "importance": imps}).sort_values("importance", ascending=False)

# Chart 1: Feature Importance
plt.figure(figsize=(8,6))
top = imp.head(16)
plt.barh(top["feature"][::-1], top["importance"][::-1])
plt.title("Top Feature Importances")
plt.tight_layout()
plt.savefig(VIS / "feature_importance.png", dpi=150)
plt.close()

# Chart 2: Predicted vs Actual
plt.figure(figsize=(5,5))
plt.scatter(yte, yp, alpha=0.6)
mn, mx = float(min(yte.min(), yp.min())), float(max(yte.max(), yp.max()))
plt.plot([mn, mx], [mn, mx])
plt.xlabel("Actual Total")
plt.ylabel("Predicted Total")
plt.title(f"Predicted vs Actual\nMAE={mae:,.0f}  RMSE={rmse:,.0f}  R²={r2:.2f}")
plt.tight_layout()
plt.savefig(VIS / "predicted_vs_actual.png", dpi=150)
plt.close()

# Metrics text
(Path(VIS / "metrics.txt")).write_text(
    f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.4f}\n", encoding="utf-8"
)

print("Wrote visualizations to", VIS)
