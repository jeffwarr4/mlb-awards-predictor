# src/train_model.py
# ---------------------------------------------------------------
# Step 3 of the pipeline.
# Trains four models (MVP top5, MVP winner, CY top5, CY winner)
# on historical data and saves them to models/.
#
# Inputs:
#   data/processed/player_season_features_fg.csv  (from merge_fangraphs.py)
#
# Outputs (per task in models/<task_name>/):
#   model_logreg.joblib
#   model_randomforest.joblib
#   feature_columns.joblib
#   recall_at5_logreg.csv
#   recall_at5_randomforest.csv
#   logreg_top_coeffs.csv
#   rf_top_importances.csv
#   metrics_summary.csv           ← all tasks + models in one place
#
# Run:
#   python src/train_model.py
#   python src/train_model.py --task MVP_top5   # single task
# ---------------------------------------------------------------

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import (
    FG_DATASET, MODELS_DIR, TASKS,
    TRAIN_START, TRAIN_END, TEST_START, TEST_END, RANDOM_STATE
)


# ── Feature exclusion list ────────────────────────────────────────
EXCLUDE_ALWAYS = {
    # identifiers
    "playerID","teamID","lgID","yearID",
    # all label columns
    "is_top5_MVP","is_winner_MVP","is_top5_CY","is_winner_CY",
    # raw vote data (unavailable at prediction time)
    "voteShare_MVP","voteShare_CY",
    "pointsWon_MVP","pointsMax_MVP","votesFirst_MVP",
    "pointsWon_CY","pointsMax_CY","votesFirst_CY",
}


# ── Metrics ───────────────────────────────────────────────────────
def global_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_hat = (y_prob >= threshold).astype(int)
    return {
        "AUC":       round(roc_auc_score(y_true, y_prob), 4),
        "AP":        round(average_precision_score(y_true, y_prob), 4),  # area under PR
        "F1":        round(f1_score(y_true, y_hat, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_hat, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_hat, zero_division=0), 4),
    }


def recall_at_n(df_eval: pd.DataFrame, prob_col: str, label_col: str, n: int = 5) -> tuple:
    """
    For each (yearID, lgID): rank predictions, take top-n, count true positives.
    Returns (mean_recall, detail_df).
    """
    rows = []
    for (year, lg), grp in df_eval.groupby(["yearID","lgID"]):
        grp = grp.sort_values(prob_col, ascending=False)
        pred_top = set(grp.head(n).index)
        true_top = set(grp.index[grp[label_col] == 1])
        if not true_top:
            continue
        hits = len(pred_top & true_top)
        rows.append({
            "yearID":     year,
            "lgID":       lg,
            "hits":       hits,
            "true_pos":   len(true_top),
            "recall_at_n": hits / min(n, len(true_top)),
        })
    detail = pd.DataFrame(rows).sort_values(["yearID","lgID"])
    return (detail["recall_at_n"].mean() if not detail.empty else np.nan), detail


def top1_hit_rate(df_eval: pd.DataFrame, prob_col: str, label_col: str) -> float:
    """Did the model's #1 pick per (year, league) actually win?"""
    hits = 0; total = 0
    for _, grp in df_eval.groupby(["yearID","lgID"]):
        top = grp.sort_values(prob_col, ascending=False).head(1)
        if not top.empty:
            hits  += int(top[label_col].iloc[0] == 1)
            total += 1
    return hits / total if total else np.nan


# ── Main training function ────────────────────────────────────────
def train_task(task_name: str, label_col: str, df: pd.DataFrame,
               feature_cols: list, all_metrics: list) -> None:

    print(f"\n{'='*60}")
    print(f"  Task: {task_name}  |  Label: {label_col}")
    print(f"{'='*60}")

    y = df[label_col].astype(int)
    X = df[feature_cols]

    # Temporal split — never leak future data into training
    train_mask = (df["yearID"] >= TRAIN_START) & (df["yearID"] <  TEST_START)
    test_mask  = (df["yearID"] >= TEST_START)  & (df["yearID"] <= TEST_END)

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask],  y[test_mask]
    meta_te    = df.loc[test_mask, ["yearID","lgID","playerID"]].copy()

    print(f"  Train: {train_mask.sum():,} rows ({TRAIN_START}–{TEST_START-1}) | "
          f"Pos: {y_tr.sum()}")
    print(f"  Test:  {test_mask.sum():,} rows ({TEST_START}–{TEST_END})   | "
          f"Pos: {y_te.sum()}")

    if y_tr.sum() < 5:
        print(f"  ⚠️  Too few positives in training — skipping {task_name}")
        return

    # ── Logistic Regression ──
    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=0.5,           # mild regularisation; tune if needed
            random_state=RANDOM_STATE,
        ))
    ])

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    lr.fit(X_tr, y_tr)
    rf.fit(X_tr, y_tr)

    p_lr = lr.predict_proba(X_te)[:,1]
    p_rf = rf.predict_proba(X_te)[:,1]

    # ── Global metrics ──
    print(f"\n  Global metrics on {TEST_START}–{TEST_END}:")
    for name, probs in [("LogReg", p_lr), ("RandForest", p_rf)]:
        if y_te.sum() == 0:
            print(f"  {name}: no positives in test set — skipping metrics")
            continue
        m = global_metrics(y_te, probs)
        print(f"  {name:12s}  AUC={m['AUC']}  AP={m['AP']}  "
              f"F1={m['F1']}  Prec={m['Precision']}  Rec={m['Recall']}")
        all_metrics.append({"task": task_name, "model": name, **m})

    # ── Recall@5 per year/league ──
    eval_df = meta_te.copy()
    eval_df["y_true"] = y_te.values
    eval_df["p_lr"]   = p_lr
    eval_df["p_rf"]   = p_rf

    r5_lr,  r5_lr_tbl  = recall_at_n(eval_df.rename(columns={"y_true":label_col}),
                                       "p_lr", label_col, n=5)
    r5_rf,  r5_rf_tbl  = recall_at_n(eval_df.rename(columns={"y_true":label_col}),
                                       "p_rf", label_col, n=5)
    t1_lr = top1_hit_rate(eval_df.rename(columns={"y_true":label_col}), "p_lr", label_col)
    t1_rf = top1_hit_rate(eval_df.rename(columns={"y_true":label_col}), "p_rf", label_col)

    print(f"\n  Recall@5 (mean over {TEST_START}–{TEST_END}):")
    print(f"  LogReg:     {r5_lr:.3f}   Top-1 hit rate: {t1_lr:.3f}")
    print(f"  RandForest: {r5_rf:.3f}   Top-1 hit rate: {t1_rf:.3f}")

    # Update metrics table with Recall@5
    for row in all_metrics:
        if row["task"] == task_name:
            if row["model"] == "LogReg":
                row["Recall@5"] = round(r5_lr, 4)
                row["Top1HitRate"] = round(t1_lr, 4)
            elif row["model"] == "RandForest":
                row["Recall@5"] = round(r5_rf, 4)
                row["Top1HitRate"] = round(t1_rf, 4)

    # ── Save models & artefacts ──
    out_dir = MODELS_DIR / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr,           out_dir / "model_logreg.joblib")
    joblib.dump(rf,           out_dir / "model_randomforest.joblib")
    joblib.dump(feature_cols, out_dir / "feature_columns.joblib")

    r5_lr_tbl.to_csv(out_dir / "recall_at5_logreg.csv",       index=False)
    r5_rf_tbl.to_csv(out_dir / "recall_at5_randomforest.csv", index=False)

    # Feature importances / coefficients
    try:
        coefs = pd.Series(
            lr.named_steps["clf"].coef_[0], index=feature_cols
        ).sort_values(key=np.abs, ascending=False)
        coefs.head(25).to_csv(out_dir / "logreg_top_coeffs.csv", header=["coefficient"])
    except Exception: pass

    try:
        imps = pd.Series(
            rf.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)
        imps.head(25).to_csv(out_dir / "rf_top_importances.csv", header=["importance"])
    except Exception: pass

    print(f"\n  💾  Saved to {out_dir.resolve()}")


# ── Entry point ───────────────────────────────────────────────────
def train_all(task_filter: str = None) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"📂  Loading {FG_DATASET.name} ...")
    df = pd.read_csv(FG_DATASET)
    df = df[(df["yearID"] >= TRAIN_START) & df["lgID"].isin(["AL","NL"])].copy()
    print(f"    {len(df):,} rows | years {df['yearID'].min()}–{df['yearID'].max()}")

    # Build feature column list (exclude identifiers, labels, raw award data)
    feat_cols = [
        c for c in df.columns
        if c not in EXCLUDE_ALWAYS and df[c].dtype != object
    ]
    # Replace inf / NaN in features
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"    Features: {len(feat_cols)}")

    # Run each task
    all_metrics = []
    for task_name, label_col in TASKS.items():
        if task_filter and task_name != task_filter:
            continue
        if label_col not in df.columns:
            print(f"  ⚠️  {label_col} not found in dataset — skipping {task_name}")
            continue
        train_task(task_name, label_col, df, feat_cols, all_metrics)

    # Save combined metrics summary
    if all_metrics:
        summary = pd.DataFrame(all_metrics)
        summary_path = MODELS_DIR / "metrics_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\n📊  Metrics summary saved to {summary_path}")
        print(summary.to_string(index=False))

    print("\n✅  Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLB award prediction models")
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASKS.keys()),
                        help="Train a single task (default: all)")
    args = parser.parse_args()
    train_all(task_filter=args.task)
