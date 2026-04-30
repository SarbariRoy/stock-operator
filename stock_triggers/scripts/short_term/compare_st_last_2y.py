from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
OOS_PATH = DATA_DIR / "st_score_walk_forward_oos_with_xgboost.csv"
SUMMARY_OUT = DATA_DIR / "st_score_last_2y_summary.csv"
BLENDS_OUT = DATA_DIR / "st_score_last_2y_best_blends.csv"


def rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    working = pd.DataFrame(
        {
            "score": pd.to_numeric(scores, errors="coerce"),
            "label": pd.to_numeric(labels, errors="coerce"),
        }
    ).dropna()
    working = working[working["label"].isin([0, 1])]
    if working.empty:
        return float("nan")

    n_pos = int((working["label"] == 1).sum())
    n_neg = int((working["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = working["score"].rank(method="average")
    sum_ranks_pos = float(ranks[working["label"] == 1].sum())
    return float((sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg))


def spearman_rank(left: pd.Series, right: pd.Series) -> float:
    working = pd.DataFrame(
        {
            "left": pd.to_numeric(left, errors="coerce"),
            "right": pd.to_numeric(right, errors="coerce"),
        }
    ).dropna()
    if len(working) < 3:
        return float("nan")
    return float(working["left"].rank(method="average").corr(working["right"].rank(method="average")))


def summarize(df: pd.DataFrame, score_col: str) -> dict:
    working = df.copy()
    working["_score"] = pd.to_numeric(working[score_col], errors="coerce")
    working["_return"] = pd.to_numeric(working["return_pct"], errors="coerce")
    working = working[working["_score"].notna()].copy()

    closed = working[working["status"].isin(["target", "stop"])].copy()
    closed["_target_hit"] = (closed["status"] == "target").astype(int)

    auc = rank_auc(closed["_score"], closed["_target_hit"])
    rank_ic = spearman_rank(working["_score"], working["_return"])

    q_low = float(working["_score"].quantile(0.2))
    q_high = float(working["_score"].quantile(0.8))

    top_all = working[working["_score"] >= q_high]
    bottom_all = working[working["_score"] <= q_low]
    top_closed = closed[closed["_score"] >= q_high]
    bottom_closed = closed[closed["_score"] <= q_low]

    top_wr = float(top_closed["_target_hit"].mean() * 100.0) if not top_closed.empty else float("nan")
    bottom_wr = float(bottom_closed["_target_hit"].mean() * 100.0) if not bottom_closed.empty else float("nan")
    win_lift = float(top_wr - bottom_wr) if not (pd.isna(top_wr) or pd.isna(bottom_wr)) else float("nan")

    top_ret = float(pd.to_numeric(top_all["_return"], errors="coerce").mean()) if not top_all.empty else float("nan")
    bottom_ret = float(pd.to_numeric(bottom_all["_return"], errors="coerce").mean()) if not bottom_all.empty else float("nan")
    return_spread = float(top_ret - bottom_ret) if not (pd.isna(top_ret) or pd.isna(bottom_ret)) else float("nan")

    return {
        "n_scored": int(len(working)),
        "n_closed_scored": int(len(closed)),
        "auc": auc,
        "rank_ic": rank_ic,
        "win_rate_lift_pp": win_lift,
        "return_spread_pct": return_spread,
    }


def add_deltas(df: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    out = df.copy()
    out["auc_delta_vs_logistic"] = out["auc"] - float(baseline["auc"])
    out["rank_ic_delta_vs_logistic"] = out["rank_ic"] - float(baseline["rank_ic"])
    out["win_rate_lift_delta_vs_logistic_pp"] = out["win_rate_lift_pp"] - float(baseline["win_rate_lift_pp"])
    out["return_spread_delta_vs_logistic_pct"] = out["return_spread_pct"] - float(baseline["return_spread_pct"])
    return out


def main() -> None:
    if not OOS_PATH.exists():
        raise SystemExit(f"Missing OOS file: {OOS_PATH}")

    oos = pd.read_csv(OOS_PATH)
    oos["signal_date"] = pd.to_datetime(oos.get("signal_date"), errors="coerce")
    oos = oos[oos["signal_date"].notna()].copy()
    if oos.empty:
        raise SystemExit("OOS file has no valid signal_date rows.")

    max_date = pd.Timestamp(oos["signal_date"].max()).normalize()
    cutoff = max_date - pd.DateOffset(years=2)
    oos_2y = oos[oos["signal_date"] >= cutoff].copy()

    modes = {
        "logistic": "score_logistic",
        "svm": "score_svm",
        "xgboost": "score_xgboost",
        "random_forest": "score_random_forest",
        "logistic+svm": "score_logistic_svm",
        "logistic+rf": "score_logistic_rf",
        "svm+rf": "score_svm_rf",
        "xgboost+rf": "score_xgboost_rf",
        "logistic+svm+xgboost": "score_logistic_svm_xgboost",
        "logistic+svm+rf": "score_logistic_svm_rf",
        "logistic+xgboost+rf": "score_logistic_xgboost_rf",
        "svm+xgboost+rf": "score_svm_xgboost_rf",
        "logistic+svm+xgboost+rf": "score_logistic_svm_xgboost_rf",
    }

    rows: list[dict] = []
    for mode, col in modes.items():
        if col not in oos_2y.columns:
            continue
        metrics = summarize(oos_2y, col)
        metrics["mode"] = mode
        rows.append(metrics)

    summary = pd.DataFrame(rows)
    baseline_row = summary.loc[summary["mode"] == "logistic"]
    if baseline_row.empty:
        raise SystemExit("Logistic baseline not found in 2Y summary.")
    baseline = baseline_row.iloc[0].to_dict()
    summary = add_deltas(summary, baseline)
    summary = summary.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False])
    summary.to_csv(SUMMARY_OUT, index=False)

    blend_rows: list[dict] = []

    # 2-way logistic+rf sweep
    for w_rf in np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2):
        w_log = round(1.0 - float(w_rf), 2)
        tmp = oos_2y.copy()
        tmp["_blend"] = (
            (w_log * pd.to_numeric(tmp["score_logistic"], errors="coerce").fillna(0.0))
            + (float(w_rf) * pd.to_numeric(tmp["score_random_forest"], errors="coerce").fillna(0.0))
        )
        metrics = summarize(tmp, "_blend")
        metrics.update({"blend_type": "logistic+rf", "w_logistic": w_log, "w_svm": 0.0, "w_rf": float(w_rf)})
        blend_rows.append(metrics)

    # 3-way logistic+svm+rf simplex sweep
    for w_log in np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1):
        for w_svm in np.round(np.arange(0.0, 1.0 - float(w_log) + 1e-9, 0.1), 1):
            w_rf = round(1.0 - float(w_log) - float(w_svm), 1)
            if w_rf < -1e-9:
                continue
            w_rf = max(0.0, w_rf)
            tmp = oos_2y.copy()
            tmp["_blend"] = (
                (float(w_log) * pd.to_numeric(tmp["score_logistic"], errors="coerce").fillna(0.0))
                + (float(w_svm) * pd.to_numeric(tmp["score_svm"], errors="coerce").fillna(0.0))
                + (float(w_rf) * pd.to_numeric(tmp["score_random_forest"], errors="coerce").fillna(0.0))
            )
            metrics = summarize(tmp, "_blend")
            metrics.update(
                {
                    "blend_type": "logistic+svm+rf",
                    "w_logistic": float(w_log),
                    "w_svm": float(w_svm),
                    "w_rf": float(w_rf),
                }
            )
            blend_rows.append(metrics)

    blends = pd.DataFrame(blend_rows)
    blends = add_deltas(blends, baseline)
    blends = blends.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False])
    blends.to_csv(BLENDS_OUT, index=False)

    print("LAST_2Y_COMPARE_COMPLETE")
    print(f"Window: {cutoff.date()} to {max_date.date()}")
    print(f"Rows in window: {len(oos_2y)}")

    print("\nTOP_MODELS_LAST_2Y")
    print(
        summary[
            [
                "mode",
                "auc",
                "rank_ic",
                "win_rate_lift_pp",
                "return_spread_pct",
                "auc_delta_vs_logistic",
                "win_rate_lift_delta_vs_logistic_pp",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nTOP_BLENDS_LAST_2Y")
    print(
        blends[
            [
                "blend_type",
                "w_logistic",
                "w_svm",
                "w_rf",
                "auc",
                "win_rate_lift_pp",
                "return_spread_pct",
                "auc_delta_vs_logistic",
                "win_rate_lift_delta_vs_logistic_pp",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    target = blends[
        (blends["blend_type"] == "logistic+svm+rf")
        & (blends["w_logistic"].round(1) == 0.7)
        & (blends["w_svm"].round(1) == 0.1)
        & (blends["w_rf"].round(1) == 0.2)
    ]
    if not target.empty:
        row = target.iloc[0]
        print("\nTARGET_BLEND_0p7_0p1_0p2_LAST_2Y")
        print(
            f"auc={row['auc']:.6f}, rank_ic={row['rank_ic']:.6f}, "
            f"lift={row['win_rate_lift_pp']:.6f}, spread={row['return_spread_pct']:.6f}, "
            f"auc_delta={row['auc_delta_vs_logistic']:.6f}, lift_delta={row['win_rate_lift_delta_vs_logistic_pp']:.6f}"
        )

    print(f"\nSaved: {SUMMARY_OUT}")
    print(f"Saved: {BLENDS_OUT}")


if __name__ == "__main__":
    main()
