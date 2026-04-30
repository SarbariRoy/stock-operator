from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "stock_triggers" / "data"
OOS_PATH = DATA_DIR / "st_score_walk_forward_oos_with_xgboost.csv"


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


def _prepare_top_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["w_logistic", "w_svm", "w_xgboost", "w_random_forest"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "blend_type" not in out.columns:
        out["blend_type"] = ""
    return out


def main() -> None:
    if not OOS_PATH.exists():
        raise SystemExit(f"Missing OOS file: {OOS_PATH}")

    oos = pd.read_csv(OOS_PATH)
    required_cols = {
        "score_logistic",
        "score_svm",
        "score_xgboost",
        "score_random_forest",
        "status",
        "return_pct",
    }
    missing = sorted(required_cols - set(oos.columns))
    if missing:
        raise SystemExit(f"OOS file is missing required columns: {missing}")

    baseline = summarize(oos, "score_logistic")

    two_way_rows: list[dict] = []
    for w_svm in np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2):
        w_log = round(1.0 - float(w_svm), 2)
        blend_col = (w_log * pd.to_numeric(oos["score_logistic"], errors="coerce").fillna(0.0)) + (
            float(w_svm) * pd.to_numeric(oos["score_svm"], errors="coerce").fillna(0.0)
        )
        tmp = oos.copy()
        tmp["_blend"] = blend_col
        metrics = summarize(tmp, "_blend")
        metrics.update({"w_logistic": w_log, "w_svm": float(w_svm), "blend_type": "logistic+svm"})
        two_way_rows.append(metrics)

    for w_rf in np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2):
        w_log = round(1.0 - float(w_rf), 2)
        blend_col = (w_log * pd.to_numeric(oos["score_logistic"], errors="coerce").fillna(0.0)) + (
            float(w_rf) * pd.to_numeric(oos["score_random_forest"], errors="coerce").fillna(0.0)
        )
        tmp = oos.copy()
        tmp["_blend"] = blend_col
        metrics = summarize(tmp, "_blend")
        metrics.update({"w_logistic": w_log, "w_random_forest": float(w_rf), "blend_type": "logistic+rf"})
        two_way_rows.append(metrics)

    for w_rf in np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2):
        w_svm = round(1.0 - float(w_rf), 2)
        blend_col = (w_svm * pd.to_numeric(oos["score_svm"], errors="coerce").fillna(0.0)) + (
            float(w_rf) * pd.to_numeric(oos["score_random_forest"], errors="coerce").fillna(0.0)
        )
        tmp = oos.copy()
        tmp["_blend"] = blend_col
        metrics = summarize(tmp, "_blend")
        metrics.update({"w_svm": w_svm, "w_random_forest": float(w_rf), "blend_type": "svm+rf"})
        two_way_rows.append(metrics)

    for w_rf in np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2):
        w_xgb = round(1.0 - float(w_rf), 2)
        blend_col = (w_xgb * pd.to_numeric(oos["score_xgboost"], errors="coerce").fillna(0.0)) + (
            float(w_rf) * pd.to_numeric(oos["score_random_forest"], errors="coerce").fillna(0.0)
        )
        tmp = oos.copy()
        tmp["_blend"] = blend_col
        metrics = summarize(tmp, "_blend")
        metrics.update({"w_xgboost": w_xgb, "w_random_forest": float(w_rf), "blend_type": "xgboost+rf"})
        two_way_rows.append(metrics)

    two_way = pd.DataFrame(two_way_rows)
    two_way = add_deltas(two_way, baseline)

    three_way_rows: list[dict] = []
    # 3-way simplex grid at 0.1 resolution.
    for w_log in np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1):
        for w_svm in np.round(np.arange(0.0, 1.0 - float(w_log) + 1e-9, 0.1), 1):
            w_xgb = round(1.0 - float(w_log) - float(w_svm), 1)
            if w_xgb < -1e-9:
                continue
            w_xgb = max(0.0, w_xgb)
            blend_col = (
                float(w_log) * pd.to_numeric(oos["score_logistic"], errors="coerce").fillna(0.0)
                + float(w_svm) * pd.to_numeric(oos["score_svm"], errors="coerce").fillna(0.0)
                + float(w_xgb) * pd.to_numeric(oos["score_xgboost"], errors="coerce").fillna(0.0)
            )
            tmp = oos.copy()
            tmp["_blend"] = blend_col
            metrics = summarize(tmp, "_blend")
            metrics.update(
                {
                    "w_logistic": float(w_log),
                    "w_svm": float(w_svm),
                    "w_xgboost": float(w_xgb),
                    "blend_type": "logistic+svm+xgboost",
                }
            )
            three_way_rows.append(metrics)

    three_way = pd.DataFrame(three_way_rows)
    three_way = add_deltas(three_way, baseline)

    # 3-way blends that include random forest at 0.1 resolution.
    three_way_rf_rows: list[dict] = []
    for w_log in np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1):
        for w_svm in np.round(np.arange(0.0, 1.0 - float(w_log) + 1e-9, 0.1), 1):
            w_rf = round(1.0 - float(w_log) - float(w_svm), 1)
            if w_rf < -1e-9:
                continue
            w_rf = max(0.0, w_rf)
            blend_col = (
                float(w_log) * pd.to_numeric(oos["score_logistic"], errors="coerce").fillna(0.0)
                + float(w_svm) * pd.to_numeric(oos["score_svm"], errors="coerce").fillna(0.0)
                + float(w_rf) * pd.to_numeric(oos["score_random_forest"], errors="coerce").fillna(0.0)
            )
            tmp = oos.copy()
            tmp["_blend"] = blend_col
            metrics = summarize(tmp, "_blend")
            metrics.update(
                {
                    "w_logistic": float(w_log),
                    "w_svm": float(w_svm),
                    "w_random_forest": float(w_rf),
                    "blend_type": "logistic+svm+rf",
                }
            )
            three_way_rf_rows.append(metrics)

    for w_log in np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1):
        for w_xgb in np.round(np.arange(0.0, 1.0 - float(w_log) + 1e-9, 0.1), 1):
            w_rf = round(1.0 - float(w_log) - float(w_xgb), 1)
            if w_rf < -1e-9:
                continue
            w_rf = max(0.0, w_rf)
            blend_col = (
                float(w_log) * pd.to_numeric(oos["score_logistic"], errors="coerce").fillna(0.0)
                + float(w_xgb) * pd.to_numeric(oos["score_xgboost"], errors="coerce").fillna(0.0)
                + float(w_rf) * pd.to_numeric(oos["score_random_forest"], errors="coerce").fillna(0.0)
            )
            tmp = oos.copy()
            tmp["_blend"] = blend_col
            metrics = summarize(tmp, "_blend")
            metrics.update(
                {
                    "w_logistic": float(w_log),
                    "w_xgboost": float(w_xgb),
                    "w_random_forest": float(w_rf),
                    "blend_type": "logistic+xgboost+rf",
                }
            )
            three_way_rf_rows.append(metrics)

    for w_svm in np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1):
        for w_xgb in np.round(np.arange(0.0, 1.0 - float(w_svm) + 1e-9, 0.1), 1):
            w_rf = round(1.0 - float(w_svm) - float(w_xgb), 1)
            if w_rf < -1e-9:
                continue
            w_rf = max(0.0, w_rf)
            blend_col = (
                float(w_svm) * pd.to_numeric(oos["score_svm"], errors="coerce").fillna(0.0)
                + float(w_xgb) * pd.to_numeric(oos["score_xgboost"], errors="coerce").fillna(0.0)
                + float(w_rf) * pd.to_numeric(oos["score_random_forest"], errors="coerce").fillna(0.0)
            )
            tmp = oos.copy()
            tmp["_blend"] = blend_col
            metrics = summarize(tmp, "_blend")
            metrics.update(
                {
                    "w_svm": float(w_svm),
                    "w_xgboost": float(w_xgb),
                    "w_random_forest": float(w_rf),
                    "blend_type": "svm+xgboost+rf",
                }
            )
            three_way_rf_rows.append(metrics)

    if three_way_rf_rows:
        three_way = pd.concat([three_way, pd.DataFrame(three_way_rf_rows)], ignore_index=True)
        three_way = add_deltas(three_way, baseline)

    # 4-way simplex grid at 0.2 resolution for tractable sweep.
    four_way_rows: list[dict] = []
    for w_log in np.round(np.arange(0.0, 1.0 + 1e-9, 0.2), 1):
        for w_svm in np.round(np.arange(0.0, 1.0 - float(w_log) + 1e-9, 0.2), 1):
            for w_xgb in np.round(np.arange(0.0, 1.0 - float(w_log) - float(w_svm) + 1e-9, 0.2), 1):
                w_rf = round(1.0 - float(w_log) - float(w_svm) - float(w_xgb), 1)
                if w_rf < -1e-9:
                    continue
                w_rf = max(0.0, w_rf)
                blend_col = (
                    float(w_log) * pd.to_numeric(oos["score_logistic"], errors="coerce").fillna(0.0)
                    + float(w_svm) * pd.to_numeric(oos["score_svm"], errors="coerce").fillna(0.0)
                    + float(w_xgb) * pd.to_numeric(oos["score_xgboost"], errors="coerce").fillna(0.0)
                    + float(w_rf) * pd.to_numeric(oos["score_random_forest"], errors="coerce").fillna(0.0)
                )
                tmp = oos.copy()
                tmp["_blend"] = blend_col
                metrics = summarize(tmp, "_blend")
                metrics.update(
                    {
                        "w_logistic": float(w_log),
                        "w_svm": float(w_svm),
                        "w_xgboost": float(w_xgb),
                        "w_random_forest": float(w_rf),
                        "blend_type": "logistic+svm+xgboost+rf",
                    }
                )
                four_way_rows.append(metrics)

    four_way = pd.DataFrame(four_way_rows)
    if not four_way.empty:
        four_way = add_deltas(four_way, baseline)

    two_way_path = DATA_DIR / "st_score_weighted_blends_2way.csv"
    three_way_path = DATA_DIR / "st_score_weighted_blends_3way.csv"
    four_way_path = DATA_DIR / "st_score_weighted_blends_4way.csv"
    best_path = DATA_DIR / "st_score_weighted_blends_best.csv"

    two_way.sort_values(["auc_delta_vs_logistic", "win_rate_lift_delta_vs_logistic_pp"], ascending=[False, False]).to_csv(two_way_path, index=False)
    three_way.sort_values(["auc_delta_vs_logistic", "win_rate_lift_delta_vs_logistic_pp"], ascending=[False, False]).to_csv(three_way_path, index=False)
    if not four_way.empty:
        four_way.sort_values(["auc_delta_vs_logistic", "win_rate_lift_delta_vs_logistic_pp"], ascending=[False, False]).to_csv(four_way_path, index=False)

    # Candidate sets requested: keep lift gains while recovering AUC drop.
    two_way_auc_recovered = two_way[(two_way["auc_delta_vs_logistic"] >= 0.0) & (two_way["win_rate_lift_delta_vs_logistic_pp"] > 0.0)].copy()
    three_way_auc_recovered = three_way[(three_way["auc_delta_vs_logistic"] >= 0.0) & (three_way["win_rate_lift_delta_vs_logistic_pp"] > 0.0)].copy()
    four_way_auc_recovered = four_way[(four_way["auc_delta_vs_logistic"] >= 0.0) & (four_way["win_rate_lift_delta_vs_logistic_pp"] > 0.0)].copy() if not four_way.empty else pd.DataFrame()

    best_sections: list[pd.DataFrame] = []

    if not two_way_auc_recovered.empty:
        top = two_way_auc_recovered.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False]).head(10).copy()
        top["section"] = "two_way_auc_recovered_with_lift_gain"
        best_sections.append(top)

    if not three_way_auc_recovered.empty:
        top = three_way_auc_recovered.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False]).head(10).copy()
        top["section"] = "three_way_auc_recovered_with_lift_gain"
        best_sections.append(top)

    if not four_way_auc_recovered.empty:
        top = four_way_auc_recovered.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False]).head(10).copy()
        top["section"] = "four_way_auc_recovered_with_lift_gain"
        best_sections.append(top)

    top_two_way_lift = two_way.sort_values("win_rate_lift_delta_vs_logistic_pp", ascending=False).head(10).copy()
    top_two_way_lift["section"] = "top_two_way_lift"
    best_sections.append(top_two_way_lift)

    top_three_way_lift = three_way.sort_values("win_rate_lift_delta_vs_logistic_pp", ascending=False).head(10).copy()
    top_three_way_lift["section"] = "top_three_way_lift"
    best_sections.append(top_three_way_lift)

    if not four_way.empty:
        top_four_way_lift = four_way.sort_values("win_rate_lift_delta_vs_logistic_pp", ascending=False).head(10).copy()
        top_four_way_lift["section"] = "top_four_way_lift"
        best_sections.append(top_four_way_lift)

    best = pd.concat(best_sections, ignore_index=True)
    best.to_csv(best_path, index=False)

    print("WEIGHTED_BLEND_SWEEP_COMPLETE")
    print(f"Baseline logistic: auc={baseline['auc']:.6f}, rank_ic={baseline['rank_ic']:.6f}, lift={baseline['win_rate_lift_pp']:.6f}, spread={baseline['return_spread_pct']:.6f}")

    best_two_way = _prepare_top_view(
        two_way.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False]).head(5)
    )
    best_three_way = _prepare_top_view(
        three_way.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False]).head(5)
    )
    best_four_way = _prepare_top_view(
        four_way.sort_values(["win_rate_lift_delta_vs_logistic_pp", "auc_delta_vs_logistic"], ascending=[False, False]).head(5)
    ) if not four_way.empty else pd.DataFrame()

    print("\nTOP_2WAY_BY_LIFT_DELTA")
    print(best_two_way[["blend_type", "w_logistic", "w_svm", "w_xgboost", "w_random_forest", "auc_delta_vs_logistic", "rank_ic_delta_vs_logistic", "win_rate_lift_delta_vs_logistic_pp", "return_spread_delta_vs_logistic_pct"]].to_string(index=False))

    print("\nTOP_3WAY_BY_LIFT_DELTA")
    print(best_three_way[["blend_type", "w_logistic", "w_svm", "w_xgboost", "w_random_forest", "auc_delta_vs_logistic", "rank_ic_delta_vs_logistic", "win_rate_lift_delta_vs_logistic_pp", "return_spread_delta_vs_logistic_pct"]].to_string(index=False))

    if not best_four_way.empty:
        print("\nTOP_4WAY_BY_LIFT_DELTA")
        print(best_four_way[["blend_type", "w_logistic", "w_svm", "w_xgboost", "w_random_forest", "auc_delta_vs_logistic", "rank_ic_delta_vs_logistic", "win_rate_lift_delta_vs_logistic_pp", "return_spread_delta_vs_logistic_pct"]].to_string(index=False))

    print("\nAUC_RECOVERED_AND_LIFT_GAIN_COUNTS")
    print(f"2-way candidates: {len(two_way_auc_recovered)}")
    print(f"3-way candidates: {len(three_way_auc_recovered)}")
    print(f"4-way candidates: {len(four_way_auc_recovered)}")

    print(f"\nSaved: {two_way_path}")
    print(f"Saved: {three_way_path}")
    if not four_way.empty:
        print(f"Saved: {four_way_path}")
    print(f"Saved: {best_path}")


if __name__ == "__main__":
    main()
