from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stock_triggers.scripts.short_term.train_st_logistic_model import (  # noqa: E402
    ST_FAMILY_LEVELS,
    ST_NUMERIC_FEATURES,
    _build_st_score_model,
    compute_st_features,
)
from stock_triggers.scripts.short_term.train_st_svm_model import _build_st_svm_model  # noqa: E402
from stock_triggers.ui.patterns.st_score import (  # noqa: E402
    _predict_st_score_probabilities,
    _predict_st_score_probabilities_svm,
)

TARGET_PCT = 3.0
STOP_PCT = 2.0
HOLD_DAYS = 7


def _resolve_history(price_grouped: dict[str, pd.DataFrame], ticker: str) -> pd.DataFrame | None:
    clean = str(ticker).strip()
    if clean in price_grouped:
        return price_grouped[clean]
    if clean.endswith(".NS"):
        return price_grouped.get(clean[:-3])
    return price_grouped.get(clean + ".NS")


def _build_st_outcomes(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.copy()
    prices["Ticker"] = prices["Ticker"].astype(str).str.strip()
    prices = prices.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]) 
    grouped = {str(ticker): grp.sort_values("Date") for ticker, grp in prices.groupby("Ticker", sort=False)}

    rows: list[dict] = []
    for _, row in signals.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        family = str(row.get("pattern_family", "")).strip().upper()
        signal_date = pd.to_datetime(row.get("signal_date"), errors="coerce")
        entry_price = pd.to_numeric(row.get("entry_price"), errors="coerce")
        if not ticker or not family or pd.isna(signal_date) or pd.isna(entry_price) or float(entry_price) <= 0:
            continue

        history = _resolve_history(grouped, ticker)
        if history is None:
            continue

        future = history[history["Date"] > signal_date].head(HOLD_DAYS).copy()
        if len(future) < HOLD_DAYS:
            continue

        entry = float(entry_price)
        target = entry * (1.0 + TARGET_PCT / 100.0)
        stop = entry * (1.0 - STOP_PCT / 100.0)

        status = "holding"
        hit_target = 0
        for _, bar in future.iterrows():
            low = pd.to_numeric(bar.get("Low"), errors="coerce")
            high = pd.to_numeric(bar.get("High"), errors="coerce")
            # Match ST training precedence: stop check first.
            if pd.notna(low) and float(low) <= stop:
                status = "stop"
                hit_target = 0
                break
            if pd.notna(high) and float(high) >= target:
                status = "target"
                hit_target = 1
                break

        if status == "target":
            ret = TARGET_PCT
        elif status == "stop":
            ret = -STOP_PCT
        else:
            close_7 = pd.to_numeric(future.iloc[-1].get("Close"), errors="coerce")
            ret = float(((float(close_7) / entry) - 1.0) * 100.0) if pd.notna(close_7) else np.nan

        rows.append(
            {
                "ticker": ticker,
                "signal_date": signal_date.date().isoformat(),
                "pattern_family": family,
                "st_hit_target_7d": int(hit_target),
                "status": status,
                "return_pct": float(ret) if pd.notna(ret) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _rank_auc(scores: pd.Series, labels: pd.Series) -> float:
    working = pd.DataFrame({"score": pd.to_numeric(scores, errors="coerce"), "label": pd.to_numeric(labels, errors="coerce")}).dropna()
    working = working[working["label"].isin([0, 1])]
    if working.empty:
        return float("nan")
    n_pos = int((working["label"] == 1).sum())
    n_neg = int((working["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = working["score"].rank(method="average")
    sum_pos = float(ranks[working["label"] == 1].sum())
    return float((sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg))


def _spearman_rank(left: pd.Series, right: pd.Series) -> float:
    working = pd.DataFrame({"left": pd.to_numeric(left, errors="coerce"), "right": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(working) < 3:
        return float("nan")
    return float(working["left"].rank(method="average").corr(working["right"].rank(method="average")))


def _summarize(df: pd.DataFrame, score_col: str) -> dict:
    working = df.copy()
    working["_score"] = pd.to_numeric(working.get(score_col), errors="coerce")
    working["_return"] = pd.to_numeric(working.get("return_pct"), errors="coerce")
    working = working[working["_score"].notna()].copy()

    closed = working[working["status"].isin(["target", "stop"])].copy()
    closed["_target_hit"] = (closed["status"] == "target").astype(int)

    rank_ic = _spearman_rank(working["_score"], working["_return"])
    auc = _rank_auc(closed["_score"], closed["_target_hit"])

    q_low = float(working["_score"].quantile(0.2))
    q_high = float(working["_score"].quantile(0.8))

    top_all = working[working["_score"] >= q_high]
    bottom_all = working[working["_score"] <= q_low]
    top_closed = closed[closed["_score"] >= q_high]
    bottom_closed = closed[closed["_score"] <= q_low]

    top_wr = float(top_closed["_target_hit"].mean() * 100.0) if not top_closed.empty else float("nan")
    bottom_wr = float(bottom_closed["_target_hit"].mean() * 100.0) if not bottom_closed.empty else float("nan")
    lift = float(top_wr - bottom_wr) if not (pd.isna(top_wr) or pd.isna(bottom_wr)) else float("nan")

    top_ret = float(pd.to_numeric(top_all["_return"], errors="coerce").mean()) if not top_all.empty else float("nan")
    bottom_ret = float(pd.to_numeric(bottom_all["_return"], errors="coerce").mean()) if not bottom_all.empty else float("nan")
    spread = float(top_ret - bottom_ret) if not (pd.isna(top_ret) or pd.isna(bottom_ret)) else float("nan")

    return {
        "n_scored": int(len(working)),
        "n_closed_scored": int(len(closed)),
        "auc": auc,
        "rank_ic": rank_ic,
        "win_rate_lift_pp": lift,
        "return_spread_pct": spread,
    }


def _build_xgb_matrix(df: pd.DataFrame, numeric_features: list[str], family_levels: list[str], medians: dict[str, float] | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    x = pd.DataFrame(index=df.index)
    med = {} if medians is None else dict(medians)
    for feature in numeric_features:
        series = pd.to_numeric(df.get(feature), errors="coerce")
        if medians is None:
            median = float(series.median()) if series.notna().any() else 0.0
            med[feature] = median
        else:
            median = float(med.get(feature, 0.0))
        x[feature] = series.fillna(median)

    families = df.get("pattern_family", pd.Series("", index=df.index)).astype(str).str.upper().str.strip()
    for level in family_levels:
        x[f"fam_{level}"] = (families == level).astype(float)

    return x.astype("float64"), med


def main() -> None:
    signals = pd.read_csv(ROOT / "stock_triggers" / "data" / "st_lt_training_signals_history.csv")
    prices = pd.read_csv(ROOT / "stock_triggers" / "data" / "st_lt_prices_eod.csv", parse_dates=["Date"])

    signals["signal_date"] = pd.to_datetime(signals["signal_date"], errors="coerce")
    signals = signals.dropna(subset=["signal_date", "ticker", "pattern_family", "entry_price"]).copy()
    signals["ticker"] = signals["ticker"].astype(str).str.strip()
    signals["pattern_family"] = signals["pattern_family"].astype(str).str.strip().str.upper()
    signals["entry_price"] = pd.to_numeric(signals["entry_price"], errors="coerce")
    signals = signals[signals["entry_price"] > 0].copy()

    outcomes = _build_st_outcomes(signals, prices)
    features = compute_st_features(signals, prices)
    features["signal_date"] = pd.to_datetime(features["signal_date"], errors="coerce").dt.date.astype("string")

    train = features.merge(outcomes, on=["ticker", "signal_date", "pattern_family"], how="inner")
    train["signal_date_dt"] = pd.to_datetime(train["signal_date"], errors="coerce")
    train = train.dropna(subset=["signal_date_dt"]).copy()
    train["month"] = train["signal_date_dt"].dt.to_period("M").astype(str)

    numeric_features = [feature for feature in ST_NUMERIC_FEATURES if feature in train.columns]
    family_levels = list(ST_FAMILY_LEVELS)

    months = sorted(train["month"].dropna().unique().tolist())

    fold_results: list[pd.DataFrame] = []
    folds_used = 0
    for i, test_month in enumerate(months):
        train_months = months[:i]
        if len(train_months) < 6:
            continue

        tr = train[train["month"].isin(train_months)].copy()
        te = train[train["month"] == test_month].copy()
        if tr.empty or te.empty:
            continue

        y_tr = pd.to_numeric(tr["st_hit_target_7d"], errors="coerce").fillna(0).astype(int)
        if y_tr.nunique() < 2:
            continue

        logistic_model = _build_st_score_model(tr, numeric_features=numeric_features, family_levels=family_levels)
        p_log = _predict_st_score_probabilities(te, logistic_model)

        svm_model = _build_st_svm_model(tr, numeric_features=numeric_features, family_levels=family_levels)
        p_svm = _predict_st_score_probabilities_svm(te, svm_model)

        x_tr, medians = _build_xgb_matrix(tr, numeric_features, family_levels, medians=None)
        x_te, _ = _build_xgb_matrix(te, numeric_features, family_levels, medians=medians)
        pos = int((y_tr == 1).sum())
        neg = int((y_tr == 0).sum())
        scale_pos_weight = float(neg / max(pos, 1))

        xgb = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
        )
        xgb.fit(x_tr, y_tr)
        p_xgb = xgb.predict_proba(x_te)[:, 1]

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=8,
            min_samples_split=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=4,
        )
        rf.fit(x_tr, y_tr)
        p_rf = rf.predict_proba(x_te)[:, 1]

        fold = te[["month", "ticker", "signal_date", "status", "return_pct"]].copy()
        fold["score_logistic"] = p_log * 100.0
        fold["score_svm"] = p_svm * 100.0
        fold["score_xgboost"] = p_xgb * 100.0
        fold["score_random_forest"] = p_rf * 100.0
        fold["score_logistic_svm"] = ((p_log + p_svm) / 2.0) * 100.0
        fold["score_logistic_rf"] = ((p_log + p_rf) / 2.0) * 100.0
        fold["score_svm_rf"] = ((p_svm + p_rf) / 2.0) * 100.0
        fold["score_xgboost_rf"] = ((p_xgb + p_rf) / 2.0) * 100.0
        fold["score_logistic_svm_xgboost"] = ((p_log + p_svm + p_xgb) / 3.0) * 100.0
        fold["score_logistic_svm_rf"] = ((p_log + p_svm + p_rf) / 3.0) * 100.0
        fold["score_logistic_xgboost_rf"] = ((p_log + p_xgb + p_rf) / 3.0) * 100.0
        fold["score_svm_xgboost_rf"] = ((p_svm + p_xgb + p_rf) / 3.0) * 100.0
        fold["score_logistic_svm_xgboost_rf"] = ((p_log + p_svm + p_xgb + p_rf) / 4.0) * 100.0
        fold_results.append(fold)
        folds_used += 1

    if not fold_results:
        raise SystemExit("No usable walk-forward folds were produced.")

    oos = pd.concat(fold_results, ignore_index=True)

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
    for mode, score_col in modes.items():
        summary = _summarize(oos, score_col)
        summary["mode"] = mode
        summary["folds_used"] = folds_used
        summary["months"] = int(oos["month"].nunique())
        rows.append(summary)

    summary_df = pd.DataFrame(rows)
    baseline = summary_df.loc[summary_df["mode"] == "logistic"].iloc[0]
    for metric, delta_col in [
        ("auc", "auc_delta_vs_logistic"),
        ("rank_ic", "rank_ic_delta_vs_logistic"),
        ("win_rate_lift_pp", "win_rate_lift_delta_vs_logistic_pp"),
        ("return_spread_pct", "return_spread_delta_vs_logistic_pct"),
    ]:
        summary_df[delta_col] = summary_df[metric] - float(baseline[metric])

    summary_path = ROOT / "stock_triggers" / "data" / "st_score_walk_forward_summary_with_xgboost.csv"
    oos_path = ROOT / "stock_triggers" / "data" / "st_score_walk_forward_oos_with_xgboost.csv"
    summary_df.to_csv(summary_path, index=False)
    oos.to_csv(oos_path, index=False)

    print("WALK_FORWARD_COMPARISON_WITH_XGBOOST")
    print(
        summary_df[
            [
                "mode",
                "folds_used",
                "months",
                "n_scored",
                "n_closed_scored",
                "auc",
                "rank_ic",
                "win_rate_lift_pp",
                "return_spread_pct",
                "auc_delta_vs_logistic",
                "rank_ic_delta_vs_logistic",
                "win_rate_lift_delta_vs_logistic_pp",
                "return_spread_delta_vs_logistic_pct",
            ]
        ].to_string(index=False)
    )
    print(f"Saved: {summary_path}")
    print(f"Saved: {oos_path}")


if __name__ == "__main__":
    main()
