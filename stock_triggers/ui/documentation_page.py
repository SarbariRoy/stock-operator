from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
_CHART_DIR = _DOCS_DIR / "assets" / "pattern-charts"
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_PATTERN_CHART_FILES = {
    "A": "pattern-a-breakout.svg",
    "B": "pattern-b-pullback.svg",
    "C": "pattern-c-macd.svg",
    "D": "pattern-d-rsi.svg",
    "E": "pattern-e-squeeze.svg",
    "F": "pattern-f-vwap.svg",
    "G": "pattern-g-vcp.svg",
}

_PATTERN_CHART_CAPTIONS = {
    "A": "ADANIPOWER · 2025-09-19 — uptrend breakout above 40-day high close with strong volume",
    "B": "BHARTIARTL · 2025-12-30 — pullback rebound near SMA20 inside ongoing uptrend",
    "C": "ONGC · 2026-03-27 — MACD bullish crossover while SMA50 > SMA200",
    "D": "PIDILITIND · 2025-08-06 — RSI oversold bounce with price confirming above prior day high",
    "E": "BEL · 2025-06-20 — Bollinger Band squeeze then breakout above upper band",
    "F": "COALINDIA · 2026-03-04 — VWAP reclaim on volume after brief dip below rolling average",
    "G": "BRITANNIA · 2024-09-12 — VCP contraction with shallower pullbacks then breakout above resistance",
}

# Schematic OHLC data for each candle enhancer diagram.
# Each entry: {"candles": [(label, O, H, L, C), ...], "n_signal": int, "caption": str}
# Context candles (declining) precede the signal candle(s) to show the setup.
_ENHANCER_CANDLE_DATA: dict[str, dict] = {
    "enhancer_hammer": {
        "candles": [
            ("D1", 116, 117, 112, 113),
            ("D2", 113, 114, 109, 110),
            ("D3", 110, 111, 106, 107),
            ("D4", 105, 106, 100, 105.5),
        ],
        "n_signal": 1,
        "caption": "Small body near top of range · lower shadow ≥ 50% of total range · minimal upper shadow.",
    },
    "enhancer_confirmed_hammer_a": {
        "candles": [
            ("D1", 117, 118, 113, 114),
            ("D2", 114, 115, 110, 111),
            ("D3", 111, 112, 107, 108),
            ("D4", 106, 107, 100, 106.5),
        ],
        "n_signal": 1,
        "caption": "Hammer geometry plus ≥ 2 confirming factors: RSI oversold, price near support, or above-average volume.",
    },
    "enhancer_engulfing_trend_combo": {
        "candles": [
            ("D1", 119, 120, 115, 116),
            ("D2", 116, 117, 112, 113),
            ("D3", 113, 114, 109, 110),
            ("D4", 109.5, 116.5, 109, 116),
        ],
        "n_signal": 2,
        "caption": "Live engulfing shape, but the extra combo bonus only applies when the signal family is A, C, or G.",
    },
    "enhancer_piercing_variant_b_combo": {
        "candles": [
            ("D1", 118, 119, 113, 114),
            ("D2", 114, 115, 109, 110),
            ("D3", 111, 112, 105, 106),
            ("D4", 106, 110.5, 105.5, 108.8),
        ],
        "n_signal": 2,
        "caption": "Practical piercing variant, but the extra combo bonus only applies when the signal family is B.",
    },
    "enhancer_inverted_hammer": {
        "candles": [
            ("D1", 116, 117, 112, 113),
            ("D2", 113, 114, 109, 110),
            ("D3", 110, 111, 106, 107),
            ("D4", 104, 110, 103.5, 104.5),
        ],
        "n_signal": 1,
        "caption": "Small body near the bottom · upper shadow ≥ 45% of range · minimal lower shadow.",
    },
    "enhancer_doji": {
        "candles": [
            ("D1", 116, 117, 112, 113),
            ("D2", 113, 114, 109, 110),
            ("D3", 110, 111, 106, 107),
            ("D4", 107, 107.3, 102, 107),
        ],
        "n_signal": 1,
        "caption": "Dragonfly Doji — open ≈ close near session high · long lower shadow (T-shape) · body ≤ 30% of range.",
    },
    "enhancer_marubozu": {
        "candles": [
            ("D1", 118, 119, 114, 115),
            ("D2", 115, 116, 111, 112),
            ("D3", 112, 113, 108, 109),
            ("D4", 103, 111.5, 102.5, 111),
        ],
        "n_signal": 1,
        "caption": "Large green body ≥ 80% of range · shadows ≤ 10% each — near-shadowless demand candle.",
    },
    "enhancer_belt_hold": {
        "candles": [
            ("D1", 116, 117, 112, 113),
            ("D2", 113, 114, 109, 110),
            ("D3", 115, 116, 108, 109),
            ("D4", 108, 116.5, 108, 116),
        ],
        "n_signal": 2,
        "caption": "Requires prior red candle. Green opens at session low (lower shadow ≤ 5%) · body ≥ 75% of range.",
    },
    "enhancer_engulfing": {
        "candles": [
            ("D1", 116, 117, 112, 113),
            ("D2", 113, 114, 109, 110),
            ("D3", 111, 112, 108, 109),
            ("D4", 108.5, 115, 107.5, 114),
        ],
        "n_signal": 2,
        "caption": "Green opens at or below prior close · closes at or above prior open — body fully swallows the prior red candle.",
    },
    "enhancer_harami": {
        "candles": [
            ("D1", 116, 117, 112, 113),
            ("D2", 113, 114, 109, 110),
            ("D3", 118, 119, 108, 109),
            ("D4", 110.5, 115, 110, 114),
        ],
        "n_signal": 2,
        "caption": "Small green body sits entirely inside the prior large red candle — indecision halting the down-move.",
    },
    "enhancer_morning_star": {
        "candles": [
            ("D1", 113, 114, 109, 110),
            ("D2", 116, 117, 108, 109),
            ("D3", 108.5, 110, 107, 108),
            ("D4", 109, 118, 108.5, 117),
        ],
        "n_signal": 3,
        "caption": "Three-bar reversal: large red → small star (body < 30%) → large green closing above midpoint of bar 1.",
    },
    "enhancer_piercing": {
        "candles": [
            ("D1", 118, 119, 114, 115),
            ("D2", 113, 114, 109, 110),
            ("D3", 116, 117, 108, 109),
            ("D4", 107, 114, 106.5, 113.5),
        ],
        "n_signal": 2,
        "caption": "Green opens below prior session low (gap down) · closes above midpoint of the prior red body.",
    },
    "enhancer_piercing_variant": {
        "candles": [
            ("D1", 118, 119, 114, 115),
            ("D2", 113, 114, 109, 110),
            ("D3", 116, 117, 108, 109),
            ("D4", 110.5, 115.5, 109.5, 113.5),
        ],
        "n_signal": 2,
        "caption": "Cash-market relaxation: green opens within 3% above prior close (no gap required) · still closes above midpoint.",
    },
    "enhancer_three_white_soldiers": {
        "candles": [
            ("D1", 112, 113, 108, 109),
            ("D2", 109, 110, 106, 107),
            ("D3", 104, 108.5, 103.5, 108),
            ("D4", 107, 113, 106.5, 112.5),
            ("D5", 111, 117, 110.5, 116.5),
        ],
        "n_signal": 3,
        "caption": "Three consecutive green candles · each body ≥ 50% · progressively higher closes and highs · total gain ≥ 4%.",
    },
}


SECTION_COPY: dict[str, dict[str, str]] = {
    "overview": {
        "title": "Overview",
        "intro": (
            "This app is a review workspace for stock signals. Tomorrow's Picks is the fast shortlist for what to inspect next. "
            "Backtesting Lab is the validation surface for checking how the same ideas behaved historically."
        ),
    },
    "tomorrow": {
        "title": "Tomorrow's Picks",
        "intro": (
            "This page is optimized for quick daily review. It helps you rank candidate signals, inspect one stock in detail, "
            "and see the reasons a row looks strong or weak before you move into a deeper decision workflow."
        ),
    },
    "scores": {
        "title": "Scores And Filters",
        "intro": (
            "The app carries more than one score because they answer different questions. Some scores are heuristic and component-based, "
            "while others summarize historical behavior or estimated stop risk."
        ),
    },
    "patterns": {
        "title": "Pattern Families",
        "intro": (
            "Pattern families A through G represent different setup types. The family label tells you what kind of structure triggered the row, "
            "while the detailed pattern name tells you the specific rule variant."
        ),
    },
    "lab": {
        "title": "Backtesting Lab",
        "intro": (
            "Backtesting Lab is where you stress-test filters, stops, targets, and score behavior. It is designed for evaluating recorded signals, "
            "not for placing trades automatically."
        ),
    },
    "trade_records": {
        "title": "Trade Records",
        "intro": (
            "Trade Records is the main drill-down table in Backtesting Lab. It shows one simulated trade row per signal after the current filters, "
            "including entry, stop, status, returns, and score components."
        ),
    },
    "manual_positions": {
        "title": "Manual Positions",
        "intro": (
            "Manual positions let you track trades or ideas that you want to monitor in the same workspace even if they were not generated by the current daily signal run."
        ),
    },
    "past_results": {
        "title": "Past Results",
        "intro": (
            "Past Results is the quick historical snapshot for a single stock on Tomorrow's Picks. It evaluates a small tail of earlier signals using a simple hold window so you can see recent behavior quickly."
        ),
    },
    "scoring_formula": {
        "title": "Score Formula",
        "intro": (
            "The signal score is a transparent, weighted combination of five independent quality measures. "
            "Understanding how each component is built helps you interpret where a score comes from and why it changes. "
            "After the base weighted sum, two additive bonuses can raise the total above the raw component floor."
        ),
    },
    "enhancers": {
        "title": "Candle Enhancers",
        "intro": (
            "Candle enhancers are optional single-bar or two-bar shape patterns that add a small bonus to the base signal score "
            "when a recognizable candlestick formation appears near the signal date. "
            "The combined enhancer bonus is capped so candle overlays cannot dominate the overall score. "
            "Each enhancer fires independently — two reinforcing shapes on the same bar compound the bonus up to the cap."
        ),
    },
    "workflow": {
        "title": "Daily Workflow",
        "intro": (
            "The engine runs a sequential pipeline every day: update prices, scan all patterns, recompute family weights, "
            "and optionally rebuild the stop risk model. Understanding the pipeline helps you know which data files are "
            "authoritative, when outputs are stale, and which scripts to run if something needs refreshing manually."
        ),
    },
}


HELP_ITEMS: dict[str, dict[str, str]] = {
    "overview": {
        "section": "overview",
        "label": "What this app does",
        "summary": "A quick explanation of the two main pages and how they fit together.",
        "detail": (
            "Use **Tomorrow's Picks** to decide what deserves chart review next. Use **Backtesting Lab** to test how similar signals behaved under a chosen target, stop, and filter set."
        ),
    },
    "tomorrow_picks": {
        "section": "tomorrow",
        "label": "Tomorrow's Picks",
        "summary": "The daily shortlist of candidate signals to review for the next session.",
        "detail": (
            "Tomorrow's Picks ranks the currently relevant signals, applies the selected score method, and lets you inspect one row in detail. "
            "It is meant to reduce the number of names you need to review manually, not replace the review itself."
        ),
    },
    "scoring_method": {
        "section": "scores",
        "label": "Lens",
        "summary": "Sets the ranking lens that drives Tomorrow's Picks.",
        "detail": (
            "The selected lens controls the main ranking signal in Tomorrow's Picks. Use **Heuristic score** for the model-driven setup view, "
            "**Reliability score** for a more empirical confidence read, and **Stop risk** when you want downside distance to lead the board."
        ),
    },
    "heuristic_score": {
        "section": "scores",
        "label": "Heuristic score",
        "summary": "A composite quality score built from setup, trend, volume, risk, and supporting bonuses.",
        "detail": (
            "Heuristic score is a weighted summary of setup quality. Higher usually means the setup aligns better with the scoring model, but it does not guarantee a profitable trade."
        ),
    },
    "reliability_score": {
        "section": "scores",
        "label": "Reliability score",
        "summary": "A score intended to reflect how trustworthy the signal looks based on learned behavior.",
        "detail": (
            "Reliability score is useful when you want the ranking to emphasize historical consistency more than raw pattern shape. Treat it as a confidence hint, not a promise."
        ),
    },
    "stop_risk": {
        "section": "scores",
        "label": "Stop risk",
        "summary": "An estimate of downside distance between the entry and the stop.",
        "detail": (
            "Stop risk is shown as a percentage-like risk measure. Lower is generally better because it implies a tighter stop relative to the entry."
        ),
    },
    "min_score_filter": {
        "section": "scores",
        "label": "Cutoff",
        "summary": "Sets the threshold used to trim the shortlist.",
        "detail": (
            "In Tomorrow's Picks, the cutoff keeps only rows that clear the active ranking threshold. In Backtesting Lab, the same control trims the simulated trade set before you evaluate results."
        ),
    },
    "sort_order": {
        "section": "scores",
        "label": "Order",
        "summary": "Sets how the visible shortlist is ordered after filtering.",
        "detail": (
            "Use order to decide whether the shortlist leads with score, lowest risk, or a simple alphabetical view. Sorting changes the presentation only, not the underlying rows."
        ),
    },
    "score_breakdown": {
        "section": "scores",
        "label": "Score breakdown",
        "summary": "The running explanation of why a selected signal earned its score.",
        "detail": (
            "Score breakdown is meant to answer the question: **why is this row scoring where it is?** It decomposes the row into setup, trend, volume, risk, RSI, and any learned bonuses."
        ),
    },
    "overview_metrics": {
        "section": "scores",
        "label": "Overview metrics",
        "summary": "The compact entry, stop, risk, and score summary for the selected signal.",
        "detail": (
            "These are the first numbers to sanity-check on a selected signal. Entry and stop frame the trade. Risk shows the downside distance. The displayed score depends on the currently active ranking method."
        ),
    },
    "entry_price": {
        "section": "scores",
        "label": "Entry price",
        "summary": "The reference buy price used for the signal or simulated trade.",
        "detail": (
            "Entry price is the price used as the starting point for risk and return calculations. In the lab, it feeds position sizing, target price, and PnL."
        ),
    },
    "stop_price": {
        "section": "scores",
        "label": "Stop price",
        "summary": "The protective exit level used to cap downside risk.",
        "detail": (
            "Stop price is the level where the trade is considered invalid or risk control takes over. It is central to both trade risk and stop-hit status."
        ),
    },
    "trade_risk": {
        "section": "scores",
        "label": "Risk",
        "summary": "The percentage distance from entry price to stop price.",
        "detail": (
            "Risk is typically the percent gap between entry and stop. Wider risk may demand smaller position size, while tighter risk can improve capital efficiency if the setup quality still holds."
        ),
    },
    "quick_check": {
        "section": "tomorrow",
        "label": "Quick check",
        "summary": "A fast checklist for trend, breakout posture, volume, stretch, and stop width.",
        "detail": (
            "Quick check is a simplified quality screen for the selected stock. It answers a few high-value questions quickly before you decide whether the chart deserves more attention."
        ),
    },
    "status": {
        "section": "trade_records",
        "label": "Status",
        "summary": "The current or final outcome label for the simulated trade row.",
        "detail": (
            "Typical status values distinguish rows that hit target, hit stop, are still holding, or were tracked under a specific outcome rule. Always interpret status alongside exit date and return."
        ),
    },
    "past_results": {
        "section": "past_results",
        "label": "Past results",
        "summary": "A recent-history check for earlier signals on the same stock.",
        "detail": (
            "Past Results uses a simple evaluation window over recent historical rows for the selected stock. It is not the full Backtesting Lab, but a quick way to see whether the name has behaved cleanly lately."
        ),
    },
    "hold_days": {
        "section": "past_results",
        "label": "Hold days",
        "summary": "How many forward trading days the simple past-results evaluator allows before timing out.",
        "detail": (
            "Hold days controls the evaluation window in the quick Past Results view. Larger values give trades more time to work, while smaller values force a tighter review horizon."
        ),
    },
    "outcome": {
        "section": "past_results",
        "label": "Outcome",
        "summary": "The summarized result of a historical signal after applying the simple evaluation rule.",
        "detail": (
            "Outcome translates the raw evaluation result into plain language such as stopped out, held to end, timed exit, or no future data."
        ),
    },
    "return_pct": {
        "section": "trade_records",
        "label": "Return %",
        "summary": "Percentage return on the simulated trade from entry to current or exit value.",
        "detail": (
            "Return % shows the percent gain or loss produced by the trade row. In open trades it may be marked to the latest close, while in closed trades it reflects the exit outcome."
        ),
    },
    "exit_date": {
        "section": "trade_records",
        "label": "Exit date",
        "summary": "The date the simulated trade closed or reached its evaluated end state.",
        "detail": (
            "Exit date is when the tracker considered the row finished. If the trade is still open, exit date may be blank."
        ),
    },
    "learned_pattern_weights": {
        "section": "patterns",
        "label": "Learned pattern weights",
        "summary": "Historical weights assigned to pattern families based on past signal behavior.",
        "detail": (
            "Learned pattern weights describe how much each pattern family contributes as a bonus or penalty in the scoring system. They come from historical signal behavior and should be treated as model inputs, not standalone signals."
        ),
    },
    "pattern_family": {
        "section": "patterns",
        "label": "Pattern family",
        "summary": "The high-level setup family, such as A through G.",
        "detail": (
            "Pattern family groups signals by setup archetype. The family tells you the broad structure, while the detailed pattern field tells you the specific rule that fired."
        ),
    },
    "pattern": {
        "section": "patterns",
        "label": "Pattern",
        "summary": "The exact rule variant that generated the signal.",
        "detail": (
            "Pattern is the fine-grained label for the trigger logic. It lets you separate broad family behavior from a specific setup flavor."
        ),
    },
    "pattern_score_100": {
        "section": "patterns",
        "label": "Score /100",
        "summary": "A normalized pattern-quality score used in the learned pattern-weight table.",
        "detail": (
            "Score /100 summarizes historical pattern quality on a 0 to 100 scale. It is one input into the final pattern-family weight rather than a live trade score by itself."
        ),
    },
    "pattern_weight_30": {
        "section": "patterns",
        "label": "Weight /30",
        "summary": "The effective bonus or penalty budget allocated to that pattern family.",
        "detail": (
            "Weight /30 is the pattern-family contribution applied in scoring. Larger positive values mean the family earned more trust from the training history."
        ),
    },
    "count": {
        "section": "patterns",
        "label": "Count",
        "summary": "How many historical rows contributed to the learned statistic.",
        "detail": (
            "Count matters because a good-looking edge from very few samples is weaker evidence than a similar edge from a deeper history."
        ),
    },
    "win_rate": {
        "section": "patterns",
        "label": "Win rate",
        "summary": "The share of qualifying trades that ended as wins under the chosen historical definition.",
        "detail": (
            "Win rate is useful for context, but it should be read together with sample count and average returns rather than in isolation."
        ),
    },
    "loss_rate": {
        "section": "patterns",
        "label": "Loss rate",
        "summary": "The share of qualifying trades that ended as losses.",
        "detail": (
            "Loss rate complements win rate and helps you judge whether an attractive pattern score is backed by balanced historical behavior."
        ),
    },
    "edge_pp": {
        "section": "patterns",
        "label": "Edge pp",
        "summary": "The win-rate edge measured in percentage points over baseline.",
        "detail": (
            "Edge pp shows how far a pattern's observed behavior sits above or below a baseline. Positive values suggest historical improvement; negative values suggest drag."
        ),
    },
    "analysis_setup": {
        "section": "lab",
        "label": "Analysis Setup",
        "summary": "The section where you choose scope, stops, targets, and visible record filters for the lab.",
        "detail": (
            "Analysis Setup defines the simulated environment before you study results. The chosen signal families, stop rules, targets, and visible filters all change the trade set you evaluate."
        ),
    },
    "target_pct": {
        "section": "lab",
        "label": "Target %",
        "summary": "The profit target used by the lab simulation.",
        "detail": (
            "Target % defines how much upside the tracker waits for before calling a trade a target hit. Tight targets usually increase hit frequency and reduce per-trade payoff."
        ),
    },
    "stop_mode": {
        "section": "lab",
        "label": "Stop mode",
        "summary": "The rule used to derive or interpret the stop in the lab simulation.",
        "detail": (
            "Stop mode changes how the stop is set or enforced. Some modes use a fixed percentage, others use ATR or structure, and some high-score modes hold to target while keeping the stop only for risk display."
        ),
    },
    "capital_per_trade": {
        "section": "lab",
        "label": "Capital per trade",
        "summary": "The notional capital allocated to each simulated trade row.",
        "detail": (
            "Capital per trade determines the simulated position size in rupee terms. It affects invested amount, quantity, current value, and PnL."
        ),
    },
    "model_reference": {
        "section": "lab",
        "label": "Model and reference controls",
        "summary": "Settings that change which historical rows are evaluated or how reference artifacts are read.",
        "detail": (
            "These controls decide how the lab should interpret saved signal history and evaluation windows. They are for validation and comparison, not for changing the saved production artifacts from inside the UI."
        ),
    },
    "summary_kpis": {
        "section": "lab",
        "label": "Summary KPIs",
        "summary": "The headline counts and returns for the currently visible trade set.",
        "detail": (
            "Summary KPIs are the first stop in Backtesting Lab. Read them before drilling into individual rows so you know whether the current filter set is improving or weakening the overall profile."
        ),
    },
    "trade_records": {
        "section": "trade_records",
        "label": "Trade Records",
        "summary": "The main simulated-trade table for the current lab configuration.",
        "detail": (
            "Trade Records shows the row-level evidence behind the KPI layer. It is where you inspect single trades, compare exits, and verify that a score threshold or stop rule is behaving the way you expect."
        ),
    },
    "qty": {
        "section": "trade_records",
        "label": "Quantity",
        "summary": "The number of shares or units allocated by the capital-per-trade rule.",
        "detail": (
            "Quantity is derived from the entry price and the configured capital per trade. It lets the tracker convert percentage outcomes into invested value and PnL."
        ),
    },
    "invested": {
        "section": "trade_records",
        "label": "Invested",
        "summary": "The actual notional capital committed to that simulated trade row.",
        "detail": (
            "Invested is the position value at entry after quantity is calculated. It may differ slightly from the raw capital-per-trade input because quantity is discrete."
        ),
    },
    "target_price": {
        "section": "trade_records",
        "label": "Target price",
        "summary": "The price level corresponding to the chosen target percentage.",
        "detail": (
            "Target price is the level that marks a target hit under the active target rule."
        ),
    },
    "latest_close": {
        "section": "trade_records",
        "label": "Latest close",
        "summary": "The latest available close used to mark open positions.",
        "detail": (
            "Latest close is used to estimate current value and open-trade return when a row has not yet reached an exit condition."
        ),
    },
    "current_value": {
        "section": "trade_records",
        "label": "Current value",
        "summary": "The marked-to-market value of the simulated position.",
        "detail": (
            "Current value reflects what the position would be worth using the latest available close. For closed trades it should align with the exit state."
        ),
    },
    "pnl": {
        "section": "trade_records",
        "label": "PnL",
        "summary": "Profit or loss in currency terms for the simulated trade row.",
        "detail": (
            "PnL converts the trade outcome into currency terms using invested value and quantity. It is useful when comparing economic impact rather than only percentage performance."
        ),
    },
    "days_held": {
        "section": "trade_records",
        "label": "Days held",
        "summary": "The number of trading days the simulated trade remained active.",
        "detail": (
            "Days held helps you see whether a setup is working quickly or tying up capital for a long time."
        ),
    },
    "score_pattern": {
        "section": "trade_records",
        "label": "Pattern score",
        "summary": "The pattern-specific contribution to the signal score.",
        "detail": (
            "Pattern score reflects how the triggering family or setup contributes to the overall scoring model."
        ),
    },
    "pattern_bonus": {
        "section": "trade_records",
        "label": "Pattern bonus",
        "summary": "The learned family-level bonus or penalty added to the score.",
        "detail": (
            "Pattern bonus comes from the learned pattern-weight artifact. It adjusts the score based on how that family has behaved historically."
        ),
    },
    "stock_rs20": {
        "section": "trade_records",
        "label": "Stock RS20",
        "summary": "The stock's relative strength reading versus the benchmark over roughly 20 trading days.",
        "detail": (
            "RS20 helps show whether the stock is outperforming the benchmark over a shorter window. Stronger relative strength can support trend-following setups."
        ),
    },
    "stock_rs50": {
        "section": "trade_records",
        "label": "Stock RS50",
        "summary": "The stock's relative strength reading versus the benchmark over roughly 50 trading days.",
        "detail": (
            "RS50 is the slower relative-strength companion to RS20 and helps confirm whether a shorter burst is part of a broader outperformance trend."
        ),
    },
    "rs_bonus": {
        "section": "trade_records",
        "label": "RS bonus",
        "summary": "The score bonus added when benchmark-relative strength is enabled and supportive.",
        "detail": (
            "RS bonus is a capped overlay. It nudges the score higher when the stock shows benchmark outperformance and the feature is enabled."
        ),
    },
    "enhancer_bonus": {
        "section": "trade_records",
        "label": "Enhancer bonus",
        "summary": "The extra score assigned from candle-shape enhancer rules.",
        "detail": (
            "Enhancer bonus is the optional overlay from candle-based enhancers such as doji, hammer, or morning star. It is capped so candle overlays do not dominate the base score."
        ),
    },
    "manual_positions": {
        "section": "manual_positions",
        "label": "Manual positions",
        "summary": "The table of user-tracked positions that live alongside the auto-generated lab rows.",
        "detail": (
            "Manual positions let you add your own rows to the review workspace. Use them for discretionary trades, paper trades, or ideas you want to watch outside the generated signal set."
        ),
    },
    "created_at": {
        "section": "manual_positions",
        "label": "Created at",
        "summary": "When the manual row was added to the workspace.",
        "detail": (
            "Created at is the timestamp for when you recorded the manual position in the UI."
        ),
    },
    "source_signal_date": {
        "section": "manual_positions",
        "label": "Source signal date",
        "summary": "The original signal date associated with the row, if you know it.",
        "detail": (
            "Source signal date helps keep a manual position linked back to the original idea or recommendation date."
        ),
    },
    "capital": {
        "section": "manual_positions",
        "label": "Capital",
        "summary": "The manual notional capital assigned to that row.",
        "detail": (
            "Capital determines the notional size of the manual position for current-value and PnL calculations."
        ),
    },
    "current_return_pct": {
        "section": "manual_positions",
        "label": "Current return %",
        "summary": "The marked-to-market percentage return of the manual position.",
        "detail": (
            "Current return % shows how the manual position is performing relative to the recorded entry price."
        ),
    },
    "distance_to_stop_pct": {
        "section": "manual_positions",
        "label": "Distance to stop %",
        "summary": "How close the current price is to the stop, expressed as a percentage.",
        "detail": (
            "Distance to stop % is useful for prioritizing attention. Smaller values mean the position is closer to the protective stop."
        ),
    },
    "note": {
        "section": "manual_positions",
        "label": "Note",
        "summary": "Free-form context about the manual position.",
        "detail": (
            "Use Note for reminders, discretionary context, or anything that makes the row easier to interpret later."
        ),
    },
    # ── Scoring formula components ────────────────────────────────────────────
    "score_trend_comp": {
        "section": "scoring_formula",
        "label": "Trend component (T)",
        "summary": "Measures how strongly the stock is trending at signal time.",
        "detail": (
            "T = clip(50 + trend\\_strength\\_pct × 5). A flat trend lands around 50. Strong upward trend "
            "pushes T toward 100. This weight is **0.20** in the total score, contributing up to 20 points. "
            "The trend strength percent is derived from the relationship between current price, key moving averages, and recent slope."
        ),
    },
    "score_setup_comp": {
        "section": "scoring_formula",
        "label": "Setup component (S)",
        "summary": "Measures how cleanly the specific pattern conditions align.",
        "detail": (
            "S = clip(50 + setup\\_strength\\_pct × 8). Setup quality captures how well the trigger conditions are met — "
            "not just that they fired, but how convincingly. This weight is **0.20**, contributing up to 20 points. "
            "A fresh breakout on high-conviction conditions scores higher than a borderline trigger."
        ),
    },
    "score_volume_comp": {
        "section": "scoring_formula",
        "label": "Volume component (V)",
        "summary": "Measures how much volume exceeds the baseline average on the signal day.",
        "detail": (
            "V = clip(40 + volume\\_ratio × 20). A volume ratio of 1.0 means volume exactly matched the 20-day average. "
            "The baseline 40 ensures a modestly above-average volume day already scores reasonably. "
            "This weight is **0.13**, worth up to 13 points."
        ),
    },
    "score_risk_comp": {
        "section": "scoring_formula",
        "label": "Risk component (R)",
        "summary": "Rewards tighter stops — lower stop distance means higher risk score.",
        "detail": (
            "R = clip(100 − stop\\_pct × 6). A tighter stop distance translates directly to a higher R value. "
            "This weight is **0.14**, worth up to 14 points. R is not a measure of probability — "
            "it is purely a reward for capital efficiency. A 5% stop yields R = 70; a 10% stop yields R = 40."
        ),
    },
    "score_rsi_comp": {
        "section": "scoring_formula",
        "label": "RSI component (I)",
        "summary": "Uses RSI as a minor context signal, with the best boost in the 50-60 sweet spot.",
        "detail": (
            "I is center-favored: RSI in **50–60** gets the highest RSI sub-score. "
            "Outside that range, the RSI sub-score decays linearly toward lower values as RSI moves toward 0 or 100. "
            "Color interpretation uses: <=40 red, 40–50 yellow, 50–60 green, 60–70 yellow, >=70 red. "
            "If RSI is unavailable, a neutral fallback is used. This weight is **0.03**, "
            "so it can contribute up to about 3 points to the total score."
        ),
    },
    "ma_slope_bonus": {
        "section": "scoring_formula",
        "label": "MA slope bonus",
        "summary": "An additive bonus that rewards an accelerating 50-day moving average.",
        "detail": (
            "MA slope bonus = min(3.0, slope\\_pct × 4.0). It is computed from the percentage change in SMA50 "
            "over the last 5 trading days. A rising SMA50 adds up to **3 extra points** on top of the weighted sum. "
            "A flat or declining SMA50 adds zero. This bonus is applied after the five-component base score is assembled."
        ),
    },
    "consensus_bonus": {
        "section": "scoring_formula",
        "label": "Consensus bonus",
        "summary": "Extra credit added when two or more pattern families fire on the same stock and date.",
        "detail": (
            "When multiple pattern families trigger simultaneously on the same ticker and signal date, "
            "the engine can add a consensus bonus because independent methods agreeing on the same setup "
            "is stronger evidence than a single-method trigger. "
            "The size of the bonus depends on the number of agreeing families. "
            "This bonus compounds with the pattern-family weight and MA slope bonus."
        ),
    },
    "pattern_family_bonus_formula": {
        "section": "scoring_formula",
        "label": "Pattern family bonus (B_pattern)",
        "summary": "The learned family contribution — up to 30 points from pattern_weights.json.",
        "detail": (
            "B\\_pattern comes from the learned `pattern_weights.json` artifact. "
            "The formula is approximately: B\\_pattern = (family\\_score / 100) × 30. "
            "So a family with a historical score of 80 contributes roughly 24 extra points. "
            "A family with score 50 contributes 15, and score 0 contributes nothing. "
            "Negative-scoring families can also reduce the score. "
            "This is the largest single bonus and the primary mechanism for differentiating families."
        ),
    },
    "penalty_weights": {
        "section": "scores",
        "label": "Signal penalty weights",
        "summary": "Per-ticker penalties that reduce scores for stocks that have recently underperformed.",
        "detail": (
            "Signal penalty weights are learned per-ticker adjustments stored in `signal_penalty_weights.json`. "
            "When a stock has a history of signaling but not delivering, its penalty weight nudges its score down. "
            "Penalties are applied during the scoring phase so the final displayed score already includes them. "
            "Stocks with clean historical behavior are unaffected. This mechanism helps prevent repeated false positives "
            "from dominating the shortlist."
        ),
    },
    "oos_stop_risk": {
        "section": "scores",
        "label": "Walk-forward stop risk (OOS)",
        "summary": "An out-of-sample stop risk prediction produced by the walk-forward evaluation.",
        "detail": (
            "The walk-forward stop risk model trains on rolling monthly windows and produces out-of-sample (OOS) "
            "predictions for each test month. These predictions estimate the expected stop distance for a signal, "
            "independent of the heuristic score. "
            "In Backtesting Lab the OOS filter is applied via an inner merge, meaning only signals that fall within "
            "a month that has a corresponding walk-forward prediction will appear. This is why pre-2024 signals "
            "may not appear in the lab tracker — the walk-forward warmup period has to complete before predictions start."
        ),
    },
    # ── Candle enhancers ──────────────────────────────────────────────────────
    "enhancer_hammer": {
        "section": "enhancers",
        "label": "Hammer",
        "summary": "A single bar with a small body high up and a long lower shadow, optionally with confirmation.",
        "detail": (
            "A hammer has a small real body in the upper portion of the bar's range and a lower shadow at least twice the body length. "
            "The confirmed version requires the shape plus at least two supporting conditions: recent RSI oversold context, "
            "price near recent support, and volume above average. "
            "The shape-only version fires on geometry alone. "
            "The hammer suggests buyers absorbed selling pressure and pushed price back up by the close."
        ),
    },
    "enhancer_inverted_hammer": {
        "section": "enhancers",
        "label": "Inverted Hammer",
        "summary": "A single bar with a small body in the lower portion and a long upper shadow.",
        "detail": (
            "An inverted hammer is the upside-down counterpart to the hammer. The long upper shadow indicates buyers attempted "
            "to push price higher; the return to a lower close shows partial rejection, but in a downtrend this can mark a "
            "tentative reversal. Its enhancer bonus tends to be smaller than the confirmed hammer "
            "because the pattern on its own is weaker evidence."
        ),
    },
    "enhancer_morning_star": {
        "section": "enhancers",
        "label": "Morning Star",
        "summary": "A three-bar reversal: a red candle, a small indecision bar, then a strong green candle.",
        "detail": (
            "Morning Star is a three-candle sequence: a bearish candle, a small-body or doji bar that gaps or sits below, "
            "then a bullish candle that closes well into the first candle's body. "
            "It suggests the sell-off has exhausted and buyers are regaining control. "
            "Because it requires three aligned bars, it is rarer and generally earns a larger enhancer bonus than single-bar shapes."
        ),
    },
    "enhancer_engulfing": {
        "section": "enhancers",
        "label": "Bullish Engulfing",
        "summary": "A green candle whose real body fully engulfs the previous red candle's body.",
        "detail": (
            "Bullish engulfing requires two bars: a red (bearish) candle followed by a green (bullish) candle whose open is "
            "at or below the prior close and whose close is at or above the prior open. "
            "The green bar's body 'engulfs' the prior bar's body. "
            "It signals decisive buyer dominance after a down day and is considered a reliable two-bar reversal signal."
        ),
    },
    "enhancer_harami": {
        "section": "enhancers",
        "label": "Bullish Harami",
        "summary": "A small green candle contained entirely within the prior large red candle.",
        "detail": (
            "Bullish harami is the reverse of engulfing: a large bearish bar is followed by a smaller bullish bar "
            "whose body sits completely inside the prior bar's body. "
            "The pattern suggests the down move is losing momentum. It is generally considered a weaker signal than "
            "bullish engulfing because the small bar needs to be confirmed by subsequent price action."
        ),
    },
    "enhancer_marubozu": {
        "section": "enhancers",
        "label": "Bullish Marubozu",
        "summary": "A near-shadowless green candle showing uniform buyer dominance throughout the session.",
        "detail": (
            "A bullish marubozu opens at or near its low and closes at or near its high, with minimal upper and lower shadows. "
            "It shows that buyers controlled the entire session without giving sellers an opportunity to reassert. "
            "On a signal-day bar this is a strong confirmation that the breakout or pattern trigger had genuine conviction behind it."
        ),
    },
    "enhancer_doji": {
        "section": "enhancers",
        "label": "Dragonfly Doji",
        "summary": "A T-shaped candle with a tiny body at the top and a long lower shadow.",
        "detail": (
            "A dragonfly doji has an open and close that are nearly identical and sit near the high of the range, "
            "while the long lower shadow shows sellers pushed price down but buyers fully recovered by the close. "
            "The geometry requires the lower shadow to be at least 60% of the full range and the upper shadow to be at most 15%. "
            "As an enhancer it adds a small bonus when this shape appears in the lookback window around the signal date."
        ),
    },
    "enhancer_piercing": {
        "section": "enhancers",
        "label": "Piercing Line",
        "summary": "A two-bar reversal where a strong green candle closes above the midpoint of the prior red candle.",
        "detail": (
            "Piercing Line requires two bars: a long bearish candle (body at least 50% of range) followed by a bullish candle "
            "that opens below the prior bar's low and closes above the midpoint of the prior body, but still below the prior open. "
            "The pattern suggests buyers absorbed the full down move and pushed price back into the red body. "
            "True gap-down opens are required in the textbook version, which may make this rarer on cash-market EOD data than the variant."
        ),
    },
    "enhancer_piercing_variant": {
        "section": "enhancers",
        "label": "Piercing Variant",
        "summary": "A practical cash-market adaptation of Piercing Line that relaxes the gap-down open requirement.",
        "detail": (
            "The piercing variant is designed for markets where true gap-down opens after a bearish session are uncommon. "
            "It fires when the current green bar opens near or slightly below the prior close (within 3%) and still closes above "
            "the midpoint of the prior red body. The body-size requirements are also relaxed (25%/20% vs 50%/50%). "
            "It captures the same 'buyers took back mid-body' conviction without demanding a true gap, making it more frequent in practice."
        ),
    },
    "enhancer_belt_hold": {
        "section": "enhancers",
        "label": "Belt Hold",
        "summary": "A single green candle that opens near its low and closes near its high with almost no lower shadow.",
        "detail": (
            "A bullish belt hold opens at or very near the session low (lower shadow ≤ 5% of range) and closes at or very near the high "
            "(upper shadow ≤ 10%), with a large real body (≥ 75% of range). "
            "The standard version also requires the prior candle to be bearish so the belt hold represents a genuine reversal, "
            "not just a continuation of an existing up move. "
            "The name comes from the idea that buyers 'held' control from the open all the way through to the close."
        ),
    },
    "enhancer_three_white_soldiers": {
        "section": "enhancers",
        "label": "Three White Soldiers",
        "summary": "Three consecutive bullish candles, each closing higher, signalling a strong momentum shift.",
        "detail": (
            "Three white soldiers requires three consecutive green bars where each bar has a meaningful body (≥ 50% of range), "
            "each close is higher than the prior close, and the upper shadows are small (≤ 20% of range). "
            "The combined gain across all three bars must also exceed 4%. "
            "When both higher highs and higher closes are required, the pattern shows sustained buying pressure across three sessions. "
            "As an enhancer it provides a strong bonus, but because it requires three aligned bars it fires less often than single-bar shapes."
        ),
    },
    "enhancer_confirmed_hammer_a": {
        "section": "enhancers",
        "label": "Confirmed Hammer + Pattern A",
        "summary": "A hammer shape that coincides with a Pattern A breakout trigger on the same bar.",
        "detail": (
            "This enhancer combines the hammer geometry (small body in upper portion, lower shadow ≥ 2× body) "
            "with the Pattern A breakout trigger firing on the same signal date. "
            "The coincidence of a bullish reversal candle shape and a clean breakout above a prior reference level on the same bar "
            "is treated as a stronger signal than either alone. "
            "If the Pattern A family weight is high, this enhancer can meaningfully push the total score upward."
        ),
    },
    "enhancer_engulfing_trend_combo": {
        "section": "enhancers",
        "label": "Engulfing + A/C/G",
        "summary": "A live bullish engulfing that only earns the combo bonus when the signal family is A, C, or G.",
        "detail": (
            "Historical family analysis showed that engulfing is not helpful as a global enhancer, but it performs materially better "
            "inside the trend-and-momentum families A, C, and G. "
            "This combo therefore leaves plain engulfing available on its own while reserving an extra learned bonus for those specific families."
        ),
    },
    "enhancer_piercing_variant_b_combo": {
        "section": "enhancers",
        "label": "Piercing Variant + B",
        "summary": "A practical piercing-line recovery that only earns the combo bonus when the signal family is B.",
        "detail": (
            "Strict textbook piercing line is too rare in this end-of-day history to learn a stable positive weight. "
            "The practical piercing variant is more useful, and its best measured edge appears inside Pattern B pullback-and-rebound signals. "
            "This combo keeps the standalone variant while reserving an extra learned bonus for that specific family overlap."
        ),
    },
    # ── Per-pattern deep dives ────────────────────────────────────────────────
    "pattern_a": {
        "section": "patterns",
        "label": "Pattern A — Trend Breakout",
        "summary": "A stock already in a strong uptrend pushes above a recent high with above-average volume.",
        "detail": (
            "**Conditions:** SMA50 > SMA200, close > SMA50, close > SMA200, close above the prior N-day high close, "
            "volume exceeds the 20-day average by `volume_multiplier` (default 1.5×).\n\n"
            "**Score boosters:** Strong trend strength (T higher when close is well above both MAs), large volume ratio boosts V, "
            "tight stop boosts R, and if SMA50 slope is rising the MA bonus adds up to 3 points.\n\n"
            "**Typical use:** Best for momentum continuation in high-liquidity names that are already performing well relative to peers. "
            "Works strongest in trending market conditions.\n\n"
            "**Known weaknesses:** Breakouts can fail quickly when market sentiment is weak or when the broader index is extended. "
            "On thinly traded stocks, a single high-volume day may be noise rather than institutional activity. "
            "The pattern is also prone to late entries if `breakout_days` is set too long — the breakout reference may already be stretched."
        ),
    },
    "pattern_b": {
        "section": "patterns",
        "label": "Pattern B — Pullback Rebound",
        "summary": "An uptrending stock dips toward its 20-day average and starts bouncing back.",
        "detail": (
            "**Conditions:** SMA50 > SMA200, close still above SMA50, close is within a small buffer of SMA20, "
            "today closes above yesterday by at least a minimum rebound amount, volume is at least mildly supportive.\n\n"
            "**Score boosters:** A tight pullback to SMA20 with quick recovery boosts setup strength. "
            "Moderate volume on the rebound day is sufficient — the bar does not need to be a standout volume day. "
            "Positive SMA50 slope adds the MA bonus.\n\n"
            "**Typical use:** Offers a less-extended entry than Pattern A by waiting for a pause or dip first. "
            "Best for adding to existing positions or entering names that have already proven their trend but are taking a breath.\n\n"
            "**Known weaknesses:** The condition that close stays above SMA50 means the pullback must be shallow. "
            "In a sharper correction, Pattern B won't fire. False signals can appear if the 'pullback' is actually the start of a longer downtrend, "
            "and the SMA50 > SMA200 condition alone cannot confirm trend health in slow-moving markets."
        ),
    },
    "pattern_c": {
        "section": "patterns",
        "label": "Pattern C — MACD Crossover",
        "summary": "The MACD line crosses above its signal line while the broader trend is still healthy.",
        "detail": (
            "**Conditions:** SMA50 > SMA200, MACD line was at or below the signal line on the prior bar "
            "and crosses above it on the current bar, volume above a relaxed threshold.\n\n"
            "**Score boosters:** The crossover gains more setup credit if the cross happens from well below zero "
            "(deeper reset) rather than near the zero line. Volume and trend components still apply normally. "
            "Positive SMA50 slope adds the MA bonus.\n\n"
            "**Typical use:** Captures momentum re-acceleration in names that have already pulled back. "
            "Often fires earlier than Pattern A, before price has made a dramatic breakout bar.\n\n"
            "**Known weaknesses:** MACD crosses are prone to whipsaws in choppy or sideways markets. "
            "A cross near zero is weaker evidence than one that follows a deep correction. "
            "This pattern can also fire on very small momentum moves that have little practical follow-through."
        ),
    },
    "pattern_d": {
        "section": "patterns",
        "label": "Pattern D — RSI Oversold Bounce",
        "summary": "A stock that was oversold on RSI recovers above the threshold while price confirms buyers are back.",
        "detail": (
            "**Conditions:** SMA50 > SMA200, RSI was below the oversold threshold (typically 40) within the last 1–3 bars, "
            "RSI is now back above the threshold and has reclaimed at least 35, improving by a meaningful amount from the prior bar, "
            "price confirms with a close above the previous day's high or a reclaim above SMA20, volume on the reversal day "
            "is at or above the 20-day average, and the close is still near the recent 10-bar swing low.\n\n"
            "**Score boosters:** RSI recovery through 40 adds extra setup strength. "
            "Reversal-day volume well above average further boosts the volume component. "
            "Rebounds very close to the recent swing low earn extra setup credit. Positive SMA50 slope adds the MA bonus.\n\n"
            "**Typical use:** Tries to catch 'washout then bounce' behavior inside a bigger uptrend. "
            "Most useful in strong trend environments where the dip is brief and buyers clearly step in.\n\n"
            "**Known weaknesses:** In the current learned weight history D is one of the weaker families, so it may carry little "
            "or no family bonus. RSI oversold bounces can also be the first leg of a longer breakdown — the SMA uptrend filter "
            "helps reduce this but does not eliminate it. Very fast RSI recoveries can be noise rather than a real reversal."
        ),
    },
    "pattern_e": {
        "section": "patterns",
        "label": "Pattern E — Bollinger Squeeze Breakout",
        "summary": "Volatility contracts to a multi-week low then price breaks out above the upper Bollinger Band.",
        "detail": (
            "**Conditions:** SMA50 > SMA200, Bollinger Band width is at or near a recent multi-bar low (squeeze), "
            "close breaks above the upper Bollinger Band, volume is above average.\n\n"
            "**Score boosters:** A deeper squeeze (lower prior band width) before the breakout earns more setup credit. "
            "Strong above-average volume on the breakout bar drives the volume component higher. "
            "Positive SMA50 slope adds the MA bonus.\n\n"
            "**Typical use:** Best for names that have coiled quietly for several weeks before expanding. "
            "Squeeze breakouts can produce outsized moves because pent-up energy releases quickly.\n\n"
            "**Known weaknesses:** Bollinger Band squeezes can resolve in either direction — the filter requires an upside break, "
            "but a false upside break followed by a reversal is common in choppy environments. "
            "Tight band periods can also produce low-volume breakouts that evaporate quickly."
        ),
    },
    "pattern_f": {
        "section": "patterns",
        "label": "Pattern F — VWAP Reclaim",
        "summary": "Price was trading below a rolling VWAP approximation and snaps back above it on stronger volume.",
        "detail": (
            "**Conditions:** SMA50 > SMA200, previous close was at or below the rolling VWAP approximation, "
            "current close is above the rolling VWAP, volume is above average.\n\n"
            "**Score boosters:** A crisp VWAP cross with strong volume produces the best setup and volume components. "
            "The rolling VWAP on EOD data is an approximation; exact intraday VWAP is not available. "
            "Positive SMA50 slope adds the MA bonus.\n\n"
            "**Typical use:** Best for catching 'buyers took back the line' setups after a brief dip below value. "
            "Often a cleaner-looking setup than MACD or RSI signals because the level is visible on most charts.\n\n"
            "**Known weaknesses:** End-of-day VWAP is an approximation and may not match intraday levels. "
            "On thin-volume sessions the VWAP line can wander, producing unreliable crosses. "
            "A VWAP reclaim in a weakening trend is a false positive risk because the reclaim may fade the next session."
        ),
    },
    "pattern_g": {
        "section": "patterns",
        "label": "Pattern G — VCP Breakout",
        "summary": "A volatility contraction pattern with shrinking pullbacks followed by a high-volume breakout above resistance.",
        "detail": (
            "**Conditions:** Uptrend already in place, at least three pullbacks can be identified from pivot highs, "
            "each pullback is shallower than the one before (contracting volatility), breakout above recent resistance, "
            "volume support on the breakout bar, and relative volume is dry during the contraction phase.\n\n"
            "**Score boosters:** A tighter final contraction (very shallow last base) earns strong setup credit. "
            "A clean high-volume breakout bar drives the volume component strongly. "
            "Positive SMA50 slope adds the MA bonus.\n\n"
            "**Typical use:** The most structure-heavy pattern in the set. Best used for coiled situations "
            "where multiple prior selling waves are clearly getting smaller. "
            "Because it requires a full VCP analysis it fires less frequently than other families.\n\n"
            "**Known weaknesses:** VCP identification is sensitive to parameter choices (pivot detection, contraction thresholds). "
            "The 'breakout above resistance' condition can fire before the pattern is fully complete in fast-moving stocks. "
            "Low-float or index-heavy names may show VCP shapes that are driven by mechanical index rebalancing rather than genuine accumulation."
        ),
    },
    # ── Tag chip glossary ─────────────────────────────────────────────────────
    "tag_uptrend": {
        "section": "tomorrow",
        "label": "Tag: Uptrend",
        "summary": "The stock's 50-day average is above its 200-day average at the signal date.",
        "detail": (
            "The Uptrend chip is present on every signal in Tomorrow's Picks because all pattern families require "
            "SMA50 > SMA200 as a base condition. If this were not true, the signal would not have passed the pattern filter. "
            "It is a reminder that the broader structure was bullish at the time of the trigger."
        ),
    },
    "tag_breakout": {
        "section": "tomorrow",
        "label": "Tag: Breakout",
        "summary": "The pattern name includes 'breakout', suggesting price crossed a prior reference level.",
        "detail": (
            "The Breakout chip appears when the signal's pattern name contains the word 'breakout'. "
            "This covers Pattern A variants that explicitly check for a close above a prior N-day high. "
            "It is a quick visual hint that the setup is a momentum-through-resistance type rather than a pullback or oscillator signal."
        ),
    },
    "tag_volume_ok": {
        "section": "tomorrow",
        "label": "Tag: Volume okay",
        "summary": "The signal's heuristic score is at or above 65, which typically reflects acceptable volume.",
        "detail": (
            "The 'Volume okay' chip is added when the signal's score is at least 65. "
            "Because the volume component (V) feeds directly into the heuristic score with weight 0.13, "
            "a score above this threshold is a rough proxy for the signal having cleared a basic volume quality bar. "
            "It is not a precise volume filter — use the score breakdown to see the exact volume component value."
        ),
    },
    "tag_low_risk": {
        "section": "tomorrow",
        "label": "Tag: Low risk",
        "summary": "The stop distance (risk %) is at or below 7%, suggesting a tight stop relative to entry.",
        "detail": (
            "The 'Low risk' chip appears when the risk percentage (entry-to-stop distance) is at or below 7.0%. "
            "A tighter stop means less capital at risk if the trade fails, and also contributes to a higher risk component (R) in the score. "
            "For sizing purposes, lower risk per trade enables a larger position within a fixed capital-at-risk budget."
        ),
    },
    "tag_rsi_healthy": {
        "section": "tomorrow",
        "label": "Tag: RSI (states)",
        "summary": "The RSI chip shows live RSI state for the selected stock: healthy, cooling, strong, weak, or stretched.",
        "detail": (
            "RSI tags are computed from the latest available close. The states and their bonus effects are:\n\n"
            "- **RSI healthy** (52–68): Momentum is constructive. +3 pts RSI bonus.\n"
            "- **RSI cooling** (45–52): Momentum is tapering after a stronger move. +1 pt.\n"
            "- **RSI strong** (68–78): Momentum is strong; watch for overextension. +1 pt.\n"
            "- **RSI weak** (<45): Momentum is soft for a breakout entry. −5 pts.\n"
            "- **RSI stretched** (>78): Entry may be late; strong pullback risk. −4 pts.\n\n"
            "The bonus is applied to `ui_score` (the displayed score) in real time from price data, "
            "not stored in the CSV signals file."
        ),
    },
    "tag_candle_patterns": {
        "section": "tomorrow",
        "label": "Tag: Candle patterns",
        "summary": "One or more candle-shape enhancers fired on or near the signal date.",
        "detail": (
            "When the enhancer system detects candle shapes (hammer, engulfing, morning star, etc.) in the lookback window "
            "around the signal date, the relevant shape name is appended as a chip. "
            "Multiple shapes can appear simultaneously. The candle chip is teal-coloured to distinguish it from "
            "trend/score chips (green) and caution chips (red/yellow). "
            "See the Candle Enhancers section for a full description of each shape."
        ),
    },
    # ── Lab controls ──────────────────────────────────────────────────────────
    "atr_buffer": {
        "section": "lab",
        "label": "ATR buffer / ATR multiplier",
        "summary": "Controls how far below the swing low or entry the ATR-based stop is placed.",
        "detail": (
            "This slider changes meaning depending on the active stop mode:\n\n"
            "- **Structure + ATR**: The value is an *ATR buffer* added below the recent swing low. "
            "A value of 0.5 means the stop is placed 0.5 × ATR below the structure low. "
            "This keeps the stop below noise while respecting the natural support level.\n\n"
            "- **ATR**: The value is an *ATR multiplier* applied to entry price. "
            "A value of 2.5 means stop = entry − 2.5 × ATR. "
            "Higher values give the trade more room to breathe at the cost of wider risk.\n\n"
            "In both cases the resulting stop is capped by the Fixed stop % so it cannot be wider than you expect."
        ),
    },
    "sort_direction": {
        "section": "lab",
        "label": "Sort direction",
        "summary": "Whether the sort column is ordered from highest to lowest (descending) or lowest to highest (ascending).",
        "detail": (
            "When **Descending** is checked (the default), the highest value of the chosen sort column appears first. "
            "For `signal_score`, `return_pct`, and `pnl` this is almost always what you want. "
            "When unchecked (ascending), the lowest values appear first — useful for inspecting the worst performers "
            "or the most recently dated signals when sorted by `signal_date`."
        ),
    },
    # ── Tomorrow's Picks card meta ────────────────────────────────────────────
    "recommended_date": {
        "section": "tomorrow",
        "label": "Recommended date",
        "summary": "The date the pattern trigger fired, not the intended entry date.",
        "detail": (
            "The recommended date on a stock card is the `signal_date` of the row — when the pattern conditions were met "
            "using end-of-day data. For a next-day entry workflow, the actual entry opportunity would be the trading session "
            "after this date, assuming conditions still look valid on the open. "
            "Older dates in the list mean the signal is a carry-forward from a prior session and should be re-evaluated "
            "against current price before acting."
        ),
    },
    "reason_text": {
        "section": "tomorrow",
        "label": "Reason text",
        "summary": "A short auto-generated sentence summarising the signal's key quality characteristics.",
        "detail": (
            "The reason text is built from the signal score and risk percentage using a template system. "
            "It is not a model-generated analysis — it is a pattern-matched sentence that reflects whether the setup "
            "scored above a threshold, had tight or wide risk, and whether the pattern is a breakout type. "
            "Use it as a quick orientation, not a definitive thesis. "
            "The Score breakdown section gives the actual component values if you want the full picture."
        ),
    },
    # ── Workflow section ──────────────────────────────────────────────────────
    "workflow_universe": {
        "section": "workflow",
        "label": "Stock universe",
        "summary": "The list of tickers the engine watches, stored in universe_tickers.txt.",
        "detail": (
            "The trigger engine reads tickers from `stock_triggers/data/universe_tickers.txt`. "
            "One ticker per line, in Yahoo Finance format (e.g. RELIANCE.NS). "
            "To add or remove a stock, edit this file and re-run the price update step. "
            "The Nifty 50 universe file is also available under `data/stock_universe/ind_nifty50list.csv` "
            "if you want to reset to a known starting set."
        ),
    },
    "workflow_price_update": {
        "section": "workflow",
        "label": "Price update",
        "summary": "Fetches OHLCV history from Yahoo Finance and refreshes prices_eod.csv.",
        "detail": (
            "Run `update_prices_yf.py` (or `update_prices_bhavcopy.py` for NSE Bhavcopy) once per session before building signals. "
            "The script fetches end-of-day OHLCV data and rebuilds `stock_triggers/data/prices_eod.csv`. "
            "All pattern detectors read from this single file, so keeping it current is the most important maintenance step. "
            "Prices from the past three-plus years are retained to support the longer lookback indicators (SMA200, VCP history, etc.)."
        ),
    },
    "workflow_signal_build": {
        "section": "workflow",
        "label": "Signal build",
        "summary": "Scans patterns A–G across all tickers and dates, writing results to signals_all_patterns.csv.",
        "detail": (
            "Run `generate_signals_all_patterns.py` to regenerate the full signal history. "
            "With `--backfill-history` it rescans all available price history. Without flags it processes only the most recent date. "
            "The output `signals_all_patterns.csv` is the primary input for scoring, weight learning, and the Backtesting Lab. "
            "Pattern A also has its own dedicated file via `generate_triggers_pattern_a.py` for backwards compatibility."
        ),
    },
    "workflow_weight_refresh": {
        "section": "workflow",
        "label": "Pattern weight refresh",
        "summary": "Re-learns which pattern families have been performing best and updates the family bonus weights.",
        "detail": (
            "Run `compute_pattern_weights.py` after each signal build to refresh `pattern_weights.json`. "
            "This file records the historical win rate, edge, and score for each family A–G and computes the bonus contribution "
            "each family earns in the scoring model. "
            "If pattern weights are stale, the scoring model may over- or under-credit families that have recently changed in behaviour. "
            "Also run `compute_signal_penalty_weights.py` to refresh per-ticker penalties."
        ),
    },
    "workflow_pipeline": {
        "section": "workflow",
        "label": "One-command pipeline",
        "summary": "run_signal_refresh_pipeline.py chains all the update steps into a single command.",
        "detail": (
            "For a full daily refresh, `run_signal_refresh_pipeline.py --mode daily` runs: "
            "signal penalty weight update → rescore → training history build → stop risk model → "
            "Pattern A file → all-pattern rescore → candle weights → stock scores. "
            "Use `--refresh-prices false` to skip the price download step if you have already updated prices manually. "
            "Use `--recompute-pattern-weights true` to include the family weight refresh in the same pipeline run. "
            "The pipeline writes its results back to the same data files the UI reads, so no further copying is needed."
        ),
    },
    "workflow_stop_risk_model": {
        "section": "workflow",
        "label": "Stop risk model",
        "summary": "A trained model that estimates the probability a given stop distance will hold.",
        "detail": (
            "Run `compute_signal_stop_risk_model.py` to rebuild the stop risk model from the training signal history. "
            "The model uses signal score components as features (or the full score set if `--feature-set scores_only`) "
            "and produces a predicted stop risk value for each signal. "
            "The predictions populate the `reliability_score` and related columns that appear as an alternative ranking method "
            "in Tomorrow's Picks and the lab stop risk filter."
        ),
    },
    "workflow_walk_forward": {
        "section": "workflow",
        "label": "Walk-forward evaluation",
        "summary": "Generates out-of-sample monthly stop risk predictions using a rolling train/test approach.",
        "detail": (
            "Run `evaluate_stop_risk_walk_forward.py --mode walk-forward` to rebuild `stop_risk_walk_forward_monthly.csv`. "
            "The walk-forward method trains on all data up to each month and predicts on the following month (OOS). "
            "A minimum training warmup period must elapse before OOS predictions begin — this is why pre-2024 signals "
            "are excluded from the Backtesting Lab OOS tracker view. "
            "Reducing `--min-train-rows` lowers the warmup requirement and extends the coverage further back, "
            "but at the cost of less training data per fold."
        ),
    },
}


TABLE_HELP_KEYS: dict[str, dict[str, str]] = {
    "pattern_weights": {
        "Pattern": "pattern_family",
        "Score /100": "pattern_score_100",
        "Weight /30": "pattern_weight_30",
        "Count": "count",
        "Win %": "win_rate",
        "Loss %": "loss_rate",
        "Edge pp": "edge_pp",
    },
    "quick_check": {
        "Item": "quick_check",
        "Status": "status",
    },
    "past_results": {
        "signal_date": "source_signal_date",
        "outcome": "outcome",
        "return_pct": "return_pct",
        "exit_date": "exit_date",
    },
    "trade_records": {
        "signal_date": "source_signal_date",
        "ticker": "overview",
        "pattern_family": "pattern_family",
        "pattern": "pattern",
        "entry_price": "entry_price",
        "qty": "qty",
        "invested": "invested",
        "target_price": "target_price",
        "stop_price": "stop_price",
        "latest_close": "latest_close",
        "current_value": "current_value",
        "pnl": "pnl",
        "return_pct": "return_pct",
        "days_held": "days_held",
        "exit_date": "exit_date",
        "status": "status",
        "signal_score": "heuristic_score",
        "score_pattern": "score_pattern",
        "pattern_bonus": "pattern_bonus",
        "sma50_slope_pct": "ma_slope_bonus",
        "ma_slope_bonus": "ma_slope_bonus",
        "stock_rs20": "stock_rs20",
        "stock_rs50": "stock_rs50",
        "rs_bonus": "rs_bonus",
        "enhancer_bonus": "enhancer_bonus",
    },
    "manual_positions": {
        "created_at": "created_at",
        "source_signal_date": "source_signal_date",
        "ticker": "overview",
        "pattern": "pattern",
        "entry_price": "entry_price",
        "stop_price": "stop_price",
        "latest_close": "latest_close",
        "capital": "capital",
        "current_value": "current_value",
        "pnl": "pnl",
        "current_return_pct": "current_return_pct",
        "distance_to_stop_pct": "distance_to_stop_pct",
        "status": "status",
        "note": "note",
    },
}


def get_help_item(help_key: str) -> dict[str, str]:
    return HELP_ITEMS.get(help_key, HELP_ITEMS["overview"])


def _set_docs_focus(help_key: str) -> None:
    item = get_help_item(help_key)
    st.session_state["docs_focus_key"] = help_key
    st.session_state["docs_focus_section"] = item["section"]
    st.session_state["mode"] = "Documentation"


def _ensure_help_chip_css() -> None:
    if st.session_state.get("_docs_chip_css_v8_loaded"):
        return
    st.session_state["_docs_chip_css_v8_loaded"] = True
    st.markdown(
        """
        <style>
        /* Keep help chips theme-neutral so they don't flip dark when the app stays light. */
        a.docs-help-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1rem;
            height: 1rem;
            border-radius: 50%;
            background: rgba(79, 70, 229, 0.08);
            border: 1px solid rgba(79, 70, 229, 0.24);
            color: #4f46e5 !important;
            font-size: 0.6rem;
            font-weight: 800;
            line-height: 1;
            text-decoration: none !important;
            cursor: pointer;
            vertical-align: middle;
            flex-shrink: 0;
            font-family: ui-sans-serif, system-ui, sans-serif;
            letter-spacing: -0.01em;
            transition: background 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease;
        }
        a.docs-help-chip:hover,
        a.docs-help-chip:focus {
            background: rgba(79, 70, 229, 0.14);
            border-color: #4f46e5;
            color: #4f46e5 !important;
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.08);
            text-decoration: none !important;
            outline: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_docs_page_css() -> None:
    if st.session_state.get("_docs_page_css_v6_loaded"):
        return
    st.session_state["_docs_page_css_v6_loaded"] = True
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        .main .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .stApp {
            --docs-bg-soft: #f8f8f6;
            --docs-bg-strong: #f3f4f1;
            --docs-surface: #ffffff;
            --docs-surface-tint: #fcfcfb;
            --docs-border-soft: rgba(15, 23, 42, 0.08);
            --docs-border-strong: rgba(79, 70, 229, 0.18);
            --docs-text-main: #0f172a;
            --docs-text-body: #4b5563;
            --docs-text-muted: #6b7280;
            --docs-accent: #4f46e5;
            --docs-accent-2: #5b67d6;
            --docs-shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.03);
            --docs-shadow-md: 0 6px 18px rgba(15, 23, 42, 0.05);
            --docs-shadow-lg: 0 16px 34px rgba(15, 23, 42, 0.06);
            --docs-radius-lg: 12px;
            --docs-radius-md: 10px;
            --docs-radius-sm: 8px;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }
        .docs-hero-wrap {
            position: relative;
            background: radial-gradient(circle at top left, #eef2ff, #ffffff 68%);
            border: 1px solid rgba(79, 70, 229, 0.12);
            border-radius: var(--docs-radius-lg);
            padding: 1.25rem 1.35rem 1.2rem;
            margin: 0.04rem 0 1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
        }
        .docs-hero-wrap::before {
            content: "";
            display: block;
            width: 40px;
            height: 3px;
            background: var(--docs-accent);
            margin-bottom: 0.6rem;
            border-radius: 2px;
        }
        .docs-hero-title {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: clamp(1.34rem, 1.9vw, 1.76rem);
            line-height: 1.04;
            letter-spacing: -0.014em;
            font-weight: 700;
            color: var(--docs-text-main);
            margin: 0 0 0.38rem;
        }
        .docs-hero-sub {
            font-size: clamp(0.88rem, 1vw, 0.96rem);
            line-height: 1.62;
            letter-spacing: -0.002em;
            color: var(--docs-text-body);
            margin: 0;
            max-width: 74ch;
            font-weight: 450;
        }
        .docs-section-head {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: clamp(1rem, 1.18vw, 1.14rem);
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--docs-text-main);
            margin: 0.08rem 0 0.5rem;
            line-height: 1.1;
            position: sticky;
            top: 0;
            z-index: 10;
            background: rgba(255, 255, 255, 0.92);
            padding: 0.5rem 0 0.35rem;
            backdrop-filter: blur(6px);
        }
        .docs-card {
            border: 1px solid var(--docs-border-soft);
            border-radius: var(--docs-radius-md);
            padding: 0.72rem 0.82rem;
            background: linear-gradient(180deg, var(--docs-surface) 0%, var(--docs-surface-tint) 100%);
            min-height: auto;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
            transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        }
        .docs-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
            border-color: var(--docs-border-strong);
            background: linear-gradient(180deg, var(--docs-surface) 0%, var(--docs-bg-soft) 100%);
        }
        a.docs-entry-card,
        a.docs-index-item {
            display: block;
            text-decoration: none !important;
            color: inherit !important;
        }
        a.docs-entry-card:hover,
        a.docs-entry-card:focus,
        a.docs-index-item:hover,
        a.docs-index-item:focus {
            text-decoration: none !important;
            color: inherit !important;
        }
        .docs-entry-card .docs-card {
            min-height: 122px;
        }
        .docs-entry-cta {
            display: inline-flex;
            align-items: center;
            gap: 0.36rem;
            margin-top: 0.62rem;
            color: var(--docs-accent);
            font-size: 0.82rem;
            font-weight: 600;
            line-height: 1.2;
        }
        .docs-entry-cta::after {
            content: "->";
            font-size: 0.78rem;
        }
        .docs-index-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.6rem 0.85rem;
            margin-top: 0.2rem;
        }
        .docs-index-item {
            border-bottom: 1px solid rgba(15, 23, 42, 0.08);
            padding: 0.06rem 0 0.55rem;
            transition: border-color 0.16s ease, transform 0.16s ease;
        }
        .docs-index-item:hover {
            border-color: color-mix(in srgb, var(--docs-accent) 26%, transparent);
            transform: translateX(2px);
        }
        .docs-index-title {
            display: flex;
            align-items: center;
            gap: 0.42rem;
            margin: 0 0 0.14rem;
            color: var(--docs-text-main);
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: 0.92rem;
            font-weight: 700;
            letter-spacing: -0.012em;
            line-height: 1.18;
        }
        .docs-index-copy {
            margin: 0 0 0 1.52rem;
            color: var(--docs-text-body);
            font-size: 0.8rem;
            line-height: 1.45;
        }
        .docs-card-title {
            margin: 0 0 0.22rem;
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: clamp(0.88rem, 0.98vw, 0.98rem);
            font-weight: 700;
            color: var(--docs-text-main);
            line-height: 1.18;
            letter-spacing: -0.015em;
        }
        .docs-card-sub {
            margin: 0;
            font-size: clamp(0.79rem, 0.92vw, 0.86rem);
            color: var(--docs-text-body);
            line-height: 1.6;
            font-weight: 450;
        }
        .docs-card-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1.1rem;
            height: 1.1rem;
            border-radius: 999px;
            background: color-mix(in srgb, var(--docs-accent) 10%, transparent);
            color: var(--docs-accent);
            border: 1px solid color-mix(in srgb, var(--docs-accent) 18%, transparent);
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: 0.68rem;
            font-weight: 700;
            margin-right: 0.34rem;
        }
        .docs-kicker {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid color-mix(in srgb, var(--docs-accent) 16%, transparent);
            padding: 0.12rem 0.44rem;
            margin-bottom: 0.3rem;
            color: var(--docs-accent);
            background: color-mix(in srgb, var(--docs-accent) 7%, transparent);
            font-size: 0.62rem;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            font-weight: 700;
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
        }
        .docs-intro-text {
            font-size: clamp(0.92rem, 1.15vw, 1rem);
            line-height: 1.7;
            color: var(--docs-text-body);
            font-weight: 500;
            letter-spacing: -0.002em;
        }
        .docs-subsection-title {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: clamp(1rem, 1.3vw, 1.18rem);
            font-weight: 700;
            letter-spacing: -0.01em;
            color: var(--docs-text-main);
            line-height: 1.15;
            margin: 0.2rem 0 0.48rem;
        }
        .docs-step-label {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: clamp(0.9rem, 1.1vw, 1rem);
            font-weight: 700;
            color: var(--docs-text-main);
            line-height: 1.28;
            letter-spacing: -0.008em;
        }
        .docs-story {
            border-left: 3px solid var(--docs-accent);
            border-radius: var(--docs-radius-sm);
            padding: 0.65rem 0.78rem;
            margin: 0.16rem 0 0.45rem;
            background: color-mix(in srgb, var(--docs-accent) 3%, var(--docs-surface));
            border-top: 1px solid var(--docs-border-soft);
            border-right: 1px solid var(--docs-border-soft);
            border-bottom: 1px solid var(--docs-border-soft);
        }
        .docs-story-title {
            font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
            font-size: clamp(0.95rem, 1.15vw, 1.08rem);
            font-weight: 700;
            color: var(--docs-text-main);
            line-height: 1.25;
            letter-spacing: -0.008em;
            margin: 0.08rem 0 0.2rem;
        }
        .docs-muted {
            color: var(--docs-text-muted);
            font-size: clamp(0.82rem, 0.98vw, 0.9rem);
            font-weight: 500;
            line-height: 1.58;
        }
        .docs-snapshot {
            border: 1px solid var(--docs-border-soft);
            border-radius: var(--docs-radius-md);
            background: var(--docs-surface);
            padding: 0.68rem 0.75rem;
            margin: 0.28rem 0 0.42rem;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.035);
        }
        .docs-snapshot-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 0.28rem;
            margin-top: 0.04rem;
            align-items: center;
        }
        .docs-step-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.28rem;
            border-radius: 999px;
            border: 1px solid var(--docs-border-soft);
            background: color-mix(in srgb, var(--docs-accent) 4%, transparent);
            color: var(--docs-text-main);
            padding: 0.18rem 0.42rem 0.18rem 0.36rem;
            font-size: clamp(0.75rem, 0.88vw, 0.8rem);
            line-height: 1.2;
            font-weight: 600;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
            transition: background 0.16s ease, border-color 0.16s ease;
        }
        .docs-step-chip:hover {
            background: color-mix(in srgb, var(--docs-accent) 12%, transparent);
            border-color: var(--docs-accent);
            transform: translateY(-1px);
        }
        .docs-step-dot {
            width: 0.28rem;
            height: 0.28rem;
            border-radius: 999px;
            background: var(--docs-accent);
            flex-shrink: 0;
        }
        .docs-step-arrow {
            color: var(--docs-text-muted);
            font-weight: 700;
            margin: 0 0.08rem;
            font-size: 0.85em;
        }
        .docs-ref-note {
            padding: 0.58rem 0.68rem;
            border-radius: var(--docs-radius-sm);
            border: 1px dashed var(--docs-border-soft);
            background: color-mix(in srgb, var(--docs-accent) 4%, transparent);
            color: var(--docs-text-muted);
            font-size: clamp(0.76rem, 0.92vw, 0.84rem);
            line-height: 1.5;
            margin-top: 0.28rem;
            font-weight: 500;
        }
        .docs-note {
            background: #f9fafb;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 10px;
            padding: 0.7rem 0.8rem;
            margin: 0.6rem 0;
            color: var(--docs-text-body);
            font-size: 0.85rem;
            line-height: 1.55;
        }
        .stButton > button {
            border-radius: 8px;
            border: 1px solid var(--docs-border-soft);
            background: transparent;
            color: var(--docs-text-body);
            font-weight: 600;
            font-size: clamp(0.82rem, 0.95vw, 0.88rem);
            min-height: 1.95rem;
            transition: border-color 0.15s ease, transform 0.15s ease;
            font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        }
        .stButton > button:hover {
            border-color: var(--docs-accent);
            color: var(--docs-accent);
        }
        .stButton > button:active {
            transform: scale(0.97);
        }
        @media (max-width: 1024px) {
            .docs-card {
                padding: 0.68rem 0.76rem;
            }
            .docs-card-title {
                font-size: 0.92rem;
                margin-bottom: 0.22rem;
            }
            .docs-card-sub {
                font-size: 0.82rem;
            }
            .docs-hero-wrap {
                padding: 1.05rem 1.1rem 1.02rem;
            }
            .docs-hero-title {
                font-size: 1.4rem;
                margin-bottom: 0.28rem;
            }
            .docs-section-head {
                font-size: 1rem;
                margin-bottom: 0.44rem;
            }
        }
        @media (max-width: 768px) {
            .docs-index-grid {
                grid-template-columns: 1fr;
                gap: 0.42rem;
            }
            .docs-hero-wrap {
                padding: 0.94rem 0.96rem 0.92rem;
            }
            .docs-hero-title {
                font-size: 1.22rem;
                margin-bottom: 0.24rem;
            }
            .docs-hero-sub {
                font-size: 0.83rem;
            }
            .docs-section-head {
                font-size: 0.96rem;
                margin-bottom: 0.4rem;
            }
            .docs-subsection-title {
                font-size: 0.9rem;
                margin-bottom: 0.3rem;
            }
            .docs-card {
                padding: 0.66rem 0.72rem;
            }
            .docs-card-title {
                font-size: 0.86rem;
                margin-bottom: 0.2rem;
            }
        }
        @media (max-width: 480px) {
            .main .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }
            .docs-hero-wrap {
                padding: 0.74rem 0.8rem 0.8rem;
            }
            .docs-hero-title {
                font-size: 1.04rem;
                margin-bottom: 0.22rem;
            }
            .docs-hero-sub {
                font-size: 0.8rem;
            }
            .docs-section-head {
                font-size: 0.9rem;
                margin-bottom: 0.36rem;
            }
            .docs-card {
                padding: 0.62rem 0.68rem;
            }
            .docs-card-title {
                font-size: 0.82rem;
                margin-bottom: 0.18rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow_snapshot(label: str, flow: str) -> None:
    steps = [part.strip() for part in flow.split("->") if part.strip()]
    chips = []
    for idx, step in enumerate(steps):
        chips.append(f"<span class='docs-step-chip'><span class='docs-step-dot'></span>{step}</span>")
        if idx < len(steps) - 1:
            chips.append("<span class='docs-step-arrow'>-></span>")

    st.markdown(
        f"<div class='docs-snapshot'><div class='docs-kicker'>{label}</div>"
        f"<div class='docs-snapshot-steps'>{''.join(chips) if chips else flow}</div></div>",
        unsafe_allow_html=True,
    )


def _jump_link_for_section(section_id: str) -> str:
    return f"?docs_section={section_id}"


def handle_docs_section_query_param() -> None:
    section_id = str(st.query_params.get("docs_section", "") or "").strip()
    if not section_id:
        return
    allowed_sections = {
        "getting_started",
        "score_explainer",
        "reference",
        "daily_review_flow",
        "pattern_library",
        "risk_catalysts",
    }
    if section_id in allowed_sections:
        params = dict(st.query_params)
        params.pop("docs_section", None)
        st.query_params.from_dict(params)
        st.session_state["docs_active_section"] = section_id
        st.rerun()


def handle_help_query_param() -> None:
    """Call once at top of app — intercepts ?help=<key> clicks from HTML chips."""
    focus = st.query_params.get("help", "")
    if focus and focus in HELP_ITEMS:
        # Remove the param so it doesn't persist across reruns
        params = dict(st.query_params)
        params.pop("help", None)
        st.query_params.from_dict(params)
        _set_docs_focus(focus)
        st.rerun()


def render_help_button(help_key: str, *, key: str, tooltip: str | None = None) -> None:
    _ensure_help_chip_css()
    item = get_help_item(help_key)
    title_text = (tooltip or f"{item['label']}: {item['summary']}").replace('"', "&quot;").replace("'", "&#39;")
    st.markdown(
        f'<a class="docs-help-chip" href="?help={help_key}" title="{title_text}">?</a>',
        unsafe_allow_html=True,
    )


def render_heading_with_help(
    title: str,
    help_key: str,
    *,
    key: str,
    level: int = 3,
    caption: str | None = None,
) -> None:
    left, right = st.columns([0.94, 0.06])
    with left:
        st.markdown(f"{'#' * int(level)} {title}")
    with right:
        render_help_button(help_key, key=key)
    if caption:
        st.caption(caption)


def render_caption_with_help(text: str, help_key: str, *, key: str) -> None:
    left, right = st.columns([0.9, 0.1], gap="small")
    with left:
        st.markdown(
            f"<div style='font-size:0.78rem; color:var(--heading-color); font-weight:700; line-height:1.15;'>{text}</div>",
            unsafe_allow_html=True,
        )
    with right:
        render_help_button(help_key, key=key)


def table_help_map(table_name: str, columns: Iterable[str]) -> dict[str, str]:
    table_map = TABLE_HELP_KEYS.get(table_name, {})
    return {column: table_map[column] for column in columns if column in table_map}


def build_dataframe_column_config(column_help_keys: dict[str, str]) -> dict[str, object]:
    config: dict[str, object] = {}
    for column, help_key in column_help_keys.items():
        item = get_help_item(help_key)
        config[column] = st.column_config.Column(column, help=item["summary"])
    return config


def render_table_help_glossary(
    title: str,
    column_help_keys: dict[str, str],
    *,
    key_prefix: str,
    expanded: bool = False,
) -> None:
    if not column_help_keys:
        return

    with st.expander(f"{title} column help", expanded=expanded):
        for idx, (column, help_key) in enumerate(column_help_keys.items()):
            item = get_help_item(help_key)
            left, right = st.columns([0.88, 0.12])
            with left:
                st.markdown(f"**{column}**  \\n+{item['summary']}")
            with right:
                render_help_button(help_key, key=f"{key_prefix}_{idx}")


def _render_pattern_map() -> None:
    """Render the A-G pattern-family relationship map as a Graphviz diagram."""
    st.markdown(
        """
**All 7 pattern families share one prerequisite — the stock must be in an established uptrend**
(50-day SMA above 200-day SMA). Once that structural condition is met, each pattern hunts for a
different entry trigger:

| | Pattern | Entry trigger |
|---|---|---|
| **A** | Breakout | Volume-confirmed close above resistance |
| **B** | Pullback | First healthy retracement in an uptrend |
| **C** | MACD | MACD line crosses above signal line |
| **D** | RSI | Bounce off oversold RSI (<35) |
| **E** | BB Squeeze | Bollinger Band contraction expansion |
| **F** | VWAP | Intraday reclaim of VWAP |
| **G** | VCP | Volatility Contraction Pattern breakout |
"""
    )
    try:
        st.graphviz_chart(
            """
            digraph PatternMap {
                rankdir=TB;
                graph [nodesep=0.25, ranksep=0.35];
                node [shape=box, style="rounded,filled", fillcolor="#0f172a",
                      color="#38bdf8", fontcolor="#fafafa", fontname="Helvetica",
                      fontsize=9, margin="0.12,0.06", width=0.7, height=0.35];
                edge [color="#475569"];
                TREND [label="Uptrend  SMA50 > SMA200", shape=ellipse,
                       fillcolor="#172033", color="#f59e0b", fontsize=9,
                       width=2.2, height=0.4];
                A [label="A\nBreakout"];
                B [label="B\nPullback"];
                C [label="C\nMACD"];
                D [label="D\nRSI"];
                E [label="E\nBB Sqz"];
                F [label="F\nVWAP"];
                G [label="G\nVCP"];
                TREND -> A;
                TREND -> B;
                TREND -> C;
                TREND -> D;
                TREND -> E;
                TREND -> F;
                TREND -> G;
            }
            """,
            use_container_width=True,
        )
    except Exception:
        pass  # Graphviz not installed — skip silently


def _render_pattern_example_chart(family: str) -> None:
    """Embed the pre-built SVG example chart for a pattern family."""
    svg_path = _CHART_DIR / _PATTERN_CHART_FILES.get(family, "")
    if not svg_path.is_file():
        return
    try:
        import re

        svg_text = svg_path.read_text(encoding="utf-8")
        # Strip XML declaration so it embeds cleanly as inline HTML
        svg_text = svg_text.strip()
        if svg_text.startswith("<?xml"):
            svg_text = svg_text[svg_text.index("<svg"):]
        svg_text = re.sub(
            r"<svg\b",
            "<svg style='display:block;width:100%;max-width:760px;height:auto;margin:0 auto;'",
            svg_text,
            count=1,
        )
        caption = _PATTERN_CHART_CAPTIONS.get(family, "")
        st.markdown(
            f"<div style='border:1px solid rgba(15,23,42,0.10);border-radius:8px;"
            f"overflow:hidden;margin:0.4rem auto 0.15rem;max-width:760px;background:#fff;'>{svg_text}</div>",
            unsafe_allow_html=True,
        )
        if caption:
            st.caption(f"Historical example — {caption}")
    except Exception:
        pass  # File unreadable — skip silently


def _render_candle_enhancer_diagram(help_key: str) -> None:
    """Render a compact schematic candlestick diagram for a candle enhancer."""
    import plotly.graph_objects as go

    entry = _ENHANCER_CANDLE_DATA.get(help_key)
    if entry is None:
        return

    candles = entry["candles"]
    n_signal: int = entry["n_signal"]
    caption: str = entry.get("caption", "")

    xs = [c[0] for c in candles]
    opens = [c[1] for c in candles]
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]

    fig = go.Figure(go.Candlestick(
        x=xs,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        increasing_line_color="#22c55e",
        increasing_fillcolor="#22c55e",
        decreasing_line_color="#ef4444",
        decreasing_fillcolor="#ef4444",
        showlegend=False,
        name="",
    ))

    # Arrow annotation at the start of the signal bar(s)
    sig_start_x = xs[len(candles) - n_signal]
    sig_peak = max(highs[len(candles) - n_signal:])
    fig.add_annotation(
        x=sig_start_x,
        y=sig_peak,
        text="▲ signal",
        showarrow=True,
        arrowhead=2,
        arrowsize=0.9,
        arrowcolor="#f59e0b",
        font=dict(color="#f59e0b", size=10),
        ay=-28,
        ax=0,
        bgcolor="#0e1117",
    )

    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=28, b=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#9ca3af", size=10),
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            rangeslider=dict(visible=False),
            type="category",
            tickfont=dict(size=9, color="#475569"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#1e293b",
            showticklabels=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"enhancer_diag_{help_key}")
    if caption:
        st.caption(f"Schematic — {caption}")


def _pattern_family_lines() -> list[str]:
    return [
        "**A**: breakout-style setups that rely on strength through a prior reference zone.",
        "**B**: alternate breakout or continuation behavior with a different trigger shape.",
        "**C**: MACD-oriented setups that look for momentum confirmation.",
        "**D**: RSI-oriented setups that look for strength and reset behavior.",
        "**E**: Bollinger-style setups that use band structure or compression clues.",
        "**F**: VWAP-oriented setups that care about price behavior around intraday or anchored reference value.",
        "**G**: VCP-style contraction and breakout setups.",
    ]


def _render_score_formula() -> None:
    st.markdown(
        "The five independent components are weighted and summed into a base score:"
    )
    st.latex(
        r"\text{Score}_{\text{base}} = \underbrace{0.20\,T}_{\text{Trend}}"
        r"+ \underbrace{0.20\,S}_{\text{Setup}}"
        r"+ \underbrace{0.13\,V}_{\text{Volume}}"
        r"+ \underbrace{0.14\,R}_{\text{Risk}}"
        r"+ \underbrace{0.03\,I}_{\text{RSI}}"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Component ranges (each clipped 0–100)**\n\n"
            "- **T** = clip(50 + trend\\_pct × 5)\n"
            "- **S** = clip(50 + setup\\_pct × 8)\n"
            "- **V** = clip(40 + volume\\_ratio × 20)\n"
            "- **R** = clip(100 − stop\\_pct × 6)\n"
            "- **I** = RSI value, or 50 if unavailable"
        )
    with col2:
        st.markdown(
            "**Additive bonuses applied after base score**\n\n"
            "- **B_pattern** — learned family bonus up to +30 pts\n"
            "- **B_slope** — SMA50 acceleration bonus, capped +3 pts\n"
            "- **B_consensus** — multi-family agreement bonus\n\n"
            "Final score = clip(Score_base + B_pattern + B_slope + B_consensus, 0, 100)"
        )
    st.caption(
        "Weight breakdown: Trend 20 pts · Setup 20 pts · Volume 13 pts · Risk 14 pts · RSI 3 pts · Pattern ≤30 pts · Slope ≤3 pts."
    )


# Preferred display order within key sections (unlisted items fall back to alpha)
SECTION_ITEM_ORDER: dict[str, list[str]] = {
    "scores": [
        "heuristic_score", "reliability_score", "stop_risk",
        "min_score_filter", "sort_order", "overview_metrics",
        "entry_price", "stop_price", "trade_risk",
        "score_breakdown",
        "penalty_weights", "oos_stop_risk",
    ],
    "scoring_formula": [
        "score_trend_comp", "score_setup_comp", "score_volume_comp",
        "score_risk_comp", "score_rsi_comp",
        "ma_slope_bonus", "pattern_family_bonus_formula", "consensus_bonus",
    ],
    "patterns": [
        "pattern_a", "pattern_b", "pattern_c", "pattern_d",
        "pattern_e", "pattern_f", "pattern_g",
        "pattern_family", "pattern", "learned_pattern_weights",
        "pattern_score_100", "pattern_weight_30",
        "count", "win_rate", "loss_rate", "edge_pp",
    ],
    "lab": [
        "analysis_setup",
        "target_pct", "stop_mode", "atr_buffer",
        "capital_per_trade", "min_score_filter",
        "sort_order", "sort_direction",
        "summary_kpis", "model_reference",
    ],
    "trade_records": [
        "status", "return_pct", "pnl", "days_held", "exit_date",
        "entry_price", "stop_price", "target_price",
        "qty", "invested", "current_value", "latest_close",
        "score_pattern", "pattern_bonus",
        "stock_rs20", "stock_rs50", "rs_bonus", "enhancer_bonus",
    ],
    "tomorrow": [
        "tomorrow_picks", "scoring_method",
        "recommended_date", "reason_text",
        "tag_uptrend", "tag_breakout", "tag_volume_ok",
        "tag_low_risk", "tag_rsi_healthy", "tag_candle_patterns",
        "quick_check",
    ],
    "enhancers": [
        "enhancer_hammer", "enhancer_confirmed_hammer_a",
        "enhancer_engulfing_trend_combo",
        "enhancer_piercing_variant_b_combo",
        "enhancer_marubozu", "enhancer_belt_hold",
        "enhancer_engulfing", "enhancer_harami",
        "enhancer_morning_star", "enhancer_three_white_soldiers",
        "enhancer_piercing", "enhancer_piercing_variant",
        "enhancer_inverted_hammer", "enhancer_doji",
    ],
    "workflow": [
        "workflow_pipeline", "workflow_price_update",
        "workflow_signal_build", "workflow_weight_refresh",
        "workflow_stop_risk_model", "workflow_walk_forward",
        "workflow_universe",
    ],
}


def _sort_section_items(
    section_id: str,
    items: list[tuple[str, dict[str, str]]],
) -> list[tuple[str, dict[str, str]]]:
    """Sort items by SECTION_ITEM_ORDER preference, then alpha for unregistered keys."""
    order = SECTION_ITEM_ORDER.get(section_id, [])
    order_map = {key: idx for idx, key in enumerate(order)}
    return sorted(items, key=lambda pair: (order_map.get(pair[0], 9999), pair[1]["label"]))


def _highlight(text: str, query: str) -> str:
    """Wrap all case-insensitive occurrences of *query* in <strong> tags."""
    if not query:
        return text
    import re
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group(0)}**", text)


@st.cache_data(show_spinner=False)
def _load_docs_signals() -> pd.DataFrame:
    path = _DATA_DIR / "signals_all_patterns.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "signal_date" in df.columns:
            df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_docs_stock_scores() -> pd.DataFrame:
    path = _DATA_DIR / "stock_scores.csv"
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_docs_json(file_name: str) -> dict:
    path = _DATA_DIR / file_name
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _load_docs_csv(file_name: str) -> pd.DataFrame:
    path = _DATA_DIR / file_name
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _build_case_studies(limit: int = 3) -> list[dict[str, str]]:
    signals = _load_docs_signals()
    if signals.empty:
        return []

    if "signal_score" not in signals.columns:
        return []

    scored = signals.copy()
    scored["signal_score"] = pd.to_numeric(scored["signal_score"], errors="coerce")
    scored = scored.dropna(subset=["signal_score"])
    if "signal_date" in scored.columns:
        scored = scored.dropna(subset=["signal_date"]) 
        scored = scored.sort_values(["signal_date", "signal_score"], ascending=[False, False])
    else:
        scored = scored.sort_values("signal_score", ascending=False)

    seen_tickers: set[str] = set()
    stories: list[dict[str, str]] = []
    component_cols = [
        "score_setup",
        "score_trend",
        "score_volume",
        "score_risk",
        "score_rsi",
        "pattern_bonus",
        "ma_slope_bonus",
        "consensus_bonus",
    ]

    for _, row in scored.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker or ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)

        family = str(row.get("pattern_family", "-")).strip() or "-"
        score = float(row.get("signal_score", 0.0))
        signal_date = row.get("signal_date")
        date_text = "N/A"
        if pd.notna(signal_date):
            date_text = pd.to_datetime(signal_date).strftime("%Y-%m-%d")

        contributions: list[tuple[str, float]] = []
        for col in component_cols:
            if col in scored.columns:
                val = pd.to_numeric(row.get(col), errors="coerce")
                if pd.notna(val):
                    label = col.replace("score_", "").replace("_", " ").title()
                    contributions.append((label, float(val)))
        contributions.sort(key=lambda item: item[1], reverse=True)

        top_contrib = ", ".join([f"{name}: {value:.1f}" for name, value in contributions[:3]]) or "No component details"
        low_contrib = ", ".join([f"{name}: {value:.1f}" for name, value in contributions[-2:]]) if len(contributions) >= 2 else "No penalty detail"

        summary = (
            f"{ticker} ranked high because setup quality and trend context aligned, "
            f"while the family {family} overlay reinforced the base score."
        )
        stories.append(
            {
                "ticker": ticker,
                "family": family,
                "score": f"{score:.1f}",
                "date": date_text,
                "summary": summary,
                "top": top_contrib,
                "risks": low_contrib,
            }
        )
        if len(stories) >= limit:
            break
    return stories


def _render_focus_banner(focus_item: dict[str, str] | None) -> None:
    if not focus_item:
        return
    section_title = SECTION_COPY.get(focus_item["section"], {}).get("title", "")
    st.markdown(
        f"<div style='background:rgba(56,189,248,0.08);border-left:3px solid #38bdf8;"
        f"padding:0.75rem 1rem 0.75rem 1rem;border-radius:0 4px 4px 0;margin-bottom:1rem'>"
        f"<div style='font-size:0.68rem;text-transform:uppercase;letter-spacing:0.1em;"
        f"color:#38bdf8;margin-bottom:0.25rem;font-weight:700'>Opened from help chip · {section_title}</div>"
        f"<div style='font-size:1rem;font-weight:700;margin-bottom:0.35rem'>{focus_item['label']}</div>"
        f"<div style='font-size:0.88rem;line-height:1.55'>{focus_item['detail']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_top_rail(grouped_items: dict[str, list[tuple[str, dict[str, str]]]]) -> str:
    if st.session_state.pop("_docs_search_clear", False):
        st.session_state["docs_search"] = ""

    left, right = st.columns([0.65, 0.35], gap="small")
    with left:
        search_query = st.text_input(
            "Search docs",
            value="",
            placeholder="Search terms, metrics, patterns, controls...",
            key="docs_search",
            label_visibility="collapsed",
        )
    with right:
        section_ids = list(SECTION_COPY.keys())
        options = ["All sections"] + [f"{SECTION_COPY[s]['title']} ({len(grouped_items.get(s, []))})" for s in section_ids]
        selected = st.selectbox("Section filter", options=options, key="docs_section_filter", label_visibility="collapsed")
        st.session_state["docs_section_filter_id"] = "" if selected == "All sections" else section_ids[options.index(selected) - 1]

    return search_query.strip().lower()


def _render_hero_and_entry_cards() -> None:
    st.markdown(
        (
            "<div class='docs-hero-wrap'>"
            "<div class='docs-kicker'>Documentation Hub</div>"
            "<div class='docs-hero-title'>Documentation</div>"
            "<div class='docs-hero-sub'>"
            "Learn the workflow end-to-end, then use the reference layer for precise lookups. "
            "Tomorrow's Picks is your shortlist engine, while Backtesting Lab is your validation surface."
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(
            "<a class='docs-entry-card' href='?docs_section=getting_started'>"
            "<div class='docs-card'><p class='docs-card-title'>Start Here</p>"
            "<p class='docs-card-sub'>New to the app and need a fast orientation.</p>"
            "<div class='docs-entry-cta'>Open Getting Started</div></div></a>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<a class='docs-entry-card' href='?docs_section=score_explainer'>"
            "<div class='docs-card'><p class='docs-card-title'>Understand Scoring</p>"
            "<p class='docs-card-sub'>How score components, bonuses, and penalties combine.</p>"
            "<div class='docs-entry-cta'>Open Score Explainer</div></div></a>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<a class='docs-entry-card' href='?docs_section=reference'>"
            "<div class='docs-card'><p class='docs-card-title'>Look Up A Term</p>"
            "<p class='docs-card-sub'>Jump straight to definitions, controls, and field meanings.</p>"
            "<div class='docs-entry-cta'>Open Reference</div></div></a>",
            unsafe_allow_html=True,
        )


def _render_quick_map() -> None:
    st.markdown("<div class='docs-section-head'>Documentation Map</div>", unsafe_allow_html=True)
    blocks = [
        ("getting_started", "Getting Started", "What the app does, where to begin, daily checklist."),
        ("daily_review_flow", "Daily Review Flow", "Tomorrow's Picks -> stock detail -> lab drilldown."),
        ("score_explainer", "Score Explainer", "Components, bonuses, penalties, and practical interpretation."),
        ("pattern_library", "Pattern Library", "A-G visual cards with what-to-notice guidance."),
        ("risk_catalysts", "Risk and Catalysts", "Stop risk, penalties, catalyst gates, before/after framing."),
        ("reference", "Reference", "Grouped glossary and searchable help topics."),
    ]
    index_html = ["<div class='docs-index-grid'>"]
    for idx, (section_id, title, body) in enumerate(blocks):
        index_html.append(
            f"<a class='docs-index-item' href='{_jump_link_for_section(section_id)}'>"
            f"<div class='docs-index-title'><span class='docs-card-index'>{idx + 1}</span>{title}</div>"
            f"<p class='docs-index-copy'>{body}</p></a>"
        )
    index_html.append("</div>")
    st.markdown("".join(index_html), unsafe_allow_html=True)


def _render_getting_started() -> None:
    st.markdown("<div class='docs-section-head'>Getting Started</div>", unsafe_allow_html=True)
    left, right = st.columns([0.62, 0.38], gap="medium")
    with left:
        st.markdown(
            "Stock Operator ranks and explains signal quality so you can review fewer names with more context. "
            "Use Tomorrow's Picks for fast prioritization, then validate assumptions in Backtesting Lab."
        )
        st.markdown(
            "<div class='docs-note'>Tip: treat the daily shortlist as a review queue, not an auto-trade list. "
            "The page is strongest when you move from shortlist to context to validation.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "### If you only do three things\n"
            "1. Refresh prices and signals daily.\n"
            "2. Review top-ranked rows with score breakdown before acting.\n"
            "3. Validate recurring ideas in Backtesting Lab before increasing conviction."
        )
    with right:
        _render_workflow_snapshot(
            "Workflow Snapshot",
            "Update prices -> Generate all-pattern signals -> Refresh weights -> Review picks -> Validate in lab",
        )


def _render_daily_review_flow() -> None:
    st.markdown("<div class='docs-section-head'>Daily Review Flow</div>", unsafe_allow_html=True)
    _render_workflow_snapshot(
        "Review Snapshot",
        "Open Tomorrow's Picks -> Inspect one stock -> Check context -> Validate in Backtesting Lab",
    )
    steps = [
        ("1. Open Tomorrow's Picks", "Set lens, cutoff, and inspect the shortlisted names."),
        ("2. Inspect one stock", "Read score breakdown, risk width, pattern family, and reason text."),
        ("3. Check context", "Use pattern and candle overlays to understand conviction and weakness."),
        ("4. Validate in Backtesting Lab", "Stress test with target/stop assumptions before committing."),
    ]
    for idx in range(0, len(steps), 2):
        cols = st.columns(2, gap="small")
        row_steps = steps[idx: idx + 2]
        for col_idx, (title, body) in enumerate(row_steps):
            with cols[col_idx]:
                st.markdown(
                    f"<div class='docs-card'><p class='docs-card-title'>{title}</p>"
                    f"<p class='docs-card-sub'>{body}</p></div>",
                    unsafe_allow_html=True,
                )


def _render_case_studies() -> None:
    st.markdown("<div class='docs-section-head'>Why This Stock Ranked High</div>", unsafe_allow_html=True)
    stories = _build_case_studies(limit=3)
    if not stories:
        st.info("No case-study rows available yet. Build or refresh signals_all_patterns.csv to populate this section.")
        return

    for idx, story in enumerate(stories):
        st.markdown(
            "<div class='docs-story'>"
            f"<div class='docs-kicker'>Pattern {story['family']}</div>"
            f"<p class='docs-card-title'>{story['ticker']} · Score {story['score']}</p>"
            f"<p class='docs-muted'>As-of signal date: {story['date']}</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(story["summary"])
        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.markdown("**Top contributors**")
            st.caption(story["top"])
        with c2:
            st.markdown("**Risks and weak spots**")
            st.caption(story["risks"])
        if idx < len(stories) - 1:
            st.markdown("---")


def _render_score_explainer() -> None:
    st.markdown("<div class='docs-section-head'>How Scoring Works</div>", unsafe_allow_html=True)
    _render_workflow_snapshot(
        "Scoring Snapshot",
        "Component score -> Pattern and MA bonuses -> Consensus adjustment -> Penalty overlays -> Final rank",
    )
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown(
            "<div class='docs-card'><p class='docs-card-title'>Core components</p>"
            "<p class='docs-card-sub'>Trend, setup, volume, risk, and RSI build the base score.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            "<div class='docs-card'><p class='docs-card-title'>Positive overlays</p>"
            "<p class='docs-card-sub'>Pattern family, MA slope, and consensus bonuses can lift rank.</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            "<div class='docs-card'><p class='docs-card-title'>Risk controls</p>"
            "<p class='docs-card-sub'>Penalty weights and stop-risk gating prevent weak repeat candidates.</p></div>",
            unsafe_allow_html=True,
        )

    _render_score_formula()

    comp_cols = st.columns(2, gap="small")
    with comp_cols[0]:
        st.markdown("### Strong setup profile")
        st.caption("High setup + trend + volume with controlled stop width usually ranks cleanly.")
    with comp_cols[1]:
        st.markdown("### Weak setup profile")
        st.caption("Wide stop, weak volume, and lower setup quality can stay visible but rank lower.")


def _render_pattern_gallery() -> None:
    st.markdown("<div class='docs-section-head'>Pattern Library</div>", unsafe_allow_html=True)
    _render_workflow_snapshot(
        "Pattern Snapshot",
        "Uptrend prerequisite -> Pattern trigger A-G -> Family context -> Score impact -> Lab validation",
    )
    _render_pattern_map()
    families = [
        ("A", "pattern_a"),
        ("B", "pattern_b"),
        ("C", "pattern_c"),
        ("D", "pattern_d"),
        ("E", "pattern_e"),
        ("F", "pattern_f"),
        ("G", "pattern_g"),
    ]
    for family, help_key in families:
        item = get_help_item(help_key)
        st.markdown(f"### Pattern {family} · {item['label'].split('—')[-1].strip()}")
        st.caption(item.get("summary", ""))
        _render_pattern_example_chart(family)
        st.markdown(item.get("detail", ""))
        if st.button(
            f"Open Backtesting Lab for Pattern {family}",
            key=f"docs_gallery_nav_{family}",
            width="stretch",
        ):
            st.session_state["mode"] = "Backtest Lab"
            st.session_state["lab_family_filter"] = [family]
            st.session_state["docs_focus_key"] = ""
            st.session_state["_nav_skip_sync"] = True
            st.rerun()


def _render_enhancer_gallery() -> None:
    st.markdown("<div class='docs-section-head'>Candle Enhancers And Signal Context</div>", unsafe_allow_html=True)
    enhancer_keys = [k for k in SECTION_ITEM_ORDER.get("enhancers", []) if k.startswith("enhancer_")]
    for idx in range(0, len(enhancer_keys), 2):
        cols = st.columns(2, gap="small")
        pair = enhancer_keys[idx: idx + 2]
        for col_idx, key in enumerate(pair):
            item = get_help_item(key)
            with cols[col_idx]:
                st.markdown(f"### {item['label']}")
                st.caption(item.get("summary", ""))
                _render_candle_enhancer_diagram(key)
                st.markdown(item.get("detail", ""))


def _render_risk_and_catalysts() -> None:
    st.markdown("<div class='docs-section-head'>Risk, Penalties, And Catalysts</div>", unsafe_allow_html=True)
    _render_workflow_snapshot(
        "Risk Snapshot",
        "Raw signal -> Stop-risk filter -> Ticker penalty overlay -> Catalyst adjustment -> Final shortlist",
    )
    st.markdown(get_help_item("penalty_weights")["detail"])
    st.markdown(get_help_item("oos_stop_risk")["detail"])

    gates = _load_docs_json("catalyst_gates_validation.json")
    if gates:
        st.markdown("### Catalyst gate checks")
        rows = []
        for key, value in gates.items():
            rows.append({"Gate": str(key), "Value": value})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("Catalyst gate validation data not found. Run the catalyst validation pipeline to populate this block.")

    combo = _load_docs_csv("combo_analysis_summary.csv")
    if not combo.empty:
        st.markdown("### Family combo behavior")
        view_cols = [c for c in ["pair", "win_rate_pct", "edge_pp", "n"] if c in combo.columns]
        if view_cols:
            st.dataframe(combo[view_cols].head(10), use_container_width=True, hide_index=True)


def _render_reference_layer(
    grouped_items: dict[str, list[tuple[str, dict[str, str]]]],
    focus_key: str,
    active_filter: str,
) -> None:
    st.markdown("<div class='docs-section-head'>Reference And Glossary</div>", unsafe_allow_html=True)
    _render_workflow_snapshot(
        "Reference Snapshot",
        "Search term -> Section group -> Topic detail -> Related controls -> Action in app",
    )
    groups = {
        "Tomorrow's Picks": ["overview", "tomorrow", "scores", "past_results"],
        "Backtesting Lab": ["lab", "trade_records", "manual_positions"],
        "Patterns And Enhancers": ["patterns", "enhancers", "scoring_formula"],
        "Engine Workflow": ["workflow"],
    }

    for group_title, sections in groups.items():
        if active_filter and active_filter not in sections:
            continue
        with st.expander(group_title, expanded=bool(active_filter)):
            for section_id in sections:
                items = _sort_section_items(section_id, grouped_items.get(section_id, []))
                if not items:
                    continue
                st.markdown(f"### {SECTION_COPY.get(section_id, {}).get('title', section_id)}")
                for help_key, item in items:
                    expander_title = item["label"]
                    if item.get("summary"):
                        expander_title = f"{expander_title} — {item['summary']}"
                    with st.expander(expander_title, expanded=focus_key == help_key):
                        st.markdown(item["detail"])
                        if focus_key == help_key:
                            st.caption("Opened from a live UI help chip.")
    st.markdown(
        "<div class='docs-ref-note'>Tip: use Search docs above for exact field names, then return here for section-level context.</div>",
        unsafe_allow_html=True,
    )


def _render_search_results(
    search_q: str,
    grouped_items: dict[str, list[tuple[str, dict[str, str]]]],
    focus_key: str,
    active_filter: str,
) -> bool:
    if not search_q:
        return False

    matched_by_section: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for help_key, item in HELP_ITEMS.items():
        if active_filter and item["section"] != active_filter:
            continue
        haystack = f"{item['label']} {item.get('summary', '')} {item['detail']}".lower()
        if search_q in haystack:
            matched_by_section[item["section"]].append((help_key, item))

    total = sum(len(v) for v in matched_by_section.values())
    if total == 0:
        st.info(f"No topics matched **{search_q}** in the selected scope.")
        return True

    st.markdown("<div class='docs-section-head'>Search Results</div>", unsafe_allow_html=True)
    st.caption(f"{total} result{'s' if total != 1 else ''} grouped by section")
    for section_id, items in matched_by_section.items():
        section_title = SECTION_COPY.get(section_id, {}).get("title", section_id)
        with st.expander(f"{section_title} ({len(items)})", expanded=True):
            for help_key, item in items:
                st.markdown(f"### {_highlight(item['label'], search_q)}")
                if item.get("summary"):
                    st.caption(_highlight(item["summary"], search_q))
                st.markdown(_highlight(item["detail"], search_q))
                if focus_key == help_key:
                    st.caption("Opened from a live UI help chip.")
    st.markdown("---")
    return True


def _render_bottom_actions() -> None:
    st.markdown("<div class='docs-section-head'>Next Actions</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        if st.button("Go to Tomorrow's Picks", key="docs_go_tomorrow", width="stretch"):
            st.session_state["mode"] = "Tomorrow"
            st.session_state["_nav_skip_sync"] = True
            st.rerun()
    with c2:
        if st.button("Go to Backtesting Lab", key="docs_go_lab", width="stretch"):
            st.session_state["mode"] = "Backtest Lab"
            st.session_state["_nav_skip_sync"] = True
            st.rerun()
    with c3:
        if st.button("Open Release History", key="docs_go_changelog", width="stretch"):
            st.session_state["mode"] = "Release History"
            st.session_state["_nav_skip_sync"] = True
            st.rerun()
    with c4:
        st.caption("Source docs live in stock_triggers/docs and data artifacts in stock_triggers/data.")


def render_documentation_page() -> None:
    _ensure_docs_page_css()
    handle_docs_section_query_param()
    focus_key = str(st.session_state.get("docs_focus_key", "") or "").strip()
    focus_item = get_help_item(focus_key) if focus_key else None
    grouped_items: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for help_key, item in HELP_ITEMS.items():
        grouped_items[item["section"]].append((help_key, item))

    _render_focus_banner(focus_item)
    search_q = _render_top_rail(grouped_items)
    active_filter = str(st.session_state.get("docs_section_filter_id", "") or "").strip()

    _render_hero_and_entry_cards()
    _render_quick_map()
    st.markdown("---")

    if _render_search_results(search_q, grouped_items, focus_key, active_filter):
        _render_bottom_actions()
        return

    active_section = str(st.session_state.get("docs_active_section", "") or "")
    section_blocks: list[tuple[str, object]] = [
        ("getting_started", _render_getting_started),
        ("daily_review_flow", _render_daily_review_flow),
        ("case_studies", _render_case_studies),
        ("score_explainer", _render_score_explainer),
        ("pattern_library", _render_pattern_gallery),
        ("enhancers", _render_enhancer_gallery),
        ("risk_catalysts", _render_risk_and_catalysts),
        ("reference", lambda: _render_reference_layer(grouped_items, focus_key, active_filter)),
    ]

    if active_section:
        st.caption(f"Focused section: {active_section.replace('_', ' ').title()}")
        section_blocks = sorted(section_blocks, key=lambda block: block[0] != active_section)

    for idx, (_, render_block) in enumerate(section_blocks):
        render_block()
        if idx < len(section_blocks) - 1:
            st.markdown("---")

    st.markdown("---")
    _render_bottom_actions()