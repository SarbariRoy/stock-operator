# Phase 1: Catalyst Feature Contract & Schema

Locked feature definition for market-regime and company-event catalyst factors. All downstream code (signal enrichment, penalty fitting, UI filtering) must conform to this contract.

**Date**: 2026-04-21  
**Status**: Locked (Phase 1 gate requirement)

---

## Market-Regime Factors

Sourced from `external_factors.csv`. Computed daily; no look-ahead. Joined onto signal history by `signal_date`.

| Column | Type | Source | Definition | Nullable | Units |
|--------|------|--------|-----------|----------|-------|
| `vix_change_1d_pct` | float | Yahoo (^INDIAVIX) | (today_close − yesterday_close) / yesterday_close × 100 | Yes | percent |
| `usdinr_ret_5d_pct` | float | Yahoo (INR=X) | 5-day log return of USD/INR | Yes | percent |
| `brent_ret_5d_pct` | float | Yahoo (BZ=F or CL=F) | 5-day log return of Brent/WTI | Yes | percent |
| `fii_dii_net_cr` | float | NSE API | Net FII/DII flows in crores | Yes | crore INR |

### Derived Regimes (computed in signal enrichment)

| Column | Type | Derivation | Definition |
|--------|------|-----------|-----------|
| `vix_regime_high` | bool | `vix_close > percentile_75` | True if India VIX in top quartile for recent 60 days |
| `flow_regime_weak` | bool | `fii_dii_net_cr < percentile_25` | True if institutional inflow in bottom quartile for recent 60 days |
| `energy_regime_shock` | bool | `brent_ret_5d_pct < −5` | True if oil down >5% in 5 days |

---

## Company Event Factors

Sourced from `event_calendar.csv` (new artifact). Normalized to trading-day windows; computed daily by ticker; joined onto signal history by `signal_date` and `ticker`.

### Event Window Flags (binary, default False)

| Column | Type | Definition | Window |
|--------|------|-----------|--------|
| `within_earnings_pre` | bool | Signal fired in ±3 trading days before/around earnings | ±3 trading days |
| `within_earnings_post` | bool | Signal fired in ±3 trading days after earnings | ±3 trading days |
| `within_dividend_ex_day` | bool | Signal fired in ±3 trading days of ex-dividend date | ±3 trading days |
| `within_dividend_record` | bool | Signal fired in ±3 trading days of record date | ±3 trading days |

### Event Interaction Flags

| Column | Type | Definition |
|--------|------|-----------|
| `post_event_gap_risk` | bool | True if signal follows earnings/dividend within 1 trading day AND prior close to current open gap > 2% |
| `event_proximity_score` | float | Normalized distance to nearest event (0=at event, 1=far): max(0, 1 − min_distance_days / 5) |

---

## Constraints & Governance

1. **Temporal**: All catalyst values must be deterministic w.r.t. `signal_date`. No future information leakage.
2. **Uniqueness**: `external_factors.csv` must have unique Date; `event_calendar.csv` must have unique (ticker, event_date).
3. **Coverage**: Join success ≥95% for market factors, ≥90% for event flags.
4. **Scope**: Market factors are global; event factors are ticker-specific.
5. **Updates**: Market factors refreshed daily; event calendar refreshed weekly or post-earnings-season.

---

## Integration Points

1. **Signal enrichment**: Attach all catalyst columns to signal history via signal_date join (market) and signal_date + ticker join (events).
2. **Penalty fitting**: Include catalyst columns as optional features in penalty model training with feature-selection gates.
3. **UI filtering**: Expose three toggles: baseline (no catalyst), market-only, combined (market + events).
4. **Acceptance gating**: Verify coverage gates before proceeding to Phase 2.

---

## Failure Modes & Recovery

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Market factor column missing | Join produces all-NA values | Fallback to baseline mode; skip that catalyst factor |
| Event calendar stale (no recent updates) | Coverage <90% on recent signals | Use last-known good snapshot; alert operator |
| Look-ahead leakage detected | Spot-check: signal_date event_date comparisons | Rollback; audit data pipeline; refreeze |
| Duplicate rows introduce | Data quality gate fails | Deduplicate; recompute; validate before retry |
