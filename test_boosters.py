#!/usr/bin/env python
"""Quick test to verify booster logic is working."""

import pandas as pd
import numpy as np

df = pd.read_csv('stock_triggers/data/st_signals_all_patterns.csv')
df['st_score'] = pd.to_numeric(df['st_score'], errors='coerce')
df['st_score_pre_model'] = pd.to_numeric(df['st_score_pre_model'], errors='coerce')

print("\n" + "=" * 70)
print("ST SCORE BOOSTER IMPLEMENTATION VERIFICATION")
print("=" * 70)

# Overall distribution
st = df['st_score'].dropna()
print(f"\nScore Distribution (All {len(st)} signals):")
print(f"  Mean:      {st.mean():.2f}")
print(f"  Median:    {st.median():.2f}")
print(f"  Max:       {st.max():.2f}")
print(f"  Min:       {st.min():.2f}")

print(f"\nHigh-Score Buckets:")
for thresh in [70, 72, 75, 80, 85, 90]:
    cnt = (st >= thresh).sum()
    pct = 100 * cnt / len(st)
    print(f"  >= {thresh}: {cnt:4d} ({pct:5.2f}%)")

# Analyze boosters on specific signal
top_idx = df['st_score'].idxmax()
top = df.loc[top_idx]
print(f"\n" + "-" * 70)
print("TOP SIGNAL ANALYSIS (Highest ST Score)")
print("-" * 70)
print(f"Date: {top['signal_date']}, Ticker: {top['ticker']}")
print(f"Pattern Family: {top['pattern_family']}, Consensus: {int(top.get('consensus_count', 1))}")

print(f"\nWould Apply Booster 1 (Markov Confidence)?")
p_cont = float(top.get('markov_p_continuation', 0))
p_adv = float(top.get('markov_p_adverse', 0))
state = str(top.get('markov_state', ''))
if p_cont >= 0.65 and p_adv <= 0.20:
    print(f"  ✓ YES: P(cont)={p_cont:.3f} >= 0.65 AND P(adv)={p_adv:.3f} <= 0.20 → +8 pts")
elif state == "fresh_breakout" and p_cont >= 0.60:
    print(f"  ✓ YES: Fresh breakout with P(cont)={p_cont:.3f} → +5-7 pts")
elif state == "constructive_trend" and p_cont >= 0.58:
    print(f"  ✓ YES: Constructive trend with P(cont)={p_cont:.3f} → +3 pts")
else:
    print(f"  ✗ NO: State={state}, P(cont)={p_cont:.3f}, P(adv)={p_adv:.3f}")

print(f"\nWould Apply Booster 2 (Consensus)?")
consensus = int(top.get('consensus_count', 1))
if consensus >= 3:
    print(f"  ✓ YES: consensus_count={consensus} >= 3 → +6 pts")
elif consensus == 2:
    print(f"  ✓ YES: consensus_count={consensus} == 2 → +3 pts")
else:
    print(f"  ✗ NO: consensus_count={consensus} < 2")

print(f"\nWould Apply Booster 3 (Entry Quality)?")
gap = float(top.get('feature_gap_pct', 0))
exh = float(top.get('feature_exhaustion_risk', 0))
clh = float(top.get('feature_close_vs_prev_high_pct', 0))
bon3 = 0
if -0.5 <= gap <= 1.5:
    print(f"  ✓ Clean entry: gap_pct={gap:.2f}% in [-0.5,1.5] → +3 pts")
    bon3 += 3
if exh < 8.0:
    print(f"  ✓ Not extended: exhaustion={exh:.1f} < 8 → +2 pts")
    bon3 += 2
if clh >= -1.0:
    print(f"  ✓ Near highs: close_vs_prev_high={clh:.2f}% >= -1 → +1 pt")
    bon3 += 1
if bon3 == 0:
    print(f"  ✗ NO entry quality bonus (gap={gap:.2f}, exh={exh:.1f}, clh={clh:.2f})")
print(f"  Total from entry quality: {bon3} pts")

print(f"\nWould Apply Booster 4 (Volatility Regime)?")
regime = float(top.get('regime_median_ret_20d_pct', 0))
pct_sma = float(top.get('regime_pct_above_sma50', 50))
atr_ratio = float(top.get('feature_range_vs_atr', 1.0))
bon4 = 0
if regime > 1.5 and atr_ratio < 1.2:
    print(f"  ✓ Trending with controlled vol: regime_ret={regime:.2f}% > 1.5 AND atr_ratio={atr_ratio:.2f} < 1.2 → +4 pts")
    bon4 += 4
if pct_sma > 75:
    print(f"  ✓ Healthy uptrend: pct_sma50={pct_sma:.1f}% > 75 → +2 pts")
    bon4 += 2
if regime < -1.5:
    bon4 -= 2
if bon4 == 0:
    print(f"  ✗ NO volatility regime bonus (regime={regime:.2f}, pct_sma={pct_sma:.1f}, atr={atr_ratio:.2f})")
print(f"  Total from volatility regime: {bon4} pts")

print(f"\nScore Progression:")
print(f"  st_score_pre_model: {top.get('st_score_pre_model', 0):.2f}")
print(f"  st_score (final):   {top.get('st_score', 0):.2f}")
print(f"  Boost applied:      {top.get('st_score', 0) - top.get('st_score_pre_model', 0):.2f}")

print("\n" + "=" * 70)
