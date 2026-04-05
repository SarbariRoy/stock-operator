# Patterns A To G

This file explains the actual pattern families that exist in the current codebase.

The easiest way to read this is:

- A and G are breakout-style ideas.
- B is a pullback idea.
- C and D are indicator cross ideas.
- E is a squeeze idea.
- F is a reclaim idea.

## Pattern map

```mermaid
flowchart TD
    A[Pattern A<br/>Breakout with trend and volume]
    B[Pattern B<br/>Pullback and rebound]
    C[Pattern C<br/>MACD bullish crossover]
    D[Pattern D<br/>RSI oversold bounce]
    E[Pattern E<br/>Bollinger squeeze breakout]
    F[Pattern F<br/>VWAP reclaim]
    G[Pattern G<br/>VCP breakout]
```

Charts below are real historical examples taken from `prices_eod.csv` and `signals_all_patterns.csv`.

Each chart uses actual OHLC candles plus indicator overlays. Scale differs from chart to chart.

## Shared idea across the whole system

Every pattern is looking for a bullish setup, but the code does not stop there.

After a pattern fires, the engine still scores the row based on trend, setup quality, volume, risk, RSI context, MA slope, and learned family strength.

So a signal is really:

$$
    ext{signal} = \text{pattern trigger} + \text{quality score}
$$

## A. Pattern A: Trend breakout with volume

The stock is already in an uptrend, and today it pushes above a recent closing high with strong volume.

Example chart:

![Pattern A example breakout chart](assets/pattern-charts/pattern-a-breakout.svg)

Historical example: ADANIPOWER.NS on 2025-09-19.

Main conditions:

- SMA50 > SMA200
- close > SMA50
- close > SMA200
- close > previous N-day high close
- volume above the 20-day average by a chosen multiplier

Typical use:

- momentum continuation
- higher-quality “already strong” names

Important knobs:

- breakout_days
- volume_multiplier
- stop_pct
- optional ATR or structure stop modes in the UI detector path

## B. Pattern B: Pullback and rebound near SMA20

The stock is still in an uptrend, but instead of breaking out, it has pulled back toward the 20-day average and started bouncing again.

Example chart:

![Pattern B example pullback chart](assets/pattern-charts/pattern-b-pullback.svg)

Historical example: BHARTIARTL.NS on 2025-12-30.

Main conditions:

- SMA50 > SMA200
- close still above SMA50
- close near SMA20
- today closes above yesterday by a minimum rebound amount
- volume is at least mildly supportive

Why it exists:

- catches continuation entries before a big breakout bar
- gives a less extended entry than Pattern A sometimes does

## C. Pattern C: MACD bullish crossover

The MACD line was below the signal line, and now it has crossed above it while the broader trend is still healthy.

Example chart:

![Pattern C example MACD crossover chart](assets/pattern-charts/pattern-c-macd.svg)

Historical example: ONGC.NS on 2026-03-27.

Main conditions:

- SMA50 > SMA200
- MACD previous <= signal previous
- MACD now > signal now
- volume above a relaxed threshold

Why it exists:

- gives a momentum re-acceleration type entry
- often catches trend resumption before price makes a dramatic breakout

## D. Pattern D: RSI oversold bounce
The stock was recently oversold on RSI, is now recovering back above the threshold, and price confirms that buyers actually regained control while the broader trend is still up.

Example chart:

![Pattern D example RSI bounce chart](assets/pattern-charts/pattern-d-rsi.svg)

Historical example: PIDILITIND.NS on 2025-08-06.

Main conditions:

- SMA50 > SMA200
- RSI was below threshold within the last 1 to 3 bars
- RSI today is back above threshold, has reclaimed at least 35, and is improving by a meaningful amount versus the prior bar
- price confirms via a close above the previous day's high or a reclaim back above SMA20
- reversal-day volume is at or above the 20-day average
- close is still near the recent 10-bar swing low rather than already stretched far off support

Why it exists:

- tries to catch “washout then bounce” behavior inside a bigger uptrend while filtering out weak mechanical RSI crosses

Score boosters:

- RSI recovery through 40 adds extra setup strength
- reversal-day volume meaningfully above average boosts the volume component further
- positive SMA50 slope still adds the shared moving-average slope bonus
- rebounds that happen very close to the recent swing low get extra setup credit

Important note:

In the current learned weight file, D is one of the weaker families, so it may carry little or no family bonus.

## E. Pattern E: Bollinger squeeze breakout

The stock has gone quiet, volatility has tightened, and then price breaks out above the upper Bollinger Band.

Example chart:

![Pattern E example squeeze breakout chart](assets/pattern-charts/pattern-e-squeeze.svg)

Historical example: BEL.NS on 2025-06-20.

Main conditions:

- SMA50 > SMA200
- Bollinger Band width is at or near a recent low
- close breaks above the upper band
- volume is above average

Why it exists:

- looks for compression followed by expansion

## F. Pattern F: VWAP reclaim

On end-of-day data, the stock was trading below a rolling VWAP approximation and then closes back above it on stronger volume.

Example chart:

![Pattern F example VWAP reclaim chart](assets/pattern-charts/pattern-f-vwap.svg)

Historical example: COALINDIA.NS on 2026-03-04.

Main conditions:

- SMA50 > SMA200
- previous close <= rolling VWAP approximation
- current close > rolling VWAP approximation
- volume spike

Why it exists:

- looks for regained control after short-term weakness
- often shows up as a crisp “buyers took back the line” setup

## G. Pattern G: VCP breakout

The stock forms a volatility contraction pattern, with several pullbacks getting shallower, then breaks out above resistance.

Example chart:

![Pattern G example VCP breakout chart](assets/pattern-charts/pattern-g-vcp.svg)

Historical example: BRITANNIA.NS on 2024-09-12.

Main conditions:

- uptrend already in place
- at least three pullbacks can be found from pivots
- each pullback is shallower than the one before it
- breakout above recent resistance
- volume support
- volume dry-up during the contraction phase

Why it exists:

- this is the most structure-heavy pattern in the set
- it is trying to identify coiled breakouts, not just “price went up today”

## The score formula used after a pattern fires

The base score is currently:

$$
0.20T + 0.20S + 0.13V + 0.14R + 0.03I
$$

Then the engine can add:

$$
B_{\text{ma}} + B_{\text{pattern}} + B_{\text{consensus}}
$$

and finally clip the result into the 0 to 100 range.

## Why learned pattern weights matter

The project now stores a learned family bonus in pattern_weights.json.

That means a strong family can contribute more to the final score than a weak family.

Very roughly:

$$
B_{\text{pattern}} = \frac{\text{family score}}{100} \times 30
$$

So if a family has a historical score of 80 out of 100, it can contribute close to 24 of the possible 30 family points.

## Practical reading guide

If you want a simple working interpretation:

- Pattern A: classic strong-stock breakout
- Pattern B: pullback continuation
- Pattern C: momentum crossover
- Pattern D: oversold rebound
- Pattern E: squeeze expansion
- Pattern F: reclaim and go
- Pattern G: coiled breakout

And if two families trigger together on the same stock and date, that is usually worth extra attention because the system can add a consensus bonus too.
