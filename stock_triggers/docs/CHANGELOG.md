# Changelog

This is the complete in-app changelog for Stock Operator from repo inception.

The What's New panel stays intentionally short and recent. This page keeps the longer history in one place so you can trace how the app and pipeline evolved over time.

## 2026-05-04

### feat: background scoring defaults, alerts, and universe tomorrow view
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: feat: background scoring defaults, alerts, and universe tomorrow view.
- Details: Commit list: feat: background scoring defaults, alerts, and universe tomorrow view. Touched areas: repo root, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-05-01

### Add coverage guardrails and refresh signal data
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Add coverage guardrails and refresh signal data.
- Details: Commit list: Add coverage guardrails and refresh signal data. Touched areas: repo root, trigger data/config.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger data/config, keeping your analysis in sync with production.
## 2026-05-01

### Update signals data and script fixes
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Update signals data and script fixes.
- Details: Commit list: Update signals data and script fixes. Touched areas: trigger data/config, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-05-01

### Revert "Expand active signal bypass window from 7 to 90 days"
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Revert "Expand active signal bypass window from 7 to 90 days".
- Details: Commit list: Revert "Expand active signal bypass window from 7 to 90 days". Touched areas: trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger scripts, UI, keeping your analysis in sync with production.
## 2026-05-01

### Expand active signal bypass window from 7 to 90 days
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Expand active signal bypass window from 7 to 90 days.
- Details: Commit list: Expand active signal bypass window from 7 to 90 days. Touched areas: trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger scripts, UI, keeping your analysis in sync with production.
## 2026-05-01

### Update metadata and scripts after consensus signal combining
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 4 milestone commits being pushed to master. Highlights: Populate st_score for all signals using hybrid4 ML model; Show all recent signals (<7 days) regardless of score threshold; Combine consensus signals and apply pattern agreement boost; Update metadata and scripts after consensus signal combining.
- Details: Commit list: Populate st_score for all signals using hybrid4 ML model; Show all recent signals (<7 days) regardless of score threshold; Combine consensus signals and apply pattern agreement boost; Update metadata and scripts after consensus signal combining. Touched areas: trigger data/config, UI, trigger docs, trigger scripts.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, UI, trigger docs, trigger scripts, keeping your analysis in sync with production.
## 2026-05-01

### chore: rename data artifacts and update refresh cadence workflows
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: chore: rename data artifacts and update refresh cadence workflows.
- Details: Commit list: chore: rename data artifacts and update refresh cadence workflows. Touched areas: repo root, trigger data/config, trigger docs, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger data/config, trigger docs, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-05-01

### Deploy all ST models to production; default to hybrid4 with ST Min score of 70
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Deploy all ST models to production; default to hybrid4 with ST Min score of 70.
- Details: Commit list: Deploy all ST models to production; default to hybrid4 with ST Min score of 70. Touched areas: repo root, trigger data/config, UI.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger data/config, UI, keeping your analysis in sync with production.
## 2026-05-01

### Make ST Min score filter warning more actionable
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Make ST Min score filter warning more actionable.
- Details: Commit list: Make ST Min score filter warning more actionable. Touched areas: UI.
- Impact: Your interface is now using the latest improvements for faster insights.
## 2026-05-01

### Fix ST model warning and add filter funnel visibility
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Fix ST model warning and add filter funnel visibility.
- Details: Commit list: Fix ST model warning and add filter funnel visibility. Touched areas: UI.
- Impact: Your interface is now using the latest improvements for faster insights.
## 2026-04-29

### Fix Pattern A scoring merge collision and refresh signal artifacts
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Fix Pattern A scoring merge collision and refresh signal artifacts.
- Details: Commit list: Fix Pattern A scoring merge collision and refresh signal artifacts. Touched areas: trigger data/config, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, UI, keeping your analysis in sync with production.
## 2026-04-28

### Remove weight panels from Tomorrow's Picks
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Remove weight panels from Tomorrow's Picks.
- Details: Commit list: Remove weight panels from Tomorrow's Picks. Touched areas: UI.
- Impact: Your interface is now using the latest improvements for faster insights.
## 2026-04-28

### Apply local updates after rebase
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Apply local updates after rebase.
- Details: Commit list: Apply local updates after rebase. Touched areas: trigger data/config, trigger docs, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, trigger docs, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-27

### Telegram: lower minimum score threshold to 60
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Telegram: lower minimum score threshold to 60.
- Details: Commit list: Telegram: lower minimum score threshold to 60. Touched areas: trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-27

### Telegram: include confluence exit note and buy exit price
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Telegram: include confluence exit note and buy exit price.
- Details: Commit list: Telegram: include confluence exit note and buy exit price. Touched areas: trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-27

### Telegram: threshold 70, short/long term split, exits section
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Telegram: threshold 70, short/long term split, exits section.
- Details: Commit list: Telegram: threshold 70, short/long term split, exits section. Touched areas: trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-27

### Launch hybrid4 ST scoring ensemble across pipeline and UI
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 3 milestone commits being pushed to master. Highlights: Add recency-weighted ST scoring pipeline; Show ST monthly chart per trade; Launch hybrid4 ST scoring ensemble across pipeline and UI.
- Details: Commit list: Add recency-weighted ST scoring pipeline; Show ST monthly chart per trade; Launch hybrid4 ST scoring ensemble across pipeline and UI. Touched areas: repo root, trigger scripts, UI, trigger data/config.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger scripts, UI, trigger data/config, keeping your analysis in sync with production.
## 2026-04-25

### Master updated from 3 unpushed commits
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 3 commits being pushed to master. Highlights: Add recency-weighted ST scoring pipeline; Merge remote-tracking branch 'origin/master'; Update What's New for master push.
- Details: Commit list: Add recency-weighted ST scoring pipeline; Merge remote-tracking branch 'origin/master'; Update What's New for master push. Touched areas: repo root, trigger data/config, trigger scripts, UI, trigger docs.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger data/config, trigger scripts, UI, trigger docs, keeping your analysis in sync with production.
## 2026-04-25

### Master updated from 2 unpushed commits
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 2 commits being pushed to master. Highlights: Add recency-weighted ST scoring pipeline; Merge remote-tracking branch 'origin/master'.
- Details: Commit list: Add recency-weighted ST scoring pipeline; Merge remote-tracking branch 'origin/master'. Touched areas: repo root, trigger data/config, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger data/config, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-23

### Master updated from 2 unpushed commits
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 2 commits being pushed to master. Highlights: Add ST backtesting updates, Markov zone scoring, and reinvest capital mode; Update What's New for master push.
- Details: Commit list: Add ST backtesting updates, Markov zone scoring, and reinvest capital mode; Update What's New for master push. Touched areas: trigger scripts, UI, trigger docs.
- Impact: Both signal logic and interface reflect the latest deployment across trigger scripts, UI, trigger docs, keeping your analysis in sync with production.
## 2026-04-23

### Add ST backtesting updates, Markov zone scoring, and reinvest capital mode
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Add ST backtesting updates, Markov zone scoring, and reinvest capital mode.
- Details: Commit list: Add ST backtesting updates, Markov zone scoring, and reinvest capital mode. Touched areas: trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-22

### Master updated from 3 unpushed commits
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 3 commits being pushed to master. Highlights: fix: long term rescore toggle defaults to ON on fresh load; Update What's New for master push; Update What's New for master push.
- Details: Commit list: fix: long term rescore toggle defaults to ON on fresh load; Update What's New for master push; Update What's New for master push. Touched areas: trigger data/config, trigger docs, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, trigger docs, UI, keeping your analysis in sync with production.
## 2026-04-22

### Master updated from 2 unpushed commits
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 2 commits being pushed to master. Highlights: fix: long term rescore toggle defaults to ON on fresh load; Update What's New for master push.
- Details: Commit list: fix: long term rescore toggle defaults to ON on fresh load; Update What's New for master push. Touched areas: trigger data/config, trigger docs, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, trigger docs, UI, keeping your analysis in sync with production.
## 2026-04-22

### fix: long term rescore toggle defaults to ON on fresh load
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: fix: long term rescore toggle defaults to ON on fresh load.
- Details: Commit list: fix: long term rescore toggle defaults to ON on fresh load. Touched areas: trigger data/config, trigger docs, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, trigger docs, UI, keeping your analysis in sync with production.
## 2026-04-22

### Master updated from 2 unpushed commits
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Auto-captured from 2 commits being pushed to master. Highlights: Update catalyst zone analysis, scripts, and docs; Update What's New for master push.
- Details: Commit list: Update catalyst zone analysis, scripts, and docs; Update What's New for master push. Touched areas: repo root, trigger data/config, trigger docs, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger data/config, trigger docs, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-22

### Update catalyst zone analysis, scripts, and docs
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Update catalyst zone analysis, scripts, and docs.
- Details: Commit list: Update catalyst zone analysis, scripts, and docs. Touched areas: repo root, trigger data/config, trigger docs, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across repo root, trigger data/config, trigger docs, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-22

### Phase 2: Add catalyst feature framework with market regimes, event windows, and score adjustments
- Auto-generated from commits pushed to `refs/heads/master`.
- Summary: Phase 2: Add catalyst feature framework with market regimes, event windows, and score adjustments.
- Details: Commit list: Phase 2: Add catalyst feature framework with market regimes, event windows, and score adjustments. Touched areas: trigger data/config, trigger docs, trigger scripts, UI.
- Impact: Both signal logic and interface reflect the latest deployment across trigger data/config, trigger docs, trigger scripts, UI, keeping your analysis in sync with production.
## 2026-04-17

### Recency-weighted training and stop-risk ranking controls

- Added recency-weighted training updates so the learned models can lean harder on newer signal history.
- Moved stop-risk pressure into ranking control flow instead of leaving it as a side metric.
- Exposed live Long Term controls for the stop-risk curve, thresholds, and hard gate so ranking behavior can be tuned and inspected in-app.

## 2026-04-15

### Persistent sign-in and automated What's New updates

- Added persistent Google auth sessions so production users do not need to re-authenticate as often.
- Added a pre-push hook that can auto-refresh the rolling What's New feed on pushes to master.
- Kept the release-note generation flow in repo so the in-app summary can stay aligned with shipped changes.

## 2026-04-14

### Learned candle context, coverage analysis, and production polish

- Switched candle scoring toward learned behavior by pattern family instead of relying on one global rule.
- Added coverage analysis and refreshed backtest defaults so the lab gives a broader view of what the engine is seeing.
- Introduced the first in-app What's New panel and release-note tracking data.
- Added production auth hardening and visible build markers for deployed app clarity.

## 2026-04-11 to 2026-04-12

### Documentation became a first-class app surface

- Added a dedicated Documentation page inside the Streamlit app.
- Added help chips across the UI so live screens can jump directly to deeper explanations.
- Expanded documentation with search, per-pattern deep dives, score-formula explanations, candle-enhancer diagrams, and pattern-map visuals.
- Fixed programmatic navigation edge cases so internal docs jumps do not get overwritten by navbar sync.

## 2026-04-04 to 2026-04-06

### Workflow expansion and Pattern G activation

- Activated Pattern G in signal generation and published the related signal artifacts.
- Added backfill support to the incremental workflow and reduced unnecessary retraining work in refresh paths.
- Refactored refresh pipeline sequencing and restored more reliable long-history behavior.
- Added real historical pattern charts to documentation and tightened mobile layout behavior.

## 2026-04-01 to 2026-04-03

### Pattern-family scoring and calibrated stop-risk pipeline

- Added calibrated stop-risk and learned penalty modeling so signal ranking could reflect downside pressure more explicitly.
- Extended scoring with pattern-family contribution logic instead of treating every family identically.
- Regenerated training artifacts and candle-weight workflows to keep saved outputs aligned with the newer model logic.
- Added confirmed hammer analysis as part of the candle-enhancer work.

## 2026-03-28 to 2026-03-30

### Candle enhancers and richer scoring inputs

- Added multiple bullish candlestick enhancers and later widened the enhancer total when history supported it.
- Improved scoring with RSI integration, moving-average slope metrics, and hold-to-target logic.
- Extended price-history retrieval and refreshed stock signals and scores with broader coverage.
- Improved Tomorrow's Picks presentation with status indicators, branding, and clearer signal display.

## 2026-03-24 to 2026-03-26

### Core ranking layer and Tomorrow's Picks refinement

- Added stock-health score generation and strengthened the trigger-scoring pipeline.
- Brought RSI calculation into the signal-scoring path.
- Refined the Tomorrow's Picks interface so the daily shortlist became easier to inspect quickly.

## 2026-03-16 to 2026-03-23

### Stock Triggers app foundation

- Added the initial stock-triggers documentation and Streamlit UI for reviewing Pattern A signals.
- Added Telegram integration and deployment notes for the daily trigger workflow.
- Expanded the data universe and pipeline plumbing used by the trigger engine.
- Continued iterating on the daily shortlist UI and operational workflow cleanup.

## 2026-01-03

### Repo inception point

- Added the mutual funds selector script, which is the earliest tracked commit in this repository.
- This marks the starting point before the repo grew into the broader Stock Operator and Stock Triggers workspace.

## Notes

- This first complete changelog is intentionally milestone-based instead of commit-by-commit.
- Recent day-to-day updates still appear in the rolling What's New panel.
- Future history can keep growing here without exposing raw git history inside the app UI.