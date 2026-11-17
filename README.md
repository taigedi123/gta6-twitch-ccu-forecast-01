# GTA VI launch-day Twitch viewers, hour by hour

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/taigedi123/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/taigedi123/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for the [make.fun bounty "Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) ($250 in USDC).

`main.py` (Python 3.13, standard library only, no network, deterministic) reads
one case file `/input/<id>.in` — everything recorded up to the end of release
day's hour 1, that hour's print included — and writes `/output/<id>.out` with
five quantiles (5%, 25%, 50%, 75%, 95%) of Twitch concurrent viewers in the
Grand Theft Auto VI category for each of hours 2–24 of 2026-11-19 (UTC).

## Method

The forecaster is fitted live from the case file's own comparables (past big
launches' hourly Twitch curves), so nothing about GTA VI's scale is hardcoded:

1. **Hour 1 is a timezone signal, not a scale signal.** Across the
   comparables, a later hour's log viewers regresses on log(hour-1 print)
   with a slope near 0 at peak hours, while the ratio to hour 1 varies by
   more than an order of magnitude — hour 1 mostly says which regions had
   unlocked by 00:00 UTC, not how big the launch is.
2. **Median = a blend of three predictors of log viewers per hour:**
   per-hour OLS on log(hour-1) fitted on all comparables with a complete
   launch day (weight 0.25); the same OLS fitted only on *target-like*
   comparables (weight 0.25); and the median launch-day ratio curve of the
   8 target-like comparables nearest in log(hour-1) (weight 0.50).
   *Target-like* means a rolling midnight-local unlock, tier comparable, with
   no global unlock before/at UTC midnight. The two launches that had one
   (ELDEN RING's PC unlock at 23:00Z the day before; Cyberpunk 2077's
   simultaneous midnight GMT unlock) peaked in hour 1 and declined all day —
   a shape a rolling console-only launch whose US unlocks land in hours 6/9
   cannot have — so they are excluded from the target-like set.
3. **Spreads from honest leave-one-out residuals:** each target-like
   comparable is forecast by the same blend fitted without it; the per-hour
   empirical quantiles of those residuals (×1.3, because with only ~9–13
   target-like launches the raw 5%/95% order statistics under-cover) are
   added to the log median.

## Validation

Practice answers were fetched from SullyGnome exactly as the kit's
`input/NOTICE.txt` documents (4 launches × the GameViewers dataset only) and
scored with the contest's pinball loss on the log scale:

| case | loss | typical miss | 90% coverage |
| ---- | ---- | ------------ | ------------ |
| p00 Marvel's Wolverine | 0.237 | 48.0% | 0.96 |
| p01 Red Dead Redemption 2 | 0.143 | 24.7% | 1.00 |
| p02 God of War Ragnarök | 0.137 | 17.4% | 0.96 |
| p03 Marvel's Spider-Man 2 | 0.203 | 34.1% | 0.87 |
| **mean** | **0.180** | — | **0.95** |

House baseline (starter) on the same cases: **0.387** loss, 82.8% typical
miss, 0.78 coverage. Leave-one-out over the 9 target-like comparables
(each forecast without itself, spreads from the other 8): **0.279** loss,
0.83 coverage. Leave-one-out over all 37 comparables is much worse (0.62)
because it includes simultaneous-global launches whose shape this model
deliberately does not fit — GTA VI is a rolling launch, so the target-like
figure is the relevant one.

Runtime is ~0.05 s per case; two runs are byte-identical.

## Usage

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
