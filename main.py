"""Forecast GTA VI's launch-day Twitch viewers hour by hour (gta6-twitch-ccu-forecast-01).

Method (fitted live from the case file's own comparables, no hardcoded curves):
  1. Each comparable is a past big launch with 24 hourly viewer counts for its
     UTC release day.  Hour 1's print is mostly a timezone-composition signal
     (which regions had unlocked by 00:00 UTC), not a scale signal: across the
     comparables, the log of a later hour's viewers regresses on log(hour-1)
     with a slope near 0 at peak hours, while the ratio to hour 1 varies by
     more than an order of magnitude.  So the median forecast blends three
     predictors of log viewers at each hour h = 2..24:
       a. per-hour OLS of log(v_h) on log(hour-1 print), fitted on ALL
          comparables with a complete launch day;
       b. the same OLS fitted only on "target-like" comparables: rolling
          midnight-local unlocks, tier comparable, with no global unlock
          before/at UTC midnight (the two launches that had one peaked in
          hour 1 and then declined all day -- a shape GTA VI, a rolling
          console-only launch whose US unlocks land in hours 6/9, cannot
          have);
       c. the median launch-day ratio curve of the 8 target-like comparables
          nearest in log(hour-1 print) (KNN median ratio).
     Weights: 0.25 (a) + 0.25 (b) + 0.50 (c).  Chosen by leave-one-out
     validation over the comparables and by the 4 practice cases (see README).
  2. Quantile spreads come from honest leave-one-out residuals: each
     target-like comparable is forecast by the same blend fitted without it,
     and the per-hour empirical quantiles of those residuals (x1.3, because
     with only ~9-13 target-like launches the raw 5%/95% order statistics
     under-cover) are added to the log median.

Everything is deterministic (sorted iteration, no randomness), standard
library only, no network.  Runtime is well under a second per case.
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed here, but keep the
# starter's convention).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
HOURS = list(range(2, 25))
# Blend weights for the three median predictors (see module docstring).
W_OLS_ALL, W_OLS_TL, W_KNN_TL = 0.25, 0.25, 0.50
KNN_K = 8
# Spread inflation for the leave-one-out residual quantiles.
SPREAD_MULT = 1.3


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up (the starter's rounding)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def quantile_sorted(values, q):
    """Linear-interpolation quantile of a non-empty list."""
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def ols2(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b). Falls back to the mean."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def launch_day(comp):
    """The comparable's 24 hourly viewer counts for its UTC release day.

    Returns None when the day is incomplete, has a non-positive count, or the
    entry is not a launch at all (GTA VI's own pre-release category history).
    """
    name = str(comp.get("name", ""))
    unlock = str(comp.get("unlock", ""))
    if "not a launch" in unlock.lower() or "before release" in name.lower():
        return None
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        try:
            by_hour[parse_time(row["from"])] = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
    hours = []
    for h in range(24):
        v = by_hour.get(day_start + timedelta(hours=h))
        if v is None or not isinstance(v, (int, float)) or v <= 0:
            return None
        hours.append(float(v))
    return hours


def is_rolling(comp):
    u = str(comp.get("unlock", "")).lower()
    return "midnight local" in u or "rolling" in u


def tier(comp):
    u = str(comp.get("unlock", ""))
    if "early-access-contaminated" in u:
        return "ea"
    if "Tier: weak" in u:
        return "weak"
    return "comp"


def has_early_global_unlock(comp):
    """True when some platform unlocked worldwide before/at UTC midnight of
    the release day, so hour 1 already held the whole world (ELDEN RING's PC
    unlock at 23:00Z the day before; Cyberpunk 2077's simultaneous midnight
    GMT unlock). Those launches peak in hour 1 and decline -- a shape a
    rolling-midnight launch cannot have."""
    u = str(comp.get("unlock", "")).lower()
    if "simultaneous midnight gmt" in u:
        return True
    return "global unlock" in u and ("23:00" in u or "00:00" in u)


def is_target_like(comp):
    """A rolling midnight-local launch, tier comparable, with no early global
    unlock -- the launches shaped like GTA VI's release day."""
    return tier(comp) == "comp" and is_rolling(comp) and not has_early_global_unlock(comp)


def fit_ols(rows, p1):
    """Per-hour OLS of log viewers on log hour-1 print over `rows`.

    `rows` is a list of (comparable, 24 hourly counts). Returns a dict
    hour -> predicted log viewers at hour-1 print p1."""
    lp1 = math.log(max(1.0, p1))
    out = {}
    for h in HOURS:
        xs = [math.log(r[0]) for _, r in rows]
        ys = [math.log(r[h - 1]) for _, r in rows]
        a, b = ols2(xs, ys)
        out[h] = a + b * lp1
    return out


def fit_knn(rows, p1, k=KNN_K):
    """Median launch-day ratio curve of the k nearest rows in log hour-1.

    Returns a dict hour -> predicted log viewers at hour-1 print p1."""
    lp1 = math.log(max(1.0, p1))
    nearest = sorted(rows, key=lambda cr: (abs(math.log(cr[1][0]) - lp1), cr[0].get("name", "")))[:k]
    if not nearest:
        return None
    out = {}
    for h in HOURS:
        out[h] = lp1 + statistics.median([math.log(r[h - 1] / r[0]) for _, r in nearest])
    return out


def median_model(rows_all, rows_tl, p1):
    """Blended log-median forecast for hours 2..24 (see module docstring)."""
    ols_all = fit_ols(rows_all, p1)
    ols_tl = fit_ols(rows_tl, p1) if rows_tl else ols_all
    knn_tl = fit_knn(rows_tl, p1) if rows_tl else None
    if knn_tl is None:
        knn_tl = fit_knn(rows_all, p1)
    if knn_tl is None:
        return ols_all
    return {
        h: W_OLS_ALL * ols_all[h] + W_OLS_TL * ols_tl[h] + W_KNN_TL * knn_tl[h]
        for h in HOURS
    }


def loo_residuals(rows_all, rows_tl):
    """Leave-one-out residuals of the median model over the target-like rows.

    Each target-like comparable is forecast by the blend fitted without it;
    the residual is log(actual) - log(forecast) per hour. Falls back to all
    rows when there are too few target-like ones to leave one out from."""
    pool = rows_tl if len(rows_tl) >= 5 else rows_all
    res = {h: [] for h in HOURS}
    for comp, hours in pool:
        train_all = [(c, r) for c, r in rows_all if c is not comp]
        train_tl = [(c, r) for c, r in rows_tl if c is not comp]
        med = median_model(train_all, train_tl, hours[0])
        for h in HOURS:
            res[h].append(math.log(hours[h - 1]) - med[h])
    return res


def baseline_quantiles(p1, h):
    """The house persistence baseline (starter), used only as a fallback when
    a case carries no usable comparables."""
    z = [-1.645, -0.674, 0, 0.674, 1.645]
    return [round2(max(1.0, p1) * math.exp(zz * 0.15 * math.sqrt(h - 1))) for zz in z]


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    rows_all = []
    for comp in case.get("comparables") or []:
        hours = launch_day(comp)
        if hours is not None:
            rows_all.append((comp, hours))
    rows_tl = [(c, r) for c, r in rows_all if is_target_like(c)]

    if not rows_all:
        return {"hours": [{"hour": h, "q": baseline_quantiles(p1, h)} for h in HOURS]}

    med = median_model(rows_all, rows_tl, p1)
    res = loo_residuals(rows_all, rows_tl)

    out_hours = []
    for h in HOURS:
        qs = []
        for tau in LEVELS:
            offset = SPREAD_MULT * quantile_sorted(res[h], tau) if res[h] else 0.0
            q = math.exp(med[h] + offset)
            q = min(max(q, 1.0), 1e9)
            qs.append(round2(q))
        for i in range(1, 5):
            qs[i] = max(qs[i], qs[i - 1])
        out_hours.append({"hour": h, "q": qs})
    return {"hours": out_hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
