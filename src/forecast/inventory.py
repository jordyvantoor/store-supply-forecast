"""Inventory simulation: turn P90 forecasts into stocking decisions and score them.

Policy (CLAUDE.md): weekly review, 1-week lead time, lost sales, starting
inventory = the first order-up-to level.

Timeline for one item-store (weeks are relative to the first review, week 0):
- End of week t (the review): order-up-to level S_t = P90 for week t+1 plus
  P90 for week t+2 (lead time 1 + review period 1). Order
  q = max(0, S_t - on hand - on order). It arrives at the start of week t+2.
- Week t+1: the arrival due that week is added, then demand is served from on
  hand. Demand that can't be served is lost (no backorders).

Simplifications, stated in the README:
- Summing two weekly P90s overstates the true 2-week P90 (peaks rarely line
  up), so this policy carries a bit more safety stock than it needs.
- "Demand" is historical sales. Sales were themselves limited by the stock the
  store had, so true demand was at least this high.

Run with:  python -m forecast.inventory   (needs models/backtest_forecasts.parquet)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast.backtest import MODELS_DIR, REPORTS_DIR, SERIES, update_metrics

LEAD_TIME = 1
POLICIES = {"model": "p90", "baseline": "baseline_p90"}
K_GRID = np.round(np.arange(0.5, 2.0001, 0.05), 2)
TARGET_FILL_RATES = (0.95, 0.96, 0.97, 0.98)


def simulate(demand: np.ndarray, order_up_to: np.ndarray, lead_time: int = LEAD_TIME) -> dict:
    """Simulate many item-stores at once.

    demand:      (n_series, T) demand in weeks 1..T
    order_up_to: (n_series, R) S at the reviews at the end of weeks 0..R-1 (R <= T)
    Returns per series x week arrays: sold, lost, end_on_hand, stockout, and
    `arrivals` (n_series, T + lead_time + 2): units arriving at the start of
    each week, so arrivals[:, T + 1] is what is already on its way for the
    week after the simulation ends.
    """
    n, T = demand.shape
    on_hand = order_up_to[:, 0].astype(float).copy()   # start full: on hand = S_0
    arrivals = np.zeros((n, T + lead_time + 2))        # indexed by week
    sold = np.zeros((n, T))
    end_on_hand = np.zeros((n, T))

    for t in range(1, T + 1):
        on_hand += arrivals[:, t]
        sold[:, t - 1] = np.minimum(demand[:, t - 1], on_hand)
        on_hand -= sold[:, t - 1]
        end_on_hand[:, t - 1] = on_hand
        if t < order_up_to.shape[1]:                   # review at the end of week t
            on_order = arrivals[:, t + 1:].sum(axis=1)
            q = np.maximum(0.0, order_up_to[:, t] - on_hand - on_order)
            arrivals[:, t + 1 + lead_time] += q

    lost = demand - sold
    return {"sold": sold, "lost": lost, "end_on_hand": end_on_hand, "stockout": lost > 0,
            "arrivals": arrivals}


def summarize(sim: dict, demand: np.ndarray) -> dict:
    """Unit fill rate, average on-hand units per item-store-week, stockout share."""
    return {
        "fill_rate": float(sim["sold"].sum() / demand.sum()),
        "avg_on_hand": float(sim["end_on_hand"].mean()),
        "stockout_week_share": float(sim["stockout"].mean()),
        "lost_units": float(sim["lost"].sum()),
    }


def demand_matrix(fc: pd.DataFrame, weeks: list[int]) -> pd.DataFrame:
    """Actual sales per series (rows) for the given weeks (columns).

    Taken from the h=1 rows: the actual of origin t-1 is the sales of week t.
    """
    h1 = fc[(fc["h"] == 1) & fc["target_idx"].isin(weeks)]
    return h1.pivot_table(index=SERIES, columns="target_idx", values="actual",
                          observed=True)[weeks]


def order_up_to_matrix(fc: pd.DataFrame, reviews: list[int], p90_col: str) -> pd.DataFrame:
    """S per series (rows) at each review week (columns): P90(h=1) + P90(h=2) made there."""
    rows = fc[fc["h"].isin([1, 2]) & fc["origin_idx"].isin(reviews)]
    return rows.pivot_table(index=SERIES, columns="origin_idx", values=p90_col,
                            aggfunc="sum", observed=True)[reviews]


def policy_inputs(fc: pd.DataFrame, first_review: int, last_week: int,
                  p90_col: str) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """Demand and order-up-to arrays from the backtest forecasts.

    Demand weeks: first_review+1 .. last_week. Reviews: first_review ..
    last_week-2 (a review's h=2 target must still be inside the data).
    """
    demand = demand_matrix(fc, list(range(first_review + 1, last_week + 1)))
    s = order_up_to_matrix(fc, list(range(first_review, last_week - 1)), p90_col)
    s = s.reindex(demand.index)
    if demand.isna().any().any() or s.isna().any().any():
        raise ValueError("Missing forecasts or actuals in the simulation window")
    return demand.to_numpy(float), s.to_numpy(float), demand.index


def tradeoff_curve(demand: np.ndarray, s: np.ndarray, ks=K_GRID) -> pd.DataFrame:
    """Fill rate and inventory when every order-up-to level is scaled by k."""
    rows = [{"k": k, **summarize(simulate(demand, s * k), demand)} for k in ks]
    return pd.DataFrame(rows)


def inventory_at_fill_rate(curve: pd.DataFrame, fill_rate: float) -> float:
    """Average on-hand needed to reach `fill_rate`, interpolated along a k-curve."""
    c = curve.sort_values("fill_rate")
    if not c["fill_rate"].min() <= fill_rate <= c["fill_rate"].max():
        return float("nan")
    return float(np.interp(fill_rate, c["fill_rate"], c["avg_on_hand"]))


def main() -> None:
    """Simulate both policies, compare at equal fill rate, write reports."""
    fc = pd.read_parquet(MODELS_DIR / "backtest_forecasts.parquet")
    first_review = int(fc["fold_origin"].min())
    last_week = int(fc["target_idx"].max())
    types = fc.drop_duplicates(SERIES).set_index(SERIES)["demand_type"]

    rows, curves, arrays = [], [], {}
    for policy, col in POLICIES.items():
        demand, s, index = policy_inputs(fc, first_review, last_week, col)
        sim = simulate(demand, s)
        arrays[policy] = (demand, s)
        rows.append({"policy": policy, "group": "all", **summarize(sim, demand)})
        kind = types.reindex(index).to_numpy()
        for t in ["smooth", "erratic", "intermittent", "lumpy"]:
            m = kind == t
            part = {k: v[m] for k, v in sim.items()}
            rows.append({"policy": policy, "group": t, **summarize(part, demand[m])})
        curves.append(tradeoff_curve(demand, s).assign(policy=policy))

    table = pd.DataFrame(rows)
    curve = pd.concat(curves, ignore_index=True)
    model_all = table[(table.policy == "model") & (table.group == "all")].iloc[0]
    base_all = table[(table.policy == "baseline") & (table.group == "all")].iloc[0]
    base_inv_at_model_fill = inventory_at_fill_rate(curve[curve.policy == "baseline"],
                                                    model_all["fill_rate"])
    saving = 1 - model_all["avg_on_hand"] / base_inv_at_model_fill
    # The saving depends on the service level targeted, so report it at several.
    by_target = [{"fill_rate": fr,
                  "model_avg_on_hand": inventory_at_fill_rate(curve[curve.policy == "model"], fr),
                  "baseline_avg_on_hand": inventory_at_fill_rate(curve[curve.policy == "baseline"], fr)}
                 for fr in TARGET_FILL_RATES]
    for row in by_target:
        row["inventory_saving"] = 1 - row["model_avg_on_hand"] / row["baseline_avg_on_hand"]

    table.to_csv(REPORTS_DIR / "inventory.csv", index=False)
    curve.to_csv(REPORTS_DIR / "inventory_tradeoff.csv", index=False)
    update_metrics({"inventory": {
        "generated_by": "python -m forecast.inventory",
        "weeks_simulated": int(arrays["model"][0].shape[1]),
        "reviews": int(arrays["model"][1].shape[1]),
        "n_series": int(arrays["model"][0].shape[0]),
        "policies": table[table.group == "all"].to_dict(orient="records"),
        "baseline_avg_on_hand_at_model_fill_rate": base_inv_at_model_fill,
        "inventory_saving_at_equal_fill_rate": saving,
        "saving_by_target_fill_rate": by_target,
    }})

    print(table.round(3).to_string(index=False))
    print(f"\nModel policy: fill rate {model_all['fill_rate']:.3f}, avg on hand "
          f"{model_all['avg_on_hand']:.2f} units per item-store-week")
    print(f"Baseline policy as is: fill rate {base_all['fill_rate']:.3f}, avg on hand "
          f"{base_all['avg_on_hand']:.2f}")
    print(f"Baseline needs {base_inv_at_model_fill:.2f} units on hand to match the model's "
          f"fill rate -> model saves {saving:.1%} inventory at equal fill rate")
    for row in by_target:
        print(f"  at {row['fill_rate']:.0%} fill rate: model {row['model_avg_on_hand']:.2f} vs "
              f"baseline {row['baseline_avg_on_hand']:.2f} units -> saving {row['inventory_saving']:.1%}")


if __name__ == "__main__":
    main()
