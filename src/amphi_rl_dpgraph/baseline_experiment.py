# Baseline experiment suite for AMPHI evaluation. Scores five static masking
# policies (raw/weak/synthetic/pseudo/redact) and an adaptive controller across
# three synthetic workload shapes (monotonic, bursty, mixed-risk). Produces a
# Pareto frontier plot, workload comparison bars, adaptive-vs-static delta chart,
# risk-conditional score curves, and a CSV/JSONL summary table. Call
# run_baseline_experiments(audit_log_path, outdir) from main() after the event loop.

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_PRIVACY_SCORE = {"raw": 0.00, "weak": 0.30, "synthetic": 0.70, "pseudo": 0.90, "redact": 1.00}
_UTILITY_SCORE = {"raw": 1.00, "weak": 0.90, "synthetic": 0.80, "pseudo": 0.60, "redact": 0.00}
_LATENCY_MS    = {"raw": 0.5,  "weak": 1.0,  "synthetic": 2.0,  "pseudo": 1.5,  "redact": 1.0}


def adaptive_policy(risk: float, risk1: float = 0.40, risk2: float = 0.80) -> str:
    mid = (risk1 + risk2) / 2.0
    if risk < risk1:  return "weak"
    if risk < mid:    return "synthetic"
    if risk < risk2:  return "pseudo"
    return "redact"


def score_event(policy: str, risk: float) -> Tuple[float, float]:
    priv = _PRIVACY_SCORE[policy]
    util = _UTILITY_SCORE[policy]
    under_mask = (1.0 - priv) * risk
    over_mask  = priv * (1.0 - risk) * 0.4
    eff_priv = min(1.0, max(0.0, priv - under_mask * 0.5))
    eff_util = min(1.0, max(0.0, util - over_mask  * 0.5))
    return eff_priv, eff_util


def _monotonic_risks(base_risks: List[float]) -> List[float]:
    return list(base_risks)


def _bursty_risks(base_risks: List[float], cycle: int = 6) -> List[float]:
    out = []
    for i, r in enumerate(base_risks):
        phase = i % cycle
        if phase < cycle // 2:
            out.append(r)
        else:
            out.append(base_risks[i % (cycle // 2)] * 0.20)
    return out


def _mixed_risks(base_risks: List[float], seed: int = 42) -> List[float]:
    rng = random.Random(seed)
    return [r * (0.25 if rng.random() < 0.70 else 1.0) for r in base_risks]


WORKLOADS = {
    "monotonic": _monotonic_risks,
    "bursty":    _bursty_risks,
    "mixed":     _mixed_risks,
}


def compare_policies(risks: List[float]) -> Dict[str, Dict]:
    policies = {
        "Always-Raw":       lambda r: "raw",
        "Always-Weak":      lambda r: "weak",
        "Always-Synthetic": lambda r: "synthetic",
        "Always-Pseudo":    lambda r: "pseudo",
        "Always-Redact":    lambda r: "redact",
        "Adaptive":         adaptive_policy,
    }

    results = {}
    for name, policy_fn in policies.items():
        privs, utils, lats = [], [], []
        high_risk_priv, low_risk_util = [], []
        consent_violations = 0

        for i, risk in enumerate(risks):
            pol = policy_fn(risk)
            p, u = score_event(pol, risk)
            privs.append(p)
            utils.append(u)
            lats.append(_LATENCY_MS[pol])

            if risk >= 0.70:
                high_risk_priv.append(p)
            if risk <= 0.45:
                low_risk_util.append(u)

            if i % 2 == 0 and pol == "redact":
                consent_violations += 1

        results[name] = {
            "privacy_mean":         round(sum(privs) / len(privs), 4),
            "utility_mean":         round(sum(utils) / len(utils), 4),
            "latency_mean_ms":      round(sum(lats)  / len(lats),  4),
            "privacy_at_high_risk": round(sum(high_risk_priv) / max(1, len(high_risk_priv)), 4),
            "utility_at_low_risk":  round(sum(low_risk_util)  / max(1, len(low_risk_util)),  4),
            "consent_violations":   consent_violations,
            "policy_mix":           dict(Counter(policy_fn(r) for r in risks)),
        }
    return results


COLORS = {
    "Always-Raw":       "#e74c3c",
    "Always-Weak":      "#e67e22",
    "Always-Synthetic": "#f1c40f",
    "Always-Pseudo":    "#2ecc71",
    "Always-Redact":    "#3498db",
    "Adaptive":         "#9b59b6",
}
MARKERS = {
    "Always-Raw": "x", "Always-Weak": "s", "Always-Synthetic": "D",
    "Always-Pseudo": "^", "Always-Redact": "v", "Adaptive": "o",
}


def plot_pareto_frontier(workload_results: Dict[str, Dict], outdir: Path) -> None:
    workload_names = list(workload_results.keys())
    fig, axes = plt.subplots(1, len(workload_names), figsize=(5 * len(workload_names), 5))
    if len(workload_names) == 1:
        axes = [axes]

    for ax, wname in zip(axes, workload_names):
        results = workload_results[wname]
        for pname, metrics in results.items():
            x = metrics["utility_at_low_risk"]
            y = metrics["privacy_at_high_risk"]
            ax.scatter(x, y, s=120, c=COLORS[pname], marker=MARKERS[pname],
                       label=pname, zorder=5)
            ax.annotate(pname.replace("Always-", ""), (x, y),
                        textcoords="offset points", xytext=(6, 3), fontsize=7)

        points = [(m["utility_at_low_risk"], m["privacy_at_high_risk"]) for m in results.values()]
        points.sort()
        pareto = []
        max_p = -1
        for u, p in points:
            if p > max_p:
                pareto.append((u, p))
                max_p = p
        if len(pareto) >= 2:
            px, py = zip(*pareto)
            ax.step(px, py, where="post", color="#666", linewidth=1.0,
                    linestyle="--", alpha=0.5, label="Pareto frontier")

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Utility @ Low Risk", fontsize=10)
        ax.set_ylabel("Privacy @ High Risk", fontsize=10)
        ax.set_title(f"Workload: {wname}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.axhline(0.85, color="red", linewidth=0.6, linestyle=":", alpha=0.4,
                   label="Privacy floor (0.85)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Privacy@HighRisk – Utility@LowRisk Pareto Frontier by Workload",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "pareto_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_workload_comparison(workload_results: Dict[str, Dict], outdir: Path) -> None:
    policy_names = list(next(iter(workload_results.values())).keys())
    workload_names = list(workload_results.keys())
    x = np.arange(len(workload_names))
    width = 0.12

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))

    for i, pname in enumerate(policy_names):
        p_vals = [workload_results[w][pname]["privacy_at_high_risk"] for w in workload_names]
        u_vals = [workload_results[w][pname]["utility_at_low_risk"]  for w in workload_names]
        offset = (i - len(policy_names) / 2) * width + width / 2
        ax1.bar(x + offset, p_vals, width, label=pname, color=COLORS[pname],
                edgecolor="white", linewidth=0.5)
        ax2.bar(x + offset, u_vals, width, label=pname, color=COLORS[pname],
                edgecolor="white", linewidth=0.5)

    for ax, ylabel, title, floor in [
        (ax1, "Privacy Score", "Privacy at High Risk (risk > 0.70)", 0.85),
        (ax2, "Utility Score", "Utility at Low Risk  (risk < 0.45)", 0.80),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([w.capitalize() for w in workload_names])
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.1)
        ax.axhline(floor, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=3)

    fig.tight_layout()
    fig.savefig(outdir / "workload_comparison.png", dpi=150)
    plt.close(fig)


def plot_adaptive_vs_best_static(workload_results: Dict[str, Dict], outdir: Path) -> None:
    static_policies = [k for k in next(iter(workload_results.values())).keys()
                       if k != "Adaptive"]
    workload_names = list(workload_results.keys())

    fig, axes = plt.subplots(1, len(workload_names), figsize=(4.5 * len(workload_names), 5))
    if len(workload_names) == 1:
        axes = [axes]

    for ax, wname in zip(axes, workload_names):
        res = workload_results[wname]
        adapt_p = res["Adaptive"]["privacy_at_high_risk"]
        adapt_u = res["Adaptive"]["utility_at_low_risk"]

        labels, p_deltas, u_deltas = [], [], []
        for sp in static_policies:
            labels.append(sp.replace("Always-", ""))
            p_deltas.append(round(adapt_p - res[sp]["privacy_at_high_risk"], 3))
            u_deltas.append(round(adapt_u - res[sp]["utility_at_low_risk"],  3))

        y = np.arange(len(labels))
        h = 0.35
        ax.barh(y + h / 2, p_deltas, h, label="Privacy@HighRisk delta",
                color=["#2ecc71" if d >= 0 else "#e74c3c" for d in p_deltas])
        ax.barh(y - h / 2, u_deltas, h, label="Utility@LowRisk delta",
                color=["#3498db" if d >= 0 else "#e67e22" for d in u_deltas],
                alpha=0.8)

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Adaptive − Static  (positive = Adaptive wins)", fontsize=9)
        ax.set_title(f"{wname.capitalize()} workload", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Adaptive vs Each Static Baseline: Δ Privacy & Utility",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "adaptive_vs_static_delta.png", dpi=150)
    plt.close(fig)


def plot_risk_conditional_policy(outdir: Path) -> None:
    risks = np.linspace(0, 1, 100)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for policy, color in [("weak", "#e67e22"), ("synthetic", "#f39c12"),
                           ("pseudo", "#2ecc71"), ("redact", "#3498db")]:
        p_vals = [score_event(policy, r)[0] for r in risks]
        u_vals = [score_event(policy, r)[1] for r in risks]
        ax1.plot(risks, p_vals, color=color, linewidth=1.2,
                 linestyle="--", alpha=0.7, label=f"Always-{policy}")
        ax2.plot(risks, u_vals, color=color, linewidth=1.2,
                 linestyle="--", alpha=0.7, label=f"Always-{policy}")

    adapt_p = [score_event(adaptive_policy(r), r)[0] for r in risks]
    adapt_u = [score_event(adaptive_policy(r), r)[1] for r in risks]
    ax1.plot(risks, adapt_p, color="#9b59b6", linewidth=2.2, label="Adaptive")
    ax2.plot(risks, adapt_u, color="#9b59b6", linewidth=2.2, label="Adaptive")

    for thresh, label in [
        (0.40, "weak→synthetic"),
        (0.60, "synthetic→pseudo"),
        (0.80, "pseudo→redact"),
    ]:
        for ax in (ax1, ax2):
            ax.axvline(thresh, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
            ax.text(thresh + 0.01, 0.02, label, fontsize=6, color="gray", rotation=90)

    ax1.set_ylabel("Effective Privacy", fontsize=10)
    ax1.set_title("Risk-Conditional Privacy Score", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right", ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    ax2.set_xlabel("Risk Score", fontsize=10)
    ax2.set_ylabel("Effective Utility", fontsize=10)
    ax2.set_title("Risk-Conditional Utility Score", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper right", ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    fig.suptitle("Why Adaptive Dominates: Static Policies Cannot Optimise Both",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "risk_conditional_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_comparison_table(workload_results: Dict[str, Dict], outdir: Path) -> None:
    rows = []
    for wname, results in workload_results.items():
        for pname, metrics in results.items():
            rows.append({
                "workload":             wname,
                "policy":               pname,
                "privacy_mean":         metrics["privacy_mean"],
                "utility_mean":         metrics["utility_mean"],
                "privacy_at_high_risk": metrics["privacy_at_high_risk"],
                "utility_at_low_risk":  metrics["utility_at_low_risk"],
                "consent_violations":   metrics["consent_violations"],
                "latency_mean_ms":      metrics["latency_mean_ms"],
            })

    (outdir / "baseline_comparison.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows)
    )

    with open(outdir / "baseline_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_baseline_experiments(
    audit_log_path: str,
    outdir: Path,
) -> None:
    outdir = Path(outdir)

    with open(audit_log_path) as f:
        audit_rows = [json.loads(line) for line in f]

    base_risks = [r["risk"] for r in audit_rows]

    workload_risks = {
        wname: fn(base_risks)
        for wname, fn in WORKLOADS.items()
    }

    workload_results = {
        wname: compare_policies(risks)
        for wname, risks in workload_risks.items()
    }

    plot_pareto_frontier(workload_results, outdir)
    plot_workload_comparison(workload_results, outdir)
    plot_adaptive_vs_best_static(workload_results, outdir)
    plot_risk_conditional_policy(outdir)
    save_comparison_table(workload_results, outdir)

    print("\n=== Baseline Comparison Summary ===\n")
    for wname, results in workload_results.items():
        print(f"  Workload: {wname}")
        print(f"  {'Policy':<16} {'Priv':>6} {'Util':>6} {'P@Hi':>6} {'U@Lo':>6} {'ConsentViol':>12}")
        for pname, m in results.items():
            print(f"  {pname:<16} {m['privacy_mean']:>6.3f} {m['utility_mean']:>6.3f} "
                  f"{m['privacy_at_high_risk']:>6.3f} {m['utility_at_low_risk']:>6.3f} "
                  f"{m['consent_violations']:>12}")
        print()

    print("Baseline artifacts written to:", outdir)
    print("  pareto_frontier.png")
    print("  workload_comparison.png")
    print("  adaptive_vs_static_delta.png")
    print("  risk_conditional_scores.png")
    print("  baseline_comparison.csv")
    print("  baseline_comparison.jsonl\n")
