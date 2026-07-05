"""Plotting helpers for benchmark figures (headless ``Agg`` backend)."""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless / no display required
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def read_curve_csv(path: str, x="timesteps", y="success_rate"):
    xs, ys = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get(x) and row.get(y):
                xs.append(float(row[x]))
                ys.append(float(row[y]))
    return np.array(xs), np.array(ys)


def plot_learning_curves(
    curves: Dict[str, str],
    out_path: str,
    title: str = "Sample efficiency (success rate vs timesteps)",
    ylabel: str = "Success rate",
    y: str = "success_rate",
):
    """curves: {label: csv_path}. Overlays success-rate curves for comparison."""
    _ensure_dir(out_path)
    plt.figure(figsize=(9, 5.5))
    for label, path in curves.items():
        if not os.path.exists(path):
            print(f"[plot] missing {path}, skipping {label}")
            continue
        xs, ys = read_curve_csv(path, y=y)
        if len(xs):
            plt.plot(xs, ys, linewidth=2, marker="o", markersize=3, label=label)
    plt.xlabel("Timesteps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out_path}")


def plot_metric_bars(
    rows: List[dict],
    metrics: Sequence[str],
    out_path: str,
    title: str = "Controller comparison",
):
    """Grouped bar chart: one subplot per metric, one bar per controller."""
    _ensure_dir(out_path)
    names = [r["controller"] for r in rows]
    n = len(metrics)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for ax, metric in zip(axes, metrics):
        vals = [r.get(metric, np.nan) for r in rows]
        ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_training_curves(csv_path: str, out_path: str, title: str = "Training curves"):
    """Reproduce the classic 2x2 training-diagnostic figure from a curve CSV.

    Panels: mean reward / actor(or policy) loss / critic(or value) loss /
    entropy coef. Missing columns are skipped gracefully (PPO vs SAC differ).
    """
    _ensure_dir(out_path)
    if not os.path.exists(csv_path):
        print(f"[plot] missing {csv_path}, skipping training curves")
        return
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f"[plot] empty {csv_path}")
        return
    ts = [float(r["timesteps"]) for r in rows]

    def col(name):
        xs, ys = [], []
        for t, r in zip(ts, rows):
            if r.get(name) not in (None, ""):
                xs.append(t); ys.append(float(r[name]))
        return xs, ys

    panels = [
        ("mean_reward", "Mean step reward", "tab:blue"),
        ("actor_loss", "Actor loss (SAC)", "tab:red"),
        ("critic_loss", "Critic loss (SAC)", "tab:green"),
        ("ent_coef", "Entropy coef (SAC)", "tab:purple"),
        ("policy_loss", "Policy loss (PPO)", "tab:red"),
        ("value_loss", "Value loss (PPO)", "tab:green"),
    ]
    available = [p for p in panels if col(p[0])[0]]
    n = max(1, len(available))
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, lab, c) in zip(axes, available):
        xs, ys = col(key)
        ax.plot(xs, ys, color=c, lw=2)
        ax.set_title(f"{lab}" + (f" (last={ys[-1]:.3f})" if ys else ""))
        ax.set_xlabel("Timesteps"); ax.grid(True, alpha=0.3)
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_error_curves(
    error_traces: Dict[str, Sequence[float]],
    out_path: str,
    title: str = "Goal-distance error vs time",
):
    """Overlay per-step error curves for several controllers (one episode each)."""
    _ensure_dir(out_path)
    plt.figure(figsize=(9, 5.5))
    for label, errors in error_traces.items():
        plt.plot(errors, linewidth=2, label=label)
    plt.xlabel("Time step")
    plt.ylabel("Distance error (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {out_path}")
