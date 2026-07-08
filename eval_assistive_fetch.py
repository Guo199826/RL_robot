# eval_assistive_fetch.py
import argparse
import json
from dataclasses import asdict, dataclass, field

import numpy as np
from stable_baselines3 import SAC

from assistive_fetch.envs import make_assistive_fetch_env

REWARD_TERMS = ("dist_term", "success_term", "assist_cost_term", "stage1_shaping_term")
PHASE_LABELS = {0: "lift", 1: "approach", 2: "descend", 3: "push"}


@dataclass
class EpisodeStats:
    success: float = 0.0
    final_dist: float = float("nan")
    mean_assist_norm: float = 0.0
    mean_human_norm: float = 0.0
    mean_full_norm: float = 0.0
    phase_entries: dict = field(default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0})
    return_: float = 0.0
    steps: int = 0
    reward_term_mean: dict = field(default_factory=dict)
    reward_term_sum: dict = field(default_factory=dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independent evaluation for assistive Fetch policies."
    )
    parser.add_argument("--env_id", type=str, default="FetchPush-v4")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained SAC model. Required unless --policy human.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducible eval. If omitted, episodes are sampled "
        "randomly (different results each run). Strongly recommended with --compare "
        "so both policies see identical initial states.",
    )
    parser.add_argument("--render", action="store_true", help="Enable human render mode")
    parser.add_argument(
        "--policy",
        type=str,
        default="model",
        choices=["model", "human"],
        help="'model' = trained assist policy; 'human' = zero-assist (pure human baseline).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run BOTH the trained assist policy and the pure-human (no-assist) "
        "baseline on identical seeds and print a side-by-side comparison.",
    )
    parser.add_argument(
        "--wrapper_type",
        type=str,
        default="fetchpush_two_stage",
        choices=["fetchpush_two_stage", "generic"],
    )
    parser.add_argument("--human_gain", type=float, default=1.0)
    parser.add_argument("--assist_scale", type=float, default=0.15)
    parser.add_argument("--success_bonus", type=float, default=10.0)
    parser.add_argument("--assist_cost_coef", type=float, default=0.005)
    parser.add_argument("--dist_weight", type=float, default=1.0)
    parser.add_argument(
        "--save_json",
        type=str,
        default=None,
        help="Optional path to save aggregate + per-episode stats as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-episode breakdown",
    )
    return parser.parse_args()


def make_model_policy(model):
    def _policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return _policy


def make_human_policy(env):
    zero = np.zeros(env.action_space.shape, dtype=np.float32)

    def _policy(obs):
        return zero

    return _policy


def _count_phase_entry(prev_phase, phase, phase_entries):
    if prev_phase is None:
        phase_entries[int(phase)] = phase_entries.get(int(phase), 0) + 1
        return phase

    prev = int(prev_phase)
    cur = int(phase)
    if cur != prev:
        phase_entries[cur] = phase_entries.get(cur, 0) + 1
    return phase


def run_episode(env, policy_fn, *, seed=None, verbose=False, ep_idx=None):
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    done = False
    truncated = False

    ep_ret = 0.0
    ep_success = 0.0
    final_dist = float("nan")
    assist_norms = []
    human_norms = []
    full_norms = []
    reward_term_steps = {term: [] for term in REWARD_TERMS}
    phase_entries = {0: 0, 1: 0, 2: 0, 3: 0}
    prev_phase = None
    steps = 0

    while not (done or truncated):
        action = policy_fn(obs)
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
        ep_ret += reward
        ep_success = max(ep_success, info.get("assist_reward/success", 0.0))
        final_dist = info.get("assist_reward/dist", final_dist)

        assist_norms.append(info.get("assist_action_norm", 0.0))
        human_norms.append(info.get("human_action_norm", 0.0))
        full_norms.append(info.get("full_action_norm", 0.0))
        for term in REWARD_TERMS:
            reward_term_steps[term].append(info.get(f"assist_reward/{term}", 0.0))

        if "human_phase" in info:
            prev_phase = _count_phase_entry(prev_phase, info["human_phase"], phase_entries)

    reward_term_mean = {
        term: float(np.mean(vals)) if vals else 0.0
        for term, vals in reward_term_steps.items()
    }
    reward_term_sum = {
        term: float(np.sum(vals)) if vals else 0.0
        for term, vals in reward_term_steps.items()
    }

    stats = EpisodeStats(
        success=ep_success,
        final_dist=final_dist,
        mean_assist_norm=float(np.mean(assist_norms)) if assist_norms else 0.0,
        mean_human_norm=float(np.mean(human_norms)) if human_norms else 0.0,
        mean_full_norm=float(np.mean(full_norms)) if full_norms else 0.0,
        phase_entries=phase_entries,
        return_=ep_ret,
        steps=steps,
        reward_term_mean=reward_term_mean,
        reward_term_sum=reward_term_sum,
    )

    if verbose:
        prefix = f"Episode {ep_idx}: " if ep_idx is not None else "Episode: "
        print(
            f"{prefix}"
            f"success={stats.success:.0f} | "
            f"final_dist={stats.final_dist:.4f} | "
            f"return={stats.return_:.2f} | "
            f"steps={stats.steps} | "
            f"assist={stats.mean_assist_norm:.4f} | "
            f"human={stats.mean_human_norm:.4f} | "
            f"phase_entries={stats.phase_entries}"
        )

    return stats


def run_eval(env, policy_fn, *, episodes, base_seed=None, verbose=False, label=None):
    if label:
        print(f"\n---- Running '{label}' policy ({episodes} episodes) ----")
    episode_stats = []
    for ep in range(episodes):
        ep_seed = base_seed + ep if base_seed is not None else None
        stats = run_episode(
            env,
            policy_fn,
            seed=ep_seed,
            verbose=verbose,
            ep_idx=ep + 1,
        )
        episode_stats.append(stats)
    return episode_stats


def _mean_std(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def summarize(episodes):
    if not episodes:
        return {"n_episodes": 0}

    successes = np.array([e.success for e in episodes], dtype=np.float64)
    final_dists = np.array([e.final_dist for e in episodes], dtype=np.float64)
    assist = np.array([e.mean_assist_norm for e in episodes], dtype=np.float64)
    human = np.array([e.mean_human_norm for e in episodes], dtype=np.float64)
    returns = np.array([e.return_ for e in episodes], dtype=np.float64)
    steps = np.array([e.steps for e in episodes], dtype=np.float64)

    phase_totals = {p: 0 for p in (0, 1, 2, 3)}
    for ep in episodes:
        for p, c in ep.phase_entries.items():
            phase_totals[int(p)] = phase_totals.get(int(p), 0) + c

    n = len(episodes)
    phase_means = {p: phase_totals[p] / n for p in phase_totals}

    reward_term_mean = {}
    reward_term_sum = {}
    for term in REWARD_TERMS:
        m, s = _mean_std([e.reward_term_mean.get(term, 0.0) for e in episodes])
        sm, ss = _mean_std([e.reward_term_sum.get(term, 0.0) for e in episodes])
        reward_term_mean[term] = {"mean": m, "std": s}
        reward_term_sum[term] = {"mean": sm, "std": ss}

    return {
        "n_episodes": n,
        "success_rate": float(np.mean(successes)),
        "success_count": int(np.sum(successes)),
        "final_dist_mean": float(np.mean(final_dists)),
        "final_dist_std": float(np.std(final_dists)),
        "final_dist_median": float(np.median(final_dists)),
        "assist_norm_mean": float(np.mean(assist)),
        "assist_norm_std": float(np.std(assist)),
        "human_norm_mean": float(np.mean(human)),
        "human_norm_std": float(np.std(human)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "steps_mean": float(np.mean(steps)),
        "phase_entries_total": phase_totals,
        "phase_entries_mean_per_episode": phase_means,
        "reward_term_mean": reward_term_mean,
        "reward_term_sum": reward_term_sum,
    }


def split_by_outcome(episodes):
    success_eps = [e for e in episodes if e.success >= 0.5]
    failure_eps = [e for e in episodes if e.success < 0.5]
    return {
        "all": summarize(episodes),
        "success": summarize(success_eps) if success_eps else None,
        "failure": summarize(failure_eps) if failure_eps else None,
        "success_count": len(success_eps),
        "failure_count": len(failure_eps),
    }


def print_summary(summary, title="Evaluation summary"):
    n = summary["n_episodes"]
    print("\n" + "=" * 60)
    print(f"{title} ({n} episodes)")
    print("=" * 60)
    print(
        f"Success rate:     {summary['success_rate']:.1%} "
        f"({summary['success_count']}/{n})"
    )
    print(
        f"Final dist (m):   mean={summary['final_dist_mean']:.4f}  "
        f"std={summary['final_dist_std']:.4f}  "
        f"median={summary['final_dist_median']:.4f}"
    )
    print(
        f"Assist effort:    mean={summary['assist_norm_mean']:.4f}  "
        f"std={summary['assist_norm_std']:.4f}  "
        f"(‖scaled_assist‖ per step)"
    )
    print(
        f"Human effort:     mean={summary['human_norm_mean']:.4f}  "
        f"std={summary['human_norm_std']:.4f}  "
        f"(‖human_intent‖ per step)"
    )
    print(
        f"Episode return:   mean={summary['return_mean']:.2f}  "
        f"std={summary['return_std']:.2f}"
    )
    print(f"Steps per ep:     mean={summary['steps_mean']:.1f}")
    phase_total = sum(summary["phase_entries_total"].values())
    if phase_total > 0:
        print("Phase entries (mean per episode):")
        for phase in (0, 1, 2, 3):
            total = summary["phase_entries_total"][phase]
            mean = summary["phase_entries_mean_per_episode"][phase]
            label = {0: "lift", 1: "approach", 2: "descend", 3: "push"}.get(phase, str(phase))
            print(f"  phase {phase} ({label:8s}): total={total:4d}  mean/ep={mean:.2f}")
    else:
        print("Phase entries: n/a (use --wrapper_type fetchpush_two_stage)")
    print("=" * 60)


def _fmt(v, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return f"{v:.{digits}f}"


def _delta_str(success_val, failure_val, digits=4):
    if success_val is None or failure_val is None:
        return "n/a"
    if isinstance(success_val, float) and (
        np.isnan(success_val) or np.isnan(failure_val)
    ):
        return "n/a"
    d = success_val - failure_val
    return f"{d:+.{digits}f}"


def print_outcome_analysis(outcome_split, title="Outcome analysis"):
    s = outcome_split.get("success")
    f = outcome_split.get("failure")
    n_succ = outcome_split["success_count"]
    n_fail = outcome_split["failure_count"]
    n_all = outcome_split["all"]["n_episodes"]

    print("\n" + "=" * 78)
    print(f"{title}  [all={n_all}, success={n_succ}, failure={n_fail}]")
    print("=" * 78)

    if s is None:
        print("No successful episodes — cannot compare success vs failure.")
        print("=" * 78)
        return
    if f is None:
        print("No failed episodes — cannot compare success vs failure.")
        print("=" * 78)
        return

    col_w = 18
    header = (
        f"{'metric':<24}"
        f"{'success':>{col_w}}"
        f"{'failure':>{col_w}}"
        f"{'delta (s-f)':>{col_w}}"
    )
    print(header)
    print("-" * 78)

    def row(name, s_val, f_val, digits=4):
        print(
            f"{name:<24}"
            f"{_fmt(s_val, digits):>{col_w}}"
            f"{_fmt(f_val, digits):>{col_w}}"
            f"{_delta_str(s_val, f_val, digits):>{col_w}}"
        )

    row("Final dist (m)", s["final_dist_mean"], f["final_dist_mean"])
    row("Assist effort", s["assist_norm_mean"], f["assist_norm_mean"])
    row("Human effort", s["human_norm_mean"], f["human_norm_mean"])
    row("Episode return", s["return_mean"], f["return_mean"], digits=2)

    print("")
    print("Reward terms — episode sum (matches total return composition):")
    for term in REWARD_TERMS:
        s_mean = s["reward_term_sum"][term]["mean"]
        f_mean = f["reward_term_sum"][term]["mean"]
        row(f"  sum/{term}", s_mean, f_mean, digits=2)

    print("")
    print("Reward terms — per-step mean:")
    for term in REWARD_TERMS:
        s_mean = s["reward_term_mean"][term]["mean"]
        f_mean = f["reward_term_mean"][term]["mean"]
        row(f"  mean/{term}", s_mean, f_mean)

    phase_total = sum(s["phase_entries_total"].values()) + sum(f["phase_entries_total"].values())
    if phase_total > 0:
        print("")
        print("Phase entries (mean per episode):")
        for phase in (0, 1, 2, 3):
            label = PHASE_LABELS.get(phase, str(phase))
            s_mean = s["phase_entries_mean_per_episode"][phase]
            f_mean = f["phase_entries_mean_per_episode"][phase]
            row(f"  phase {phase} ({label})", s_mean, f_mean, digits=2)

    print("=" * 78)
    print(
        "delta = success − failure. Positive success_term / lower final_dist on "
        "success episodes\n"
        "      is expected. High phase 1/2 entries on failures often means "
        "stuck before push."
    )


def print_comparison(assist_summary, human_summary):
    a, h = assist_summary, human_summary

    def _delta(key, pct=False, better="high"):
        av, hv = a[key], h[key]
        d = av - hv
        arrow = ""
        if better == "high":
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "=")
        else:
            arrow = "↓(good)" if d < 0 else ("↑(bad)" if d > 0 else "=")
        if pct:
            return f"{av:.1%}", f"{hv:.1%}", f"{d:+.1%} {arrow}"
        return f"{av:.4f}", f"{hv:.4f}", f"{d:+.4f} {arrow}"

    rows = [
        ("Success rate", *_delta("success_rate", pct=True, better="high")),
        ("Final dist (m)", *_delta("final_dist_mean", better="low")),
        ("Human effort", *_delta("human_norm_mean", better="low")),
        ("Assist effort", *_delta("assist_norm_mean", better="high")),
        ("Episode return", *_delta("return_mean", better="high")),
    ]

    n = a["n_episodes"]
    print("\n" + "=" * 74)
    print(f"COMPARISON: model+assist  vs  human-only (no assist)   [{n} episodes each]")
    print("=" * 74)
    print(f"{'metric':<18}{'model+assist':>16}{'human-only':>16}{'delta (a-h)':>22}")
    print("-" * 74)
    for name, av, hv, d in rows:
        print(f"{name:<18}{av:>16}{hv:>16}{d:>22}")
    print("=" * 74)
    print(
        "Note: 'delta' = model+assist minus human-only. For success/return higher "
        "is better;\n      for final dist / human effort lower is better. Assist "
        "effort shows how much\n      the policy actively adds on top of the human intent."
    )


def _stats_payload(episode_stats):
    episodes = []
    for ep in episode_stats:
        d = asdict(ep)
        d["return"] = d.pop("return_")
        episodes.append(d)
    return episodes


def main():
    args = parse_args()

    need_model = args.compare or args.policy == "model"
    if need_model and not args.model_path:
        raise SystemExit("--model_path is required unless --policy human (without --compare).")

    env = make_assistive_fetch_env(
        env_id=args.env_id,
        render_mode="human" if args.render else None,
        wrapper_type=args.wrapper_type,
        human_gain=args.human_gain,
        assist_scale=args.assist_scale,
        success_bonus=args.success_bonus,
        assist_cost_coef=args.assist_cost_coef,
        dist_weight=args.dist_weight,
        use_dense_reward=True,
    )

    model = SAC.load(args.model_path, env=env) if need_model else None
    human_policy = make_human_policy(env)
    model_policy = make_model_policy(model) if model is not None else None

    results = {}

    if args.compare:
        if args.seed is None:
            print(
                "WARNING: --compare without --seed means the two policies are "
                "evaluated on DIFFERENT random episodes, which is not a fair "
                "comparison. Consider passing --seed for a paired comparison."
            )
        assist_stats = run_eval(
            env, model_policy, episodes=args.episodes, base_seed=args.seed,
            verbose=args.verbose, label="model+assist",
        )
        human_stats = run_eval(
            env, human_policy, episodes=args.episodes, base_seed=args.seed,
            verbose=args.verbose, label="human-only",
        )
        assist_summary = summarize(assist_stats)
        human_summary = summarize(human_stats)
        assist_outcomes = split_by_outcome(assist_stats)
        human_outcomes = split_by_outcome(human_stats)
        print_summary(assist_summary, title="model+assist summary")
        print_outcome_analysis(assist_outcomes, title="SUCCESS vs FAILURE — model+assist")
        print_summary(human_summary, title="human-only (no assist) summary")
        print_outcome_analysis(human_outcomes, title="SUCCESS vs FAILURE — human-only")
        print_comparison(assist_summary, human_summary)
        results = {
            "model+assist": {
                "summary": assist_summary,
                "outcomes": assist_outcomes,
                "episodes": _stats_payload(assist_stats),
            },
            "human-only": {
                "summary": human_summary,
                "outcomes": human_outcomes,
                "episodes": _stats_payload(human_stats),
            },
        }
    else:
        if args.policy == "model":
            policy_fn, label = model_policy, "model+assist"
        else:
            policy_fn, label = human_policy, "human-only"
        episode_stats = run_eval(
            env, policy_fn, episodes=args.episodes, base_seed=args.seed,
            verbose=args.verbose, label=label,
        )
        summary = summarize(episode_stats)
        outcomes = split_by_outcome(episode_stats)
        print_summary(summary, title=f"{label} summary")
        print_outcome_analysis(outcomes, title=f"SUCCESS vs FAILURE — {label}")
        results = {
            label: {
                "summary": summary,
                "outcomes": outcomes,
                "episodes": _stats_payload(episode_stats),
            },
        }

    if args.save_json:
        payload = {"config": vars(args), "results": results}
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.save_json}")

    env.close()


if __name__ == "__main__":
    main()
