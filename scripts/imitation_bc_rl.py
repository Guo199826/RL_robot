"""🅳  Imitation + RL: scripted demos -> BC warm-start -> RL fine-tune.

Adds behavior cloning (BC) on top of pure RL. The scripted state machine is a
free near-optimal demonstrator on FetchPickAndPlace; we clone it into the policy
network, then fine-tune with RL. This is the standard fix for "pure RL can't
discover the grasp": IL gets the policy into the right region cheaply.

Per algorithm (one process each, so they can run in parallel):
  - SAC: BC the actor, fine-tune with SAC + HER on the *sparse* reward.
  - PPO: BC the policy net, fine-tune with PPO on the *dense* reward.

Logs success after BC (no RL yet) and the fine-tuning success curve. Compare the
final number to the from-scratch baseline in ``results/controller_matrix_pap``.

Usage:
    python scripts/imitation_bc_rl.py --algo SAC --task FetchPickAndPlace-v4 \
        --demos 200 --bc-epochs 40 --timesteps 40000
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mc.common.envs import make_env
from mc.common.training import build_model, needs_her
from mc.common.callbacks import SuccessRateEvalCallback, CurveLoggerCallback
from mc.common.eval import evaluate, model_policy
from mc.common.metrics import summarize
from mc.common.plotting import plot_training_curves
from mc.imitation import (collect_demos, behavior_clone, behavior_clone_ppo,
                          seed_replay_buffer_with_scripted)


def eval_success(model, task, episodes=30, max_steps=50):
    env = make_env(task, reward_type="sparse")  # success always judged on sparse
    trace, _ = evaluate(env, model_policy(model), n_episodes=episodes, max_steps=max_steps)
    env.close()
    return summarize(trace)["success_rate"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["SAC", "PPO"], default="SAC")
    p.add_argument("--task", default="FetchPickAndPlace-v4")
    p.add_argument("--demos", type=int, default=200)
    p.add_argument("--bc-epochs", type=int, default=40)
    p.add_argument("--timesteps", type=int, default=40000)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--out-dir", default="results/imitation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seed-buffer", type=int, default=0,
                   help="(off-policy only) seed N scripted episodes into the replay "
                        "buffer (SACfD); fixes BC-only catastrophic forgetting")
    args = p.parse_args()
    tag = args.algo.lower() + ("_sacfd" if args.seed_buffer and needs_her(args.algo) else "")
    algo_dir = os.path.join(args.out_dir, tag)
    os.makedirs(algo_dir, exist_ok=True)

    # 1) demos from the scripted expert (cached, shared across algos)
    demo_path = os.path.join(args.out_dir, "demos.npz")
    if os.path.exists(demo_path):
        import numpy as np
        demos = dict(np.load(demo_path))
        print(f"[demos] reuse {demo_path} ({len(demos['actions'])} transitions)")
    else:
        demos = collect_demos(n_episodes=args.demos, max_steps=args.max_steps,
                              task=args.task, out_path=demo_path)

    # 2) build model: SAC -> HER+sparse, PPO -> dense
    use_her = needs_her(args.algo)
    train_env = make_env(args.task, reward_type="sparse" if use_her else "dense")
    model = build_model(args.algo, train_env, use_her=use_her, seed=args.seed, verbose=0)

    # 3) behavior cloning warm-start
    print(f"\n[bc] cloning scripted demos into {args.algo} policy...")
    if args.algo == "SAC":
        behavior_clone(model, demos, epochs=args.bc_epochs)
    else:
        behavior_clone_ppo(model, demos, epochs=args.bc_epochs)
    after_bc = eval_success(model, args.task)
    print(f"[bc] success right after BC (no RL yet): {after_bc:.0%}")

    # 3b) (off-policy) seed the replay buffer with demos so the critic is
    #     grounded in successful transitions -> robust SACfD-style fine-tuning.
    if args.seed_buffer and needs_her(args.algo):
        seed_replay_buffer_with_scripted(model, n_episodes=args.seed_buffer,
                                         max_steps=args.max_steps)

    # 4) RL fine-tune
    succ_cb = SuccessRateEvalCallback(
        lambda: make_env(args.task, reward_type="sparse"),
        eval_freq=4000, n_eval_episodes=20, max_steps=args.max_steps,
        csv_path=os.path.join(algo_dir, "bc_then_rl_curve.csv"),
    )
    curve_cb = CurveLoggerCallback(log_freq=2000,
                                   csv_path=os.path.join(algo_dir, "train_curve.csv"))
    print(f"\n[rl] fine-tuning {args.algo} for {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, callback=[succ_cb, curve_cb])
    model.save(os.path.join(algo_dir, "bc_then_rl_model"))
    train_env.close()

    after_rl = eval_success(model, args.task)
    plot_training_curves(os.path.join(algo_dir, "train_curve.csv"),
                         os.path.join(algo_dir, "training_curves.png"),
                         title=f"{args.algo} BC+RL fine-tuning ({args.task})")
    with open(os.path.join(algo_dir, "summary.json"), "w") as f:
        json.dump({"algo": args.algo, "task": args.task,
                   "after_bc": after_bc, "after_rl": after_rl,
                   "timesteps": args.timesteps, "demos_transitions": len(demos["actions"])}, f, indent=2)
    print(f"[done] {args.algo}: after_BC={after_bc:.0%}  after_RL={after_rl:.0%}")


if __name__ == "__main__":
    main()
