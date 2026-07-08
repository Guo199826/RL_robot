# train_assistive_fetch.py
import os
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from assistive_fetch.envs import make_assistive_fetch_env
from assistive_fetch.callbacks import PlottingCallback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="FetchPush-v4")
    parser.add_argument("--total_timesteps", type=int, default=300_000)
    parser.add_argument("--log_dir", type=str, default="./logs/assistive_fetch")
    parser.add_argument("--plot", action="store_true", help="实时绘制训练关键指标曲线")
    parser.add_argument("--plot_freq", type=int, default=1000, help="每多少个 step 刷新一次曲线")
    parser.add_argument("--render", action="store_true", help="训练时实时渲染仿真环境 (render_mode=human)")
    parser.add_argument("--human_gain", type=float, default=0.6)
    parser.add_argument("--assist_scale", type=float, default=0.5)
    parser.add_argument("--smoothness_coef", type=float, default=0.05)
    parser.add_argument("--effort_coef", type=float, default=0.05)
    parser.add_argument("--overassist_coef", type=float, default=0.1)
    parser.add_argument("--success_bonus", type=float, default=5.0)
    parser.add_argument("--dist_weight", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    render_mode = "human" if args.render else None

    train_env = make_assistive_fetch_env(
        env_id=args.env_id,
        render_mode=render_mode,
        human_gain=args.human_gain,
        assist_scale=args.assist_scale,
        smoothness_coef=args.smoothness_coef,
        effort_coef=args.effort_coef,
        overassist_coef=args.overassist_coef,
        success_bonus=args.success_bonus,
        dist_weight=args.dist_weight,
        use_dense_reward=True,
    )
    eval_env = make_assistive_fetch_env(
        env_id=args.env_id,
        render_mode=None,
        human_gain=args.human_gain,
        assist_scale=args.assist_scale,
        smoothness_coef=args.smoothness_coef,
        effort_coef=args.effort_coef,
        overassist_coef=args.overassist_coef,
        success_bonus=args.success_bonus,
        dist_weight=args.dist_weight,
        use_dense_reward=True,
    )

    train_env = Monitor(train_env)
    eval_env = Monitor(eval_env)

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.01, # 目标网络更新速度：SAC 稳定训练的 hyperparameter——越小越稳，越大越激进
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        tensorboard_log=args.log_dir,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best_model"),
        log_path=os.path.join(args.log_dir, "eval"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=os.path.join(args.log_dir, "checkpoints"),
        name_prefix="assistive_fetch_sac",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    callbacks = [eval_callback, checkpoint_callback]
    if args.plot:
        callbacks.append(
            PlottingCallback(
                check_freq=args.plot_freq,
                save_path=os.path.join(args.log_dir, "training_curves.png"),
                verbose=1,
            )
        )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
        )

        model.save(os.path.join(args.log_dir, "final_model"))
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()