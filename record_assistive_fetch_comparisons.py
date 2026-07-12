"""Record paired model+assist vs human-only FetchPush visualizations.

The script searches for a seed where the trained model succeeds while the
human-only baseline fails, then records the paired rollout with identical
initial conditions. It exports both MP4 and animated GIF files.
"""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC

from assistive_fetch.envs import make_assistive_fetch_env


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "logs" / "assistive_fetch" / "visualizations"
HUMAN_GAIN = 0.7
ASSIST_SCALE = 0.15
MAX_STEPS = 50
FPS = 10

VERSIONS = {
    "v3": ROOT / "logs" / "assistive_fetch" / "v3_final_model.zip",
    "v5": ROOT
    / "logs"
    / "assistive_fetch"
    / "v5_with_increased_smoothness_final_model.zip",
}


def make_env(render_mode=None):
    return make_assistive_fetch_env(
        env_id="FetchPush-v4",
        render_mode=render_mode,
        human_gain=HUMAN_GAIN,
        assist_scale=ASSIST_SCALE,
        success_bonus=10.0,
        dist_weight=1.0,
        use_dense_reward=True,
    )


def run_episode(env, policy, seed, capture=False):
    obs, _ = env.reset(seed=seed)
    frames = []
    infos = []
    if capture:
        frames.append(env.render())
        infos.append(
            {
                "assist_reward/dist": float(
                    np.linalg.norm(obs["desired_goal"] - obs["achieved_goal"])
                ),
                "assist_reward/success": 0.0,
                "human_phase": 0.0,
            }
        )

    final_info = {}
    for _ in range(MAX_STEPS):
        action = policy(obs)
        obs, _, terminated, truncated, final_info = env.step(action)
        if capture:
            frames.append(env.render())
            infos.append(dict(final_info))
        if terminated or truncated:
            break

    return {
        "success": float(final_info.get("assist_reward/success", 0.0)),
        "dist": float(final_info.get("assist_reward/dist", np.nan)),
        "frames": frames,
        "infos": infos,
    }


def find_demo_seed(model_path, max_seed=300):
    env = make_env(render_mode=None)
    model = SAC.load(model_path, env=env)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)

    def model_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action.astype(np.float32)

    def human_policy(_obs):
        return zero_action

    candidates = []
    for seed in range(max_seed):
        model_result = run_episode(env, model_policy, seed)
        human_result = run_episode(env, human_policy, seed)
        if model_result["success"] >= 0.5 and human_result["success"] < 0.5:
            separation = human_result["dist"] - model_result["dist"]
            if separation >= 0.15:
                env.close()
                return seed, model_result, human_result
            candidates.append((separation, seed, model_result, human_result))

    env.close()
    if not candidates:
        raise RuntimeError(f"No successful comparison seed found for {model_path}")
    candidates.sort(reverse=True, key=lambda item: item[0])
    _, seed, model_result, human_result = candidates[0]
    return seed, model_result, human_result


def load_font(size, bold=False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        if bold
        else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def add_panel_overlay(frame, label, info, color):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = load_font(20, bold=True)
    metric_font = load_font(15)

    draw.rounded_rectangle((12, 12, 235, 48), radius=8, fill=(*color, 225))
    draw.text((24, 20), label, font=title_font, fill=(255, 255, 255, 255))

    dist = float(info.get("assist_reward/dist", np.nan))
    success = int(float(info.get("assist_reward/success", 0.0)) >= 0.5)
    phase = int(float(info.get("human_phase", 0.0)))
    metric = f"dist={dist:.3f} m   phase={phase}   success={success}"
    draw.rounded_rectangle((12, image.height - 42, 348, image.height - 10), radius=7, fill=(20, 34, 54, 205))
    draw.text((22, image.height - 35), metric, font=metric_font, fill=(255, 255, 255, 255))
    return np.asarray(image)


def compose_pair(model_result, human_result, version, seed):
    count = max(len(model_result["frames"]), len(human_result["frames"]))
    frames = []
    header_font = load_font(20, bold=True)

    for i in range(count):
        mi = min(i, len(model_result["frames"]) - 1)
        hi = min(i, len(human_result["frames"]) - 1)
        left = add_panel_overlay(
            model_result["frames"][mi],
            f"{version.upper()}  MODEL + ASSIST",
            model_result["infos"][mi],
            (25, 145, 145),
        )
        right = add_panel_overlay(
            human_result["frames"][hi],
            "HUMAN ONLY",
            human_result["infos"][hi],
            (230, 132, 45),
        )
        pair = np.concatenate([left, right], axis=1)
        canvas = Image.new("RGB", (pair.shape[1], pair.shape[0] + 46), (20, 34, 54))
        canvas.paste(Image.fromarray(pair), (0, 46))
        draw = ImageDraw.Draw(canvas)
        title = f"Same initial condition (seed={seed})  |  step {min(i, MAX_STEPS):02d}/{MAX_STEPS}"
        bbox = draw.textbbox((0, 0), title, font=header_font)
        x = (canvas.width - (bbox[2] - bbox[0])) // 2
        draw.text((x, 11), title, font=header_font, fill=(255, 255, 255))
        frames.append(np.asarray(canvas))
    return frames


def record_version(version, model_path):
    seed, quick_model, quick_human = find_demo_seed(model_path)
    print(
        f"{version}: selected seed={seed}, "
        f"model success={quick_model['success']:.0f} dist={quick_model['dist']:.3f}, "
        f"human success={quick_human['success']:.0f} dist={quick_human['dist']:.3f}"
    )

    model_env = make_env(render_mode="rgb_array")
    human_env = make_env(render_mode="rgb_array")
    model = SAC.load(model_path, env=model_env)
    zero_action = np.zeros(human_env.action_space.shape, dtype=np.float32)

    def model_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action.astype(np.float32)

    def human_policy(_obs):
        return zero_action

    model_result = run_episode(model_env, model_policy, seed, capture=True)
    human_result = run_episode(human_env, human_policy, seed, capture=True)
    model_env.close()
    human_env.close()

    frames = compose_pair(model_result, human_result, version, seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mp4_path = OUT_DIR / f"{version}_assist_vs_human.mp4"
    gif_path = OUT_DIR / f"{version}_assist_vs_human.gif"
    poster_path = OUT_DIR / f"{version}_assist_vs_human_poster.png"

    with imageio.get_writer(
        mp4_path,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)

    # GIFs embedded in PPT play directly in PowerPoint slideshow mode.
    imageio.mimsave(gif_path, frames, duration=1000 / FPS, loop=0, palettesize=128)
    Image.fromarray(frames[min(20, len(frames) - 1)]).save(poster_path)

    print(
        f"Saved {mp4_path.name}, {gif_path.name}; "
        f"final model dist={model_result['dist']:.3f}, "
        f"human dist={human_result['dist']:.3f}"
    )
    return mp4_path, gif_path, poster_path


def main():
    for version, model_path in VERSIONS.items():
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        record_version(version, model_path)


if __name__ == "__main__":
    main()
