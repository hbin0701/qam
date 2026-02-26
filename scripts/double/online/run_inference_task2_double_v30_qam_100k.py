#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[3]
import sys

sys.path.insert(0, str(PROJECT_DIR))

from agents import agents
from envs.ogbench_utils import make_ogbench_env_and_datasets
from envs.rewards.cube_dense_reward import DenseRewardWrapper
from evaluation import evaluate
from utils.datasets import Dataset
from utils.flax_utils import restore_agent_with_file


def _to_float(x):
    try:
        return float(x)
    except Exception:
        return x


def _pick_dataset_path(dataset_dir: str, env_name: str) -> str:
    data_dir = Path(dataset_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    env_type_filter = None
    for env_type in ["single", "double", "triple", "quadruple", "octuple"]:
        if f"{env_type}-play" in env_name:
            env_type_filter = env_type
            break

    candidates = []
    for file in sorted(data_dir.glob("*.npz")):
        name = file.name
        if name.endswith("-val.npz"):
            continue
        if ("visual" in env_name) != ("visual" in name):
            continue
        if env_type_filter is not None and f"-{env_type_filter}-" not in name:
            continue
        candidates.append(str(file))

    if not candidates:
        raise RuntimeError(
            f"No dataset .npz found in {dataset_dir} matching env_name={env_name} (env_type={env_type_filter})."
        )
    return candidates[0]


def _save_videos_and_json(
    out_dir: Path,
    renders,
    render_data,
    eval_seed: int,
    fps: int,
    final_pause_seconds: float,
    summary: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = []

    reward_traces = render_data.get("reward_traces", [])
    chunk_reward_traces = render_data.get("chunk_reward_traces", [])
    frame_steps = render_data.get("frame_steps", [])
    object_pose_traces = render_data.get("object_pose_traces", [])

    for ep_idx, frames in enumerate(renders):
        ep_num = ep_idx + 1
        seed = int(eval_seed + ep_idx)
        video_path = out_dir / f"episode_{ep_num:02d}.mp4"
        has_video = frames is not None and len(frames) > 0
        frames_to_save = frames
        if has_video and final_pause_seconds > 0:
            hold_n = max(1, int(round(final_pause_seconds * fps)))
            last_frame = np.array(frames[-1], copy=True)
            frames_to_save = np.concatenate(
                [frames, np.repeat(last_frame[None, ...], hold_n, axis=0)],
                axis=0,
            )
        if has_video:
            imageio.mimsave(video_path, frames_to_save, fps=fps)

        ep_record = {
            "episode": ep_num,
            "seed": seed,
            "video_path": str(video_path) if has_video else None,
            "num_frames": int(len(frames_to_save)) if frames_to_save is not None else 0,
            "video_saved": bool(has_video),
            "frame_steps": [],
            "reward_trace": [],
            "chunk_reward_trace": [],
            "object_pose_trace": [],
        }

        if ep_idx < len(frame_steps):
            try:
                ep_record["frame_steps"] = [int(x) for x in frame_steps[ep_idx].tolist()]
            except Exception:
                ep_record["frame_steps"] = [int(x) for x in frame_steps[ep_idx]]
        if ep_idx < len(reward_traces):
            try:
                ep_record["reward_trace"] = [float(x) for x in reward_traces[ep_idx].tolist()]
            except Exception:
                ep_record["reward_trace"] = [float(x) for x in reward_traces[ep_idx]]
        if ep_idx < len(chunk_reward_traces):
            try:
                ep_record["chunk_reward_trace"] = [float(x) for x in chunk_reward_traces[ep_idx].tolist()]
            except Exception:
                ep_record["chunk_reward_trace"] = [float(x) for x in chunk_reward_traces[ep_idx]]
        if ep_idx < len(object_pose_traces):
            ep_record["object_pose_trace"] = object_pose_traces[ep_idx]

        episode_json_path = out_dir / f"episode_{ep_num:02d}.json"
        with episode_json_path.open("w") as f:
            json.dump(
                {
                    "episode": ep_num,
                    "seed": seed,
                    "video_path": str(video_path) if has_video else None,
                    "num_frames": ep_record["num_frames"],
                    "frame_steps": ep_record["frame_steps"],
                    "reward_trace": ep_record["reward_trace"],
                    "chunk_reward_trace": ep_record["chunk_reward_trace"],
                    "object_pose_trace": ep_record["object_pose_trace"],
                },
                f,
            )
        ep_record["episode_json_path"] = str(episode_json_path)

        episodes.append(ep_record)

    summary_payload = {k: _to_float(v) for k, v in summary.items()}
    with (out_dir / "rollout_summary.json").open("w") as f:
        json.dump(
            {
                "num_episodes": len(episodes),
                "episodes": [
                    {
                        "episode": e["episode"],
                        "seed": e["seed"],
                        "video_path": e["video_path"],
                        "episode_json_path": e["episode_json_path"],
                    }
                    for e in episodes
                ],
                "summary": summary_payload,
            },
            f,
            indent=2,
        )

    with (out_dir / "rollout_traces.json").open("w") as f:
        json.dump(
            {
                "num_episodes": len(episodes),
                "episodes": episodes,
            },
            f,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "/rlwrld3/home/hyeonbin/RL/qam/exp/0219-double-tmp1/"
            "double_task2_rewards_online_save/cube-double-play-singletask-task2-v0/"
            "[Final, Online, SaveAll] Double, V30, QAM/params_100000.pkl"
        ),
    )
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--final-pause-seconds", type=float, default=0.2)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".20")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    run_dir = checkpoint_path.parent
    flags_path = run_dir / "flags.json"
    if not flags_path.is_file():
        raise FileNotFoundError(f"flags.json not found next to checkpoint: {flags_path}")

    with flags_path.open("r") as f:
        saved_flags = json.load(f)

    env_name = saved_flags["env_name"]
    seed = int(saved_flags["seed"])
    eval_seed = seed if args.eval_seed is None else int(args.eval_seed)
    dataset_num_samples = int(saved_flags.get("dataset_num_samples", 0))
    max_episode_steps = int(saved_flags.get("max_episode_steps", 0))
    horizon_length = int(saved_flags.get("horizon_length", 5))
    sparse = bool(saved_flags.get("sparse", False))
    dense_reward_version = saved_flags.get("dense_reward_version", None)
    terminal_bonus = float(saved_flags.get("terminal_bonus", 50.0))
    dense_shaping_lambda = float(saved_flags.get("dense_shaping_lambda", 10.0))
    cube_success_threshold = float(saved_flags.get("cube_success_threshold", 0.04))
    video_frame_skip = int(saved_flags.get("video_frame_skip", 1))

    dataset_dir = saved_flags.get("ogbench_dataset_dir", None)
    if dataset_dir is None:
        raise ValueError("Expected ogbench_dataset_dir in flags.json for this evaluation script.")
    dataset_path = _pick_dataset_path(dataset_dir, env_name)

    env_kwargs = {}
    if max_episode_steps > 0:
        env_kwargs["max_episode_steps"] = max_episode_steps

    _, eval_env, train_dataset_dict, _ = make_ogbench_env_and_datasets(
        env_name,
        dataset_path=dataset_path,
        compact_dataset=False,
        add_info=(dense_reward_version is not None),
        **env_kwargs,
    )

    train_dataset = Dataset.create(**train_dataset_dict)
    if dataset_num_samples > 0:
        cap = min(dataset_num_samples, len(train_dataset["masks"]))
        train_dataset = Dataset.create(**{k: v[:cap] for k, v in train_dataset.items()})

    example_batch = train_dataset.sample(())

    agent_cfg = saved_flags["agent"]
    agent_class = agents[agent_cfg["agent_name"]]
    agent = agent_class.create(
        seed,
        example_batch["observations"],
        example_batch["actions"],
        agent_cfg,
    )
    agent = restore_agent_with_file(agent, str(checkpoint_path))

    dense_wrapper = None
    if dense_reward_version is not None:
        dense_wrapper = DenseRewardWrapper(
            task_name=env_name,
            version=dense_reward_version,
            debug=False,
            success_threshold=cube_success_threshold,
            v23_step_penalty=float(saved_flags.get("v23_step_penalty", 1.0)),
        )

    # Pass 1: collect evaluation statistics over N episodes.
    eval_stats, _, _, _ = evaluate(
        agent=agent,
        env=eval_env,
        action_dim=example_batch["actions"].shape[-1],
        num_eval_episodes=args.num_episodes,
        num_video_episodes=0,
        video_frame_skip=video_frame_skip,
        sparse_reward=sparse,
        dense_wrapper=dense_wrapper,
        dense_discount=float(agent_cfg.get("discount", 0.99)),
        dense_terminal_bonus=terminal_bonus,
        dense_shaping_lambda=dense_shaping_lambda,
        eval_seed=eval_seed,
    )

    # Pass 2: render videos for the same seeds/episodes.
    _, _, renders, render_data = evaluate(
        agent=agent,
        env=eval_env,
        action_dim=example_batch["actions"].shape[-1],
        num_eval_episodes=0,
        num_video_episodes=args.num_episodes,
        video_frame_skip=video_frame_skip,
        sparse_reward=sparse,
        dense_wrapper=dense_wrapper,
        dense_discount=float(agent_cfg.get("discount", 0.99)),
        dense_terminal_bonus=terminal_bonus,
        dense_shaping_lambda=dense_shaping_lambda,
        eval_seed=eval_seed,
    )

    if args.output_dir is None:
        out_dir = run_dir / f"inference_rollout_{checkpoint_path.stem}_{args.num_episodes}eps"
    else:
        out_dir = Path(args.output_dir)

    _save_videos_and_json(
        out_dir=out_dir,
        renders=renders,
        render_data=render_data,
        eval_seed=eval_seed,
        fps=args.fps,
        final_pause_seconds=float(args.final_pause_seconds),
        summary=eval_stats,
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {out_dir}")
    print("Saved:")
    print(f"  - {out_dir / 'rollout_summary.json'}")
    print(f"  - {out_dir / 'rollout_traces.json'}")
    for i in range(args.num_episodes):
        print(f"  - {out_dir / f'episode_{i+1:02d}.mp4'}")
        print(f"  - {out_dir / f'episode_{i+1:02d}.json'}")


if __name__ == "__main__":
    main()
