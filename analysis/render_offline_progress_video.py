#!/usr/bin/env python3
"""Render one offline trajectory with progress plots and save as MP4."""

import argparse
import os
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.ogbench_utils import load_dataset
from envs.rewards.cube_dense_reward import (
    DenseRewardWrapper,
    V6_GRASP_LIFT_TARGET,
    extract_gripper_pos,
    progress_v7,
)
from log_utils import _make_metric_plot_frame, _make_reward_plot_frame


def _episode_bounds(terminals: np.ndarray):
    ends = np.where(terminals > 0)[0]
    if len(ends) == 0:
        return np.array([0], dtype=np.int32), np.array([len(terminals) - 1], dtype=np.int32)
    starts = np.concatenate([[0], ends[:-1] + 1]).astype(np.int32)
    return starts, ends.astype(np.int32)


def _pick_episode(starts, ends, episode_idx, seed):
    lengths = ends - starts + 1
    if episode_idx is not None:
        if episode_idx < 0 or episode_idx >= len(starts):
            raise ValueError(f"episode_idx={episode_idx} out of range [0, {len(starts)-1}]")
        return episode_idx
    rng = np.random.default_rng(seed)
    valid = np.where(lengths > 1)[0]
    if len(valid) == 0:
        return 0
    return int(rng.choice(valid))


def _to_gym_env_name(dataset_env_name: str) -> str:
    splits = dataset_env_name.split("-")
    if "singletask" in splits:
        pos = splits.index("singletask")
        return "-".join(splits[: pos - 1] + splits[pos:])
    if "oraclerep" in splits:
        return "-".join(splits[:-3] + splits[-1:])
    return "-".join(splits[:-2] + splits[-1:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_mp4", type=str, required=True)
    parser.add_argument("--dense_reward_version", type=str, default="v25")
    parser.add_argument("--episode_idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()

    os.environ.setdefault("MUJOCO_GL", "osmesa")

    dataset = load_dataset(
        dataset_path=args.dataset_path,
        compact_dataset=False,
        add_info=True,
    )
    for req in ("qpos", "next_qpos", "terminals"):
        if req not in dataset:
            raise ValueError(f"Dataset missing required key: {req}")

    starts, ends = _episode_bounds(dataset["terminals"])
    ep_idx = _pick_episode(starts, ends, args.episode_idx, args.seed)
    s, e = int(starts[ep_idx]), int(ends[ep_idx])

    qpos_seq = dataset["qpos"][s : e + 1]
    next_qpos_seq = dataset["next_qpos"][s : e + 1]
    qvel_seq = dataset["qvel"][s : e + 1] if "qvel" in dataset else None
    next_qvel_seq = dataset["next_qvel"][s : e + 1] if "next_qvel" in dataset else None
    obs_seq = dataset["observations"][s : e + 1] if "observations" in dataset else None
    next_obs_seq = dataset["next_observations"][s : e + 1] if "next_observations" in dataset else None
    env_reward_seq = dataset["rewards"][s : e + 1] if "rewards" in dataset else np.zeros(len(qpos_seq), dtype=np.float32)
    if args.max_frames > 0:
        qpos_seq = qpos_seq[: args.max_frames]
        next_qpos_seq = next_qpos_seq[: args.max_frames]
        if qvel_seq is not None:
            qvel_seq = qvel_seq[: args.max_frames]
        if next_qvel_seq is not None:
            next_qvel_seq = next_qvel_seq[: args.max_frames]
        if obs_seq is not None:
            obs_seq = obs_seq[: args.max_frames]
        if next_obs_seq is not None:
            next_obs_seq = next_obs_seq[: args.max_frames]
        env_reward_seq = env_reward_seq[: args.max_frames]

    gym_env_name = _to_gym_env_name(args.env_name)
    env = gym.make(gym_env_name, width=args.width, height=args.height)
    env.reset()

    dense = DenseRewardWrapper(
        task_name=args.env_name,
        version=args.dense_reward_version,
        success_threshold=0.02,
    )

    progress = []
    target_state = None
    use_dynamic_target = args.dense_reward_version in ("v24", "v25")
    for i in range(len(qpos_seq)):
        gp = extract_gripper_pos(obs_seq[i]) if obs_seq is not None else None
        qvel_i = qvel_seq[i] if qvel_seq is not None else None

        if use_dynamic_target:
            env_state = dense._make_env()
            env_state.load_state(qpos_seq[i], qvel=qvel_i, gripper_pos=gp)
            env_state.grasp_lift_threshold_override = V6_GRASP_LIFT_TARGET

            target = target_state
            if target is None or env_state.cubes[int(target)].is_at_goal():
                target = dense._first_incomplete_cube_idx(env_state)
            ob_i = obs_seq[i] if obs_seq is not None else None
            touched = dense._detect_touched_cube_idx(env_state, ob_i)
            if touched is not None:
                target = touched
            if target is not None:
                env_state.active_cube_override = int(target)
            target_state = target

            p, _ = progress_v7(env_state)
        else:
            p, _ = dense.compute_progress(qpos_seq[i], qvel=qvel_i, gripper_pos=gp)

        progress.append(float(p))
    progress = np.asarray(progress, dtype=np.float32)
    potential_diff = np.zeros_like(progress)
    if len(progress) > 1:
        potential_diff[:-1] = 0.99 * progress[1:] - progress[:-1]
    total_reward = np.zeros_like(progress)
    dense.set_episode_initial_positions_from_qpos(qpos_seq[0])
    for i in range(len(qpos_seq)):
        qvel_i = qvel_seq[i] if qvel_seq is not None else None
        next_qvel_i = next_qvel_seq[i] if next_qvel_seq is not None else None
        prev_ob_i = obs_seq[i] if obs_seq is not None else None
        curr_ob_i = next_obs_seq[i] if next_obs_seq is not None else None
        total_reward[i] = dense.compute_online_reward(
            prev_qpos=qpos_seq[i],
            curr_qpos=next_qpos_seq[i],
            env_reward=float(env_reward_seq[i]),
            prev_ob=prev_ob_i,
            curr_ob=curr_ob_i,
            discount=0.99,
            terminal_bonus=1.0,
            shaping_coef=1.0,
        )

    os.makedirs(os.path.dirname(args.output_mp4) or ".", exist_ok=True)
    writer = imageio.get_writer(args.output_mp4, fps=args.fps, codec="libx264")
    try:
        for i in range(len(qpos_seq)):
            qvel_i = qvel_seq[i] if qvel_seq is not None else np.zeros(env.unwrapped.model.nv, dtype=np.float64)
            env.unwrapped.set_state(qpos_seq[i], qvel_i)
            frame = env.render().copy()

            h1 = frame.shape[0] // 3
            h2 = frame.shape[0] // 3
            h3 = frame.shape[0] - h1 - h2
            progress_plot = _make_metric_plot_frame(progress, i, h1, frame.shape[1], "progress")
            pot_plot = _make_metric_plot_frame(potential_diff, i, h2, frame.shape[1], "potential_diff")
            reward_plot = _make_reward_plot_frame(total_reward, i, h3, frame.shape[1])
            panel = np.concatenate([progress_plot, pot_plot, reward_plot], axis=0)
            composite = np.concatenate([frame, panel], axis=1)
            writer.append_data(composite)
    finally:
        writer.close()
        env.close()

    print(f"Saved: {args.output_mp4}")
    print(f"episode_idx={ep_idx}, steps={len(qpos_seq)}, start={s}, end={e}")


if __name__ == "__main__":
    main()
