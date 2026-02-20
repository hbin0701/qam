from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
from functools import partial
from contextlib import contextmanager
import fcntl
import os
from envs.rewards.cube_dense_reward import extract_gripper_pos


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, rng=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


@contextmanager
def _render_lock_if_needed():
    """Serialize EGL rendering on the same GPU across processes.

    MuJoCo EGL offscreen context creation can fail when many jobs render concurrently on one GPU.
    We use a cooperative file lock keyed by MUJOCO_EGL_DEVICE_ID.
    """
    use_lock = os.environ.get("MUJOCO_GL", "").lower() == "egl"
    if not use_lock:
        yield
        return

    gpu_id = os.environ.get("MUJOCO_EGL_DEVICE_ID", "0")
    lock_path = f"/tmp/mujoco_egl_render_gpu_{gpu_id}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def evaluate(
    agent,
    env,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    action_shape=None,
    observation_shape=None,
    action_dim=None,
    extra_sample_kwargs={},
    sparse_reward=False,
    dense_wrapper=None,
    dense_discount=0.99,
    dense_terminal_bonus=50.0,
    dense_shaping_lambda=10.0,
    eval_seed=None,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing:
          - statistics dict
          - trajectories
          - rendered videos
          - render metadata dict with reward traces and rendered frame step indices
    """
    actor_fn = supply_rng(partial(agent.sample_actions, **extra_sample_kwargs), rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))
    trajs = []
    stats = defaultdict(list)
    render_disabled = False

    renders = []
    render_reward_traces = []
    render_chunk_reward_traces = []
    render_progress_traces = []
    render_potential_diff_traces = []
    render_cube_z_traces = []
    render_lower_entry_z_traces = []
    render_cube_lift_traces = []
    render_gripper_width_traces = []
    render_frame_steps = []
    progress_video_enabled = (
        dense_wrapper is not None
        and getattr(dense_wrapper, "version", None) in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18")
    )
    v8_chunk_shaping = dense_wrapper is not None and getattr(dense_wrapper, "version", None) in ("v8", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18")
    z_trace_enabled = dense_wrapper is not None and getattr(dense_wrapper, "version", None) in ("v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18")
    lift_trace_enabled = dense_wrapper is not None and getattr(dense_wrapper, "version", None) in ("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18")
    saved_episode_init_positions = None
    if dense_wrapper is not None:
        saved_episode_init_positions = getattr(dense_wrapper, "episode_init_positions", None)
        if saved_episode_init_positions is not None:
            saved_episode_init_positions = saved_episode_init_positions.copy()
    for i in trange(num_eval_episodes + num_video_episodes):
        traj = defaultdict(list)
        should_render = i >= num_eval_episodes

        if eval_seed is not None:
            try:
                observation, info = env.reset(seed=int(eval_seed + i))
            except TypeError:
                observation, info = env.reset()
        else:
            observation, info = env.reset()
        prev_qpos_dense = None
        if dense_wrapper is not None:
            try:
                prev_qpos_dense = env.unwrapped._data.qpos.copy()
                dense_wrapper.set_episode_initial_positions_from_qpos(prev_qpos_dense)
            except Exception:
                prev_qpos_dense = None
        chunk_start_qpos_dense = None
        chunk_start_ob_dense = None
        chunk_step_count_dense = 0
            
        observation_history = []
        action_history = []
        
        done = False
        step = 0
        render = []
        render_steps = []
        reward_trace = []
        chunk_reward_trace = []
        progress_trace = []
        potential_diff_trace = []
        cube_z_trace = []
        lower_entry_z_trace = []
        cube_lift_trace = []
        gripper_width_trace = []
        prev_progress_for_vis = None
        chunk_start_progress_vis = None
        action_chunk_lens = defaultdict(lambda: 0)

        action_queue = []

        gripper_contact_lengths = []
        gripper_contact_length = 0
        while not done:
            
            action = actor_fn(observations=observation)

            if len(action_queue) == 0:
                have_new_action = True
                action = np.array(action).reshape(-1, action_dim)
                action_chunk_len = action.shape[0]
                for a in action:
                    action_queue.append(a)
                if v8_chunk_shaping and prev_qpos_dense is not None:
                    chunk_start_qpos_dense = prev_qpos_dense.copy()
                    chunk_start_ob_dense = np.array(observation, copy=True)
                    chunk_step_count_dense = 0
                    if progress_video_enabled:
                        try:
                            chunk_start_progress_vis, _ = dense_wrapper.compute_progress_for_logging(
                                prev_qpos_dense,
                                ob=observation,
                            )
                        except Exception:
                            chunk_start_progress_vis = None
            else:
                have_new_action = False
            
            action = action_queue.pop(0)
            if eval_gaussian is not None:
                action = np.random.normal(action, eval_gaussian)

            next_observation, reward, terminated, truncated, info = env.step(np.clip(action, -1, 1))
            done = terminated or truncated
            step += 1
            curr_qpos_dense = None
            logged_reward = float(reward)
            chunk_reward_trace.append(np.nan)
            if sparse_reward:
                logged_reward = float((reward != 0.0) * -1.0)
            elif dense_wrapper is not None and prev_qpos_dense is not None:
                try:
                    curr_qpos_dense = env.unwrapped._data.qpos.copy()
                    if v8_chunk_shaping:
                        base_plus_events = dense_wrapper.compute_online_reward(
                            prev_qpos=prev_qpos_dense,
                            curr_qpos=curr_qpos_dense,
                            env_reward=float(reward),
                            prev_ob=observation,
                            curr_ob=next_observation,
                            discount=dense_discount,
                            terminal_bonus=dense_terminal_bonus,
                            shaping_coef=0.0,
                        )
                        chunk_step_count_dense += 1
                        is_chunk_end = (len(action_queue) == 0) or done
                        curr_progress, _ = dense_wrapper.compute_progress_for_logging(
                            curr_qpos_dense,
                            ob=next_observation,
                        )
                        chunk_potential_diff = 0.0
                        if is_chunk_end and chunk_start_qpos_dense is not None and chunk_start_ob_dense is not None:
                            prev_progress = chunk_start_progress_vis
                            if prev_progress is None:
                                prev_progress, _ = dense_wrapper.compute_progress_for_logging(
                                    chunk_start_qpos_dense,
                                    ob=chunk_start_ob_dense,
                                )
                            chunk_potential_diff = float(
                                (dense_discount ** chunk_step_count_dense) * curr_progress - prev_progress
                            )
                        chunk_reward = float(dense_shaping_lambda * chunk_potential_diff)
                        logged_reward = float(base_plus_events + chunk_reward)
                        if is_chunk_end:
                            fill_chunk = float(chunk_reward)
                            for back_i in range(chunk_step_count_dense):
                                idx = len(chunk_reward_trace) - 1 - back_i
                                if idx >= 0:
                                    chunk_reward_trace[idx] = fill_chunk
                        if progress_video_enabled:
                            progress_trace.append(float(curr_progress))
                            # For visualization, show the same chunk-level potential diff
                            # value across all steps in this chunk.
                            potential_diff_trace.append(np.nan)
                            if is_chunk_end:
                                fill_val = float(chunk_potential_diff)
                                for back_i in range(chunk_step_count_dense):
                                    idx = len(potential_diff_trace) - 1 - back_i
                                    if idx >= 0:
                                        potential_diff_trace[idx] = fill_val
                        if is_chunk_end:
                            chunk_start_qpos_dense = None
                            chunk_start_ob_dense = None
                            chunk_step_count_dense = 0
                            chunk_start_progress_vis = None
                        prev_progress_for_vis = curr_progress
                    else:
                        logged_reward = float(
                            dense_wrapper.compute_online_reward(
                                prev_qpos=prev_qpos_dense,
                                curr_qpos=curr_qpos_dense,
                                env_reward=float(reward),
                                prev_ob=observation,
                                curr_ob=next_observation,
                                discount=dense_discount,
                                terminal_bonus=dense_terminal_bonus,
                                shaping_coef=dense_shaping_lambda,
                            )
                        )
                        if progress_video_enabled:
                            curr_progress, _ = dense_wrapper.compute_progress_for_logging(
                                curr_qpos_dense,
                                ob=next_observation,
                            )
                            if prev_progress_for_vis is None:
                                prev_progress_for_vis, _ = dense_wrapper.compute_progress_for_logging(
                                    prev_qpos_dense,
                                    ob=observation,
                                )
                            progress_trace.append(float(curr_progress))
                            potential_diff_trace.append(float(dense_discount * curr_progress - prev_progress_for_vis))
                            prev_progress_for_vis = curr_progress
                    prev_qpos_dense = curr_qpos_dense
                except Exception:
                    logged_reward = float(reward)
                    if progress_video_enabled:
                        progress_trace.append(np.nan)
                        potential_diff_trace.append(np.nan)
            elif progress_video_enabled:
                progress_trace.append(np.nan)
                potential_diff_trace.append(np.nan)
            reward_trace.append(logged_reward)
            if z_trace_enabled:
                try:
                    qpos_for_z = curr_qpos_dense
                    if qpos_for_z is None:
                        qpos_for_z = env.unwrapped._data.qpos.copy()
                    gp_for_z = extract_gripper_pos(next_observation)
                    z_env = dense_wrapper._make_env()
                    z_env.load_state(qpos_for_z, gripper_pos=gp_for_z)
                    active_idx = dense_wrapper._active_cube_idx(z_env)
                    cube_z = np.nan if active_idx is None else float(z_env.cubes[active_idx].position[2])
                    lower_entry_attr = f"_{dense_wrapper.version}_lower_entry_z"
                    lower_entry_z = float(getattr(dense_wrapper, lower_entry_attr)) if hasattr(dense_wrapper, lower_entry_attr) and getattr(dense_wrapper, lower_entry_attr) is not None else np.nan
                except Exception:
                    cube_z = np.nan
                    lower_entry_z = np.nan
                cube_z_trace.append(float(cube_z))
                lower_entry_z_trace.append(float(lower_entry_z))
            if lift_trace_enabled:
                try:
                    qpos_for_lift = curr_qpos_dense
                    if qpos_for_lift is None:
                        qpos_for_lift = env.unwrapped._data.qpos.copy()
                    gp_for_lift = extract_gripper_pos(next_observation)
                    lift_env = dense_wrapper._make_env()
                    lift_env.load_state(qpos_for_lift, gripper_pos=gp_for_lift)
                    active_idx = dense_wrapper._active_cube_idx(lift_env)
                    if active_idx is None:
                        cube_lift = np.nan
                    else:
                        cube = lift_env.cubes[active_idx]
                        cube_lift = float(cube.position[2] - cube.initial_position[2])
                    gripper_width = float(lift_env.gripper_width)
                except Exception:
                    cube_lift = np.nan
                    gripper_width = np.nan
                cube_lift_trace.append(float(cube_lift))
                gripper_width_trace.append(float(gripper_width))

            if dense_wrapper is not None and prev_qpos_dense is not None:
                try:
                    num_success_cubes = dense_wrapper.count_success_cubes(prev_qpos_dense)
                    info["num_success_cubes"] = float(num_success_cubes)
                    info["success_cube_fraction"] = float(num_success_cubes / max(dense_wrapper.num_cubes, 1))
                except Exception:
                    pass

            if should_render and (step % video_frame_skip == 0 or done) and not render_disabled:
                try:
                    with _render_lock_if_needed():
                        frame = env.render().copy()
                    render.append(frame)
                    render_steps.append(step - 1)
                except Exception as e:
                    # Keep training/eval alive when EGL offscreen rendering is unavailable.
                    render_disabled = True
                    print(f"[WARN] Disabling eval video rendering after render failure: {e}", flush=True)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            
            observation = next_observation
            if "proprio" in info and "gripper_contact" in info["proprio"]:
                gripper_contact = info["proprio"]["gripper_contact"]
            elif "gripper_contact" in info:
                gripper_contact = info["gripper_contact"]
            else:
                gripper_contact = None

            if gripper_contact is not None:
                if info["gripper_contact"] > 0.1:
                    gripper_contact_length += 1
                else:
                    if gripper_contact_length > 0:
                        gripper_contact_lengths.append(gripper_contact_length)
                    gripper_contact_length = 0

        if gripper_contact_length > 0:
            gripper_contact_lengths.append(gripper_contact_length)
        
        num_gripper_contacts = len(gripper_contact_lengths)

        if num_gripper_contacts > 0:
            avg_gripper_contact_length = np.mean(np.array(gripper_contact_lengths))
        else:
            avg_gripper_contact_length = 0
            
        add_to(stats, {"avg_gripper_contact_length": avg_gripper_contact_length, "num_gripper_contacts": num_gripper_contacts})

        if i < num_eval_episodes:
            add_to(stats, flatten(info))
            trajs.append(traj)
        else:
            renders.append(np.array(render))
            render_reward_traces.append(np.array(reward_trace, dtype=np.float32))
            render_chunk_reward_traces.append(np.array(chunk_reward_trace, dtype=np.float32))
            if progress_video_enabled:
                render_progress_traces.append(np.array(progress_trace, dtype=np.float32))
                render_potential_diff_traces.append(np.array(potential_diff_trace, dtype=np.float32))
            if z_trace_enabled:
                render_cube_z_traces.append(np.array(cube_z_trace, dtype=np.float32))
                render_lower_entry_z_traces.append(np.array(lower_entry_z_trace, dtype=np.float32))
            if lift_trace_enabled:
                render_cube_lift_traces.append(np.array(cube_lift_trace, dtype=np.float32))
                render_gripper_width_traces.append(np.array(gripper_width_trace, dtype=np.float32))
            render_frame_steps.append(np.array(render_steps, dtype=np.int32))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if dense_wrapper is not None and hasattr(dense_wrapper, "episode_init_positions"):
        dense_wrapper.episode_init_positions = saved_episode_init_positions

    render_data = {
        "reward_traces": render_reward_traces,
        "chunk_reward_traces": render_chunk_reward_traces,
        "progress_traces": render_progress_traces,
        "potential_diff_traces": render_potential_diff_traces,
        "cube_z_traces": render_cube_z_traces,
        "lower_entry_z_traces": render_lower_entry_z_traces,
        "cube_lift_traces": render_cube_lift_traces,
        "gripper_width_traces": render_gripper_width_traces,
        "frame_steps": render_frame_steps,
    }
    return stats, trajs, renders, render_data
