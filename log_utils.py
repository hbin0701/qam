import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
from PIL import Image, ImageEnhance
import glob

import shutil

import csv


def get_csv_header(path):
    with open(path, 'r') as f:
        # Use DictReader to automatically interpret the first row as headers
        dict_reader = csv.DictReader(f)
        return dict_reader.fieldnames

class CsvLogger:

    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()

    def save(self, dst_path):
        if self.file is not None:
            self.file.close()
            src_path = self.path
            try:
                shutil.copyfile(src_path, dst_path)
            except FileNotFoundError:
                print(f"Error: '{src_path}' not found.")
            except Exception as e:
                print(f"An error occurred: {e}")

            self.file = open(self.path, 'a')

    def restore(self, src_path):
        dst_path = self.path
        try:
            shutil.copyfile(src_path, dst_path)
        except FileNotFoundError:
            print(f"Error: '{src_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        self.header = get_csv_header(self.path)
        self.file = open(self.path, 'a')


def get_hash(s):
    import hashlib
    encoded_string = s.encode('utf-8')

    sha256_hash = hashlib.sha256()
    sha256_hash.update(encoded_string)
    hex_digest = sha256_hash.hexdigest()
    return hex_digest

def get_exp_name(flags):
    """Return the experiment name."""
    s = flags.flags_into_string()
    exp_name = s
    return get_hash(exp_name)


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    tags=None,
    name=None,
    mode='online',
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()

    if tags is None:
        tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, (n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, (t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')


def _draw_line(img, x0, y0, x1, y1, color):
    """Draw a line segment on an HWC uint8 image with simple interpolation."""
    h, w = img.shape[:2]
    n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if n <= 0:
        return
    xs = np.linspace(x0, x1, n).astype(np.int32)
    ys = np.linspace(y0, y1, n).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    img[ys[valid], xs[valid]] = color


def _make_reward_plot_frame(rewards, cursor_step, height, width):
    """Create a reward-vs-time plot image with a moving cursor line."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if rewards is None or len(rewards) == 0:
        return canvas

    # Plot margins.
    left, right = 32, 10
    top, bottom = 10, 24
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)

    # Axes.
    axis_color = np.array([80, 80, 80], dtype=np.uint8)
    _draw_line(canvas, left, top, left, top + plot_h, axis_color)
    _draw_line(canvas, left, top + plot_h, left + plot_w, top + plot_h, axis_color)

    r = np.asarray(rewards, dtype=np.float32)
    n = len(r)
    if n == 1:
        r_min, r_max = float(r[0] - 1.0), float(r[0] + 1.0)
    else:
        r_min, r_max = float(np.min(r)), float(np.max(r))
        if abs(r_max - r_min) < 1e-8:
            r_min -= 1.0
            r_max += 1.0

    # Reward curve.
    curve_color = np.array([52, 120, 246], dtype=np.uint8)
    xs = left + np.round(np.arange(n) * (plot_w / max(1, n - 1))).astype(np.int32)
    ys = top + np.round((1.0 - (r - r_min) / (r_max - r_min)) * plot_h).astype(np.int32)
    ys = np.clip(ys, top, top + plot_h)
    for i in range(1, n):
        _draw_line(canvas, xs[i - 1], ys[i - 1], xs[i], ys[i], curve_color)

    # Moving cursor.
    cursor_step = int(np.clip(cursor_step, 0, n - 1))
    cx = left + int(round(cursor_step * (plot_w / max(1, n - 1))))
    cursor_color = np.array([230, 30, 30], dtype=np.uint8)
    _draw_line(canvas, cx, top, cx, top + plot_h, cursor_color)

    return canvas


def get_wandb_video_with_reward(renders, reward_traces, frame_steps, n_cols=None, fps=15):
    """Return a W&B video with environment frames + synchronized reward-time plot.

    Args:
        renders: list of (t_i, h, w, c) uint8 arrays.
        reward_traces: list of (T_i,) reward arrays per episode.
        frame_steps: list of (t_i,) indices mapping each rendered frame to env step.
    """
    if renders is None or len(renders) == 0:
        return None

    composite_renders = []
    for i, render in enumerate(renders):
        if len(render) == 0:
            continue
        rewards = reward_traces[i] if i < len(reward_traces) else np.zeros((1,), dtype=np.float32)
        steps = frame_steps[i] if i < len(frame_steps) else np.arange(len(render), dtype=np.int32)
        episode_frames = []
        for j, frame in enumerate(render):
            step_idx = steps[j] if j < len(steps) else (len(rewards) - 1)
            plot = _make_reward_plot_frame(rewards, step_idx, frame.shape[0], frame.shape[1])
            episode_frames.append(np.concatenate([frame, plot], axis=1))
        composite_renders.append(np.asarray(episode_frames, dtype=np.uint8))

    if len(composite_renders) == 0:
        return None
    return get_wandb_video(composite_renders, n_cols=n_cols, fps=fps)
