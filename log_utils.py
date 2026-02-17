import os
import tempfile
from datetime import datetime
import json
from collections.abc import Mapping

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
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


class JsonlLogger:
    """JSONL logger for structured metric logging."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.file = open(self.path, 'a')

    @staticmethod
    def _to_jsonable(v):
        if isinstance(v, Mapping):
            return {str(k): JsonlLogger._to_jsonable(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [JsonlLogger._to_jsonable(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, np.bool_):
            return bool(v)
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if hasattr(v, "__array__"):
            try:
                arr = np.asarray(v)
                if arr.ndim == 0:
                    return arr.item()
                return arr.tolist()
            except Exception:
                pass
        if hasattr(v, "tolist"):
            try:
                return v.tolist()
            except Exception:
                pass
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                pass
        if isinstance(v, (wandb.Image, wandb.Video, wandb.Histogram)):
            return None
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    def log(self, row, step):
        payload = {'step': int(step), 'timestamp': datetime.utcnow().isoformat() + 'Z'}
        for k, v in row.items():
            payload[k] = self._to_jsonable(v)
        self.file.write(json.dumps(payload, ensure_ascii=True) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


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


def _format_tick(v):
    """Format numeric axis ticks compactly."""
    av = abs(float(v))
    if av >= 100:
        return f"{v:.0f}"
    if av >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _make_reward_plot_frame(rewards, cursor_step, height, width):
    """Create a reward-vs-time plot image with a moving cursor line."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if rewards is None or len(rewards) == 0:
        return canvas

    # Plot margins.
    left, right = 44, 14
    top, bottom = 16, 30
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)

    # Plot area background.
    canvas[top : top + plot_h + 1, left : left + plot_w + 1] = np.array([252, 252, 253], dtype=np.uint8)

    # Axes.
    axis_color = np.array([92, 102, 112], dtype=np.uint8)
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

    # Grid + numeric ticks.
    tick_color = np.array([170, 180, 190], dtype=np.uint8)
    y_ticks = np.linspace(r_min, r_max, 5)
    x_ticks = [0, max(0, n // 2), max(0, n - 1)]
    for yv in y_ticks:
        y = top + int(round((1.0 - (yv - r_min) / (r_max - r_min)) * plot_h))
        _draw_line(canvas, left, y, left + plot_w, y, tick_color)
    for xv in x_ticks:
        x = left + int(round(xv * (plot_w / max(1, n - 1))))
        _draw_line(canvas, x, top, x, top + plot_h, tick_color)

    # Reward curve.
    curve_color = np.array([34, 119, 217], dtype=np.uint8)
    xs = left + np.round(np.arange(n) * (plot_w / max(1, n - 1))).astype(np.int32)
    ys = top + np.round((1.0 - (r - r_min) / (r_max - r_min)) * plot_h).astype(np.int32)
    ys = np.clip(ys, top, top + plot_h)
    for i in range(1, n):
        # Slightly thicker line for readability in video.
        _draw_line(canvas, xs[i - 1], ys[i - 1] + 1, xs[i], ys[i] + 1, curve_color)
        _draw_line(canvas, xs[i - 1], ys[i - 1], xs[i], ys[i], curve_color)

    # Moving cursor.
    cursor_step = int(np.clip(cursor_step, 0, n - 1))
    cx = left + int(round(cursor_step * (plot_w / max(1, n - 1))))
    cursor_color = np.array([234, 67, 53], dtype=np.uint8)
    _draw_line(canvas, cx, top, cx, top + plot_h, cursor_color)

    # Current point marker.
    cy = top + int(round((1.0 - (r[cursor_step] - r_min) / (r_max - r_min)) * plot_h))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < height and 0 <= xx < width:
                canvas[yy, xx] = cursor_color

    # Add numeric axis labels and compact readout text.
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text_color = (65, 75, 85)

    for yv in y_ticks:
        y = top + int(round((1.0 - (yv - r_min) / (r_max - r_min)) * plot_h))
        draw.text((4, y - 5), _format_tick(yv), fill=text_color, font=font)

    for xv in x_ticks:
        x = left + int(round(xv * (plot_w / max(1, n - 1))))
        draw.text((x - 8, top + plot_h + 6), str(int(xv)), fill=text_color, font=font)

    draw.text((left, 2), "reward", fill=text_color, font=font)
    draw.text((left + plot_w - 74, 2), f"step {cursor_step}", fill=(90, 98, 108), font=font)
    draw.text((left + plot_w - 74, 12), f"r {r[cursor_step]:.3f}", fill=(90, 98, 108), font=font)

    return np.asarray(img, dtype=np.uint8)


def _make_metric_plot_frame(values, cursor_step, height, width, title):
    """Create a generic metric-vs-time plot image with a moving cursor line."""
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    if values is None or len(values) == 0:
        return canvas

    left, right = 44, 14
    top, bottom = 16, 30
    plot_w = max(1, width - left - right)
    plot_h = max(1, height - top - bottom)

    canvas[top : top + plot_h + 1, left : left + plot_w + 1] = np.array([252, 252, 253], dtype=np.uint8)

    axis_color = np.array([92, 102, 112], dtype=np.uint8)
    _draw_line(canvas, left, top, left, top + plot_h, axis_color)
    _draw_line(canvas, left, top + plot_h, left + plot_w, top + plot_h, axis_color)

    r_raw = np.asarray(values, dtype=np.float32)
    if r_raw.size == 0:
        return canvas
    finite = np.isfinite(r_raw)
    if not np.any(finite):
        r = np.zeros_like(r_raw)
    else:
        fallback = float(np.nanmean(r_raw[finite]))
        r = np.where(finite, r_raw, fallback)
    n = len(r)

    if n == 1:
        r_min, r_max = float(r[0] - 1.0), float(r[0] + 1.0)
    else:
        r_min, r_max = float(np.min(r)), float(np.max(r))
        if abs(r_max - r_min) < 1e-8:
            r_min -= 1.0
            r_max += 1.0

    tick_color = np.array([170, 180, 190], dtype=np.uint8)
    y_ticks = np.linspace(r_min, r_max, 5)
    x_ticks = [0, max(0, n // 2), max(0, n - 1)]
    for yv in y_ticks:
        y = top + int(round((1.0 - (yv - r_min) / (r_max - r_min)) * plot_h))
        _draw_line(canvas, left, y, left + plot_w, y, tick_color)
    for xv in x_ticks:
        x = left + int(round(xv * (plot_w / max(1, n - 1))))
        _draw_line(canvas, x, top, x, top + plot_h, tick_color)

    curve_color = np.array([34, 119, 217], dtype=np.uint8)
    xs = left + np.round(np.arange(n) * (plot_w / max(1, n - 1))).astype(np.int32)
    ys = top + np.round((1.0 - (r - r_min) / (r_max - r_min)) * plot_h).astype(np.int32)
    ys = np.clip(ys, top, top + plot_h)
    for i in range(1, n):
        _draw_line(canvas, xs[i - 1], ys[i - 1] + 1, xs[i], ys[i] + 1, curve_color)
        _draw_line(canvas, xs[i - 1], ys[i - 1], xs[i], ys[i], curve_color)

    cursor_step = int(np.clip(cursor_step, 0, n - 1))
    cx = left + int(round(cursor_step * (plot_w / max(1, n - 1))))
    cursor_color = np.array([234, 67, 53], dtype=np.uint8)
    _draw_line(canvas, cx, top, cx, top + plot_h, cursor_color)

    cy = top + int(round((1.0 - (r[cursor_step] - r_min) / (r_max - r_min)) * plot_h))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            yy, xx = cy + dy, cx + dx
            if 0 <= yy < height and 0 <= xx < width:
                canvas[yy, xx] = cursor_color

    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text_color = (65, 75, 85)

    for yv in y_ticks:
        y = top + int(round((1.0 - (yv - r_min) / (r_max - r_min)) * plot_h))
        draw.text((4, y - 5), _format_tick(yv), fill=text_color, font=font)

    for xv in x_ticks:
        x = left + int(round(xv * (plot_w / max(1, n - 1))))
        draw.text((x - 8, top + plot_h + 6), str(int(xv)), fill=text_color, font=font)

    draw.text((left, 2), title, fill=text_color, font=font)
    draw.text((left + plot_w - 74, 2), f"step {cursor_step}", fill=(90, 98, 108), font=font)
    draw.text((left + plot_w - 74, 12), f"v {r[cursor_step]:.3f}", fill=(90, 98, 108), font=font)

    return np.asarray(img, dtype=np.uint8)


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


def get_wandb_video_with_progress(
    renders,
    progress_traces,
    potential_diff_traces,
    frame_steps,
    n_cols=None,
    fps=15,
):
    """Return a W&B video with frames + two synchronized plots (progress, potential diff)."""
    if renders is None or len(renders) == 0:
        return None

    composite_renders = []
    for i, render in enumerate(renders):
        if len(render) == 0:
            continue
        progresses = progress_traces[i] if i < len(progress_traces) else np.zeros((1,), dtype=np.float32)
        pot_diffs = potential_diff_traces[i] if i < len(potential_diff_traces) else np.zeros((1,), dtype=np.float32)
        steps = frame_steps[i] if i < len(frame_steps) else np.arange(len(render), dtype=np.int32)
        episode_frames = []
        for j, frame in enumerate(render):
            step_idx = steps[j] if j < len(steps) else (len(progresses) - 1)
            top_h = frame.shape[0] // 2
            bottom_h = frame.shape[0] - top_h
            progress_plot = _make_metric_plot_frame(progresses, step_idx, top_h, frame.shape[1], "progress")
            pot_diff_plot = _make_metric_plot_frame(pot_diffs, step_idx, bottom_h, frame.shape[1], "potential_diff")
            panel = np.concatenate([progress_plot, pot_diff_plot], axis=0)
            episode_frames.append(np.concatenate([frame, panel], axis=1))
        composite_renders.append(np.asarray(episode_frames, dtype=np.uint8))

    if len(composite_renders) == 0:
        return None
    return get_wandb_video(composite_renders, n_cols=n_cols, fps=fps)
