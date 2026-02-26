"""Utilities for env/demo geometry tool calls."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

import numpy as np
from qam.reward.utils.logfmt import color, level_prefix


class FrameToolRuntime:
    """Tool runtime with two sources:
    - src='demo': pose trace indexed by frame_num
    - src='env': env-state trace (qpos/pose) indexed by step
    """

    def __init__(self, pose_json_path: str):
        self.pose_json_path = Path(pose_json_path).expanduser().resolve()
        if not self.pose_json_path.exists():
            raise FileNotFoundError(f"pose_json_path not found: {self.pose_json_path}")

        payload = json.loads(self.pose_json_path.read_text())
        frames: List[Dict[str, Any]] = []
        if isinstance(payload, dict):
            if isinstance(payload.get("frames"), list):
                frames = payload.get("frames", [])
            elif isinstance(payload.get("object_pose_trace"), list):
                frames = payload.get("object_pose_trace", [])
        elif isinstance(payload, list):
            frames = payload

        self.demo_by_step: Dict[int, Dict[str, Any]] = {}
        for idx, fr in enumerate(frames):
            step = fr.get("step", idx)
            pose = fr.get("pose", fr)
            try:
                self.demo_by_step[int(step)] = pose
            except Exception:
                continue

        self.env_by_step: Dict[int, Dict[str, Any]] = self._build_env_trace(payload)
        self.goal_positions: Dict[str, List[float]] = self._extract_goal_positions(payload)
        self.object_names: List[str] = self._collect_object_names()
        self.has_env_source: bool = len(self.env_by_step) > 0

    def _collect_object_names(self) -> List[str]:
        names = set()
        for pose in self.demo_by_step.values():
            if not isinstance(pose, dict):
                continue
            objects = pose.get("objects", {})
            if isinstance(objects, dict):
                names.update(str(k) for k in objects.keys())
        if any(isinstance(p, dict) and p.get("gripper_pos") is not None for p in self.demo_by_step.values()):
            names.add("gripper")
        if any("obs" in fr or "qpos" in fr for fr in self.env_by_step.values()):
            names.update({"cube_0", "cube_1", "gripper"})
        names.update(self.goal_positions.keys())
        if not names:
            names.add("gripper")
        return sorted(names)

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print(
                f"{level_prefix('FrameToolRuntime', level='info')} "
                f"call {name} {color(json.dumps(args, ensure_ascii=False), 'dim')}"
            )
            if name == "get_position":
                src = str(args.get("src", "demo")).lower()
                frame_like = args.get("frame_num", args.get("step_num", None))
                frame_num = int(frame_like) if frame_like is not None else None
                time_ref = str(args.get("time_ref", "curr")).lower()
                result = self.get_position(
                    src=src,
                    frame_num=frame_num,
                    time_ref=time_ref,
                    object_name=str(args.get("object_name")),
                )
                level = "success" if bool(result.get("ok")) else "warn"
                print(
                    f"{level_prefix('FrameToolRuntime', level=level)} "
                    f"return {name} {color(json.dumps(result, ensure_ascii=False), 'dim')}"
                )
                return result
            if name == "get_dist_xy":
                result = self.get_dist_xy(args.get("a"), args.get("b"))
                level = "success" if bool(result.get("ok")) else "warn"
                print(
                    f"{level_prefix('FrameToolRuntime', level=level)} "
                    f"return {name} {color(json.dumps(result, ensure_ascii=False), 'dim')}"
                )
                return result
            if name == "get_dist_xyz":
                result = self.get_dist_xyz(args.get("a"), args.get("b"))
                level = "success" if bool(result.get("ok")) else "warn"
                print(
                    f"{level_prefix('FrameToolRuntime', level=level)} "
                    f"return {name} {color(json.dumps(result, ensure_ascii=False), 'dim')}"
                )
                return result
            result = {"ok": False, "error": f"unknown tool: {name}"}
            print(
                f"{level_prefix('FrameToolRuntime', level='warn')} "
                f"return {name} {color(json.dumps(result, ensure_ascii=False), 'dim')}"
            )
            return result
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
            print(
                f"{level_prefix('FrameToolRuntime', level='error')} "
                f"return {name} {color(json.dumps(result, ensure_ascii=False), 'dim')}"
            )
            return result

    def get_position(self, src: str, frame_num: Optional[int], time_ref: str, object_name: str) -> Dict[str, Any]:
        if object_name in self.goal_positions:
            return {"ok": True, "position": self.goal_positions[object_name]}
        if src == "env":
            step = self._resolve_env_step(frame_num=frame_num, time_ref=time_ref)
            if step is None:
                return {"ok": False, "error": "env source unavailable"}
            frame = self.env_by_step.get(int(step))
            if frame is None:
                return {"ok": False, "error": f"env step {step} not found"}
            return self._get_position_from_env_frame(frame, object_name=object_name)

        if frame_num is None:
            frame_num = 0
        pose = self.demo_by_step.get(int(frame_num))
        if pose is None:
            return {"ok": False, "error": f"demo frame {frame_num} not found"}
        objects = pose.get("objects", {}) if isinstance(pose, dict) else {}
        obj = objects.get(object_name)
        if obj is None and object_name == "gripper":
            gp = pose.get("gripper_pos")
            if gp is not None:
                return {"ok": True, "position": gp}
        if obj is None:
            return {"ok": False, "error": f"object {object_name} not found at frame {frame_num}"}
        return {"ok": True, "position": obj.get("pos")}

    def _resolve_env_step(self, frame_num: Optional[int], time_ref: str) -> Optional[int]:
        if frame_num is not None:
            return int(frame_num)
        if not self.env_by_step:
            return None
        keys = sorted(self.env_by_step.keys())
        curr = int(keys[-1])
        if str(time_ref).lower() == "prev":
            return int(keys[-2]) if len(keys) >= 2 else curr
        return curr

    def _get_position_from_env_frame(self, frame: Dict[str, Any], object_name: str) -> Dict[str, Any]:
        if "pose" in frame and isinstance(frame.get("pose"), dict):
            pose = frame["pose"]
            objects = pose.get("objects", {})
            if isinstance(objects, dict) and object_name in objects:
                obj = objects[object_name]
                return {"ok": True, "position": obj.get("pos")}
            if object_name == "gripper" and pose.get("gripper_pos") is not None:
                return {"ok": True, "position": pose.get("gripper_pos")}
        if "qpos" in frame:
            return self._position_from_qpos(frame["qpos"], object_name)
        return {"ok": False, "error": "env frame has no usable pose/qpos fields"}

    @staticmethod
    def _cube_idx(name: str) -> Optional[int]:
        m = re.match(r"^cube_(\d+)$", str(name))
        return int(m.group(1)) if m else None

    def _position_from_qpos(self, qpos, object_name: str) -> Dict[str, Any]:
        qpos_arr = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if object_name == "gripper":
            return {"ok": False, "error": "gripper position is not directly available from qpos"}
        idx = self._cube_idx(object_name)
        if idx is None:
            return {"ok": False, "error": f"unsupported object_name for qpos source: {object_name}"}
        start = 14 + idx * 7
        if qpos_arr.shape[0] < start + 7:
            return {"ok": False, "error": f"qpos too short for {object_name} ({qpos_arr.shape[0]})"}
        pos = qpos_arr[start : start + 3].tolist()
        return {"ok": True, "position": pos}

    @staticmethod
    def _extract_goal_positions(payload: Any) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        if not isinstance(payload, dict):
            return out
        raw = payload.get("goal_positions", payload.get("goals", None))
        if raw is None:
            return out
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            return out
        for i in range(arr.shape[0]):
            out[f"goal_{i}"] = arr[i].tolist()
        return out

    @staticmethod
    def _build_env_trace(payload: Any) -> Dict[int, Dict[str, Any]]:
        env_by_step: Dict[int, Dict[str, Any]] = {}
        if not isinstance(payload, dict):
            return env_by_step

        if isinstance(payload.get("env_trace"), list):
            for idx, fr in enumerate(payload["env_trace"]):
                if not isinstance(fr, dict):
                    continue
                step = int(fr.get("step", idx))
                env_by_step[step] = fr
            return env_by_step

        obs_seq = payload.get("observations")
        qpos_seq = payload.get("qpos")
        qvel_seq = payload.get("qvel")
        next_obs_seq = payload.get("next_observations")
        next_qpos_seq = payload.get("next_qpos")
        n = 0
        for seq in (obs_seq, qpos_seq, qvel_seq, next_obs_seq, next_qpos_seq):
            if isinstance(seq, list):
                n = max(n, len(seq))
        if n <= 0:
            return env_by_step
        for i in range(n):
            fr: Dict[str, Any] = {"step": i}
            if isinstance(obs_seq, list) and i < len(obs_seq):
                fr["obs"] = obs_seq[i]
            if isinstance(qpos_seq, list) and i < len(qpos_seq):
                fr["qpos"] = qpos_seq[i]
            if isinstance(qvel_seq, list) and i < len(qvel_seq):
                fr["qvel"] = qvel_seq[i]
            if isinstance(next_obs_seq, list) and i < len(next_obs_seq):
                fr["next_obs"] = next_obs_seq[i]
            if isinstance(next_qpos_seq, list) and i < len(next_qpos_seq):
                fr["next_qpos"] = next_qpos_seq[i]
            env_by_step[i] = fr
        return env_by_step

    @staticmethod
    def get_dist_xy(a, b) -> Dict[str, Any]:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if a.shape[0] != 2 or b.shape[0] != 2:
            return {"ok": False, "error": "a and b must be length-2 vectors"}
        return {"ok": True, "dist_xy": float(np.linalg.norm(a - b))}

    @staticmethod
    def get_dist_xyz(a, b) -> Dict[str, Any]:
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        if a.shape[0] != 3 or b.shape[0] != 3:
            return {"ok": False, "error": "a and b must be length-3 vectors"}
        return {"ok": True, "dist_xyz": float(np.linalg.norm(a - b))}
