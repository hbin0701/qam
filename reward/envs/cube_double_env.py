"""
Environment schema for cube-double tasks.

This module intentionally models environment entities only (no reward logic).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence

# Success criterion (task-level): object is successful if within threshold of goal.
CUBE_SUCCESS_THRESHOLD = 0.04  # meters
NUM_CUBES = 2
TOOL_OBJECT_NAMES = ("cube_0", "cube_1", "gripper", "goal_0", "goal_1")
ENV_OBJECT_NAMES = TOOL_OBJECT_NAMES

@dataclass
class Vector3D:
    x: float
    y: float
    z: float

    @classmethod
    def from_any(cls, v: "Vector3D | Sequence[float]") -> "Vector3D":
        if isinstance(v, cls):
            return cls(v.x, v.y, v.z)
        vals = list(v)
        if len(vals) != 3:
            raise ValueError(f"Expected 3 values, got {len(vals)}")
        return cls(float(vals[0]), float(vals[1]), float(vals[2]))

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Goal:
    name: str
    position: Vector3D

    def __post_init__(self) -> None:
        self.position = Vector3D.from_any(self.position)


@dataclass
class Finger:
    name: str
    position: Vector3D

    def __post_init__(self) -> None:
        self.position = Vector3D.from_any(self.position)


@dataclass
class Gripper:
    position: Vector3D
    left: Finger
    right: Finger

    def __post_init__(self) -> None:
        self.position = Vector3D.from_any(self.position)


@dataclass
class Cube:
    name: str
    position: Vector3D

    def __post_init__(self) -> None:
        self.position = Vector3D.from_any(self.position)


EXPECTED_OBJECT_TYPES = {
    "cube_0": Cube,
    "cube_1": Cube,
    "gripper": Gripper,
    "goal_0": Goal,
    "goal_1": Goal,
}


class Env:
    """Base environment entity container."""

    def __init__(
        self,
        objects: Dict[str, object] | None = None,
        success_threshold: float = CUBE_SUCCESS_THRESHOLD,
    ):
        self.success_threshold = float(success_threshold)
        self._objects: Dict[str, object] = dict(objects or {})
        self._validate_required_objects()

    def _validate_required_objects(self) -> None:
        missing = [name for name in ENV_OBJECT_NAMES if name not in self._objects]
        if missing:
            raise ValueError(
                f"Env is missing required objects: {missing}. "
                f"Required object names are: {ENV_OBJECT_NAMES}"
            )
        for name in ENV_OBJECT_NAMES:
            obj = self._objects[name]
            expected_t = EXPECTED_OBJECT_TYPES.get(name)
            if expected_t is not None and not isinstance(obj, expected_t):
                raise ValueError(
                    f"Object {name!r} must be of type {expected_t.__name__}, "
                    f"got {type(obj).__name__}"
                )
            if not hasattr(obj, "position"):
                raise ValueError(f"Object {name!r} must have a 'position' field")
            # Ensure position is parseable as Vector3D.
            Vector3D.from_any(getattr(obj, "position"))

    def get_object(self, name: str) -> object:
        if name not in self._objects:
            raise KeyError(f"Unknown object {name!r}. Available: {sorted(self._objects)}")
        return self._objects[name]

    def get_position(self, object_name: str) -> Vector3D:
        obj = self.get_object(object_name)
        if not hasattr(obj, "position"):
            raise AttributeError(f"Object {object_name!r} has no position field")
        return Vector3D.from_any(getattr(obj, "position"))

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(self._objects.keys())


class Demo:
    """Demo/video trace container indexed by frame number."""

    def __init__(
        self,
        frames: Mapping[int, Mapping[str, Sequence[float] | Vector3D]] | None = None,
    ):
        self._frames: Dict[int, Dict[str, Vector3D]] = {}
        for frame_num, obj_map in (frames or {}).items():
            per_frame: Dict[str, Vector3D] = {}
            for object_name, pos in obj_map.items():
                per_frame[str(object_name)] = Vector3D.from_any(pos)
            self._frames[int(frame_num)] = per_frame

    def get_position(self, frame_num: int, object_name: str) -> Vector3D:
        frame = self._frames.get(int(frame_num))
        if frame is None:
            raise KeyError(f"Unknown frame_num {frame_num}")
        if object_name not in frame:
            raise KeyError(f"Unknown object {object_name!r} in frame {frame_num}")
        return frame[object_name]

    @property
    def frame_numbers(self) -> tuple[int, ...]:
        return tuple(sorted(self._frames.keys()))


def get_position(
    src: str,
    object_name: str,
    *,
    env: Env | None = None,
    demo: Demo | None = None,
    frame_num: int | None = None,
) -> Vector3D:
    """Unified position accessor for env/demo sources.

    - src='env': uses Env.get_position(object_name).
    - src='demo': uses Demo.get_position(frame_num, object_name). frame_num required.
    """
    mode = str(src).lower()
    if mode == "env":
        if env is None:
            raise ValueError("env source requested but env is None")
        return env.get_position(object_name)
    if mode == "demo":
        if demo is None:
            raise ValueError("demo source requested but demo is None")
        if frame_num is None:
            raise ValueError("frame_num is required when src='demo'")
        return demo.get_position(frame_num=frame_num, object_name=object_name)
    raise ValueError(f"Unknown src {src!r}; expected 'env' or 'demo'")


def get_dist_xy(a: Sequence[float], b: Sequence[float]) -> float:
    """2D Euclidean distance between vectors a and b."""
    av = list(a)
    bv = list(b)
    if len(av) != 2 or len(bv) != 2:
        raise ValueError("get_dist_xy expects a and b as length-2 vectors")
    ax, ay = float(av[0]), float(av[1])
    bx, by = float(bv[0]), float(bv[1])
    return float(math.hypot(ax - bx, ay - by))


def get_dist_xyz(a: Sequence[float], b: Sequence[float]) -> float:
    """3D Euclidean distance between vectors a and b."""
    av = list(a)
    bv = list(b)
    if len(av) != 3 or len(bv) != 3:
        raise ValueError("get_dist_xyz expects a and b as length-3 vectors")
    ax, ay, az = float(av[0]), float(av[1]), float(av[2])
    bx, by, bz = float(bv[0]), float(bv[1]), float(bv[2])
    dx, dy, dz = ax - bx, ay - by, az - bz
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))
