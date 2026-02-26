#!/usr/bin/env python3
"""Step1 decomposition agent: sends video + prompt to Vertex and returns stage decomposition."""

from __future__ import annotations

import argparse
from io import BytesIO
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image
import yaml

from qam.reward.envs.utils import FrameToolRuntime
from qam.reward.utils import TokenUsage, get_global_cost_tracker
from qam.reward.utils.logfmt import color, join_kv, level_prefix


PROMPT_PATH = Path(__file__).resolve().parent.parent / "assets" / "prompts" / "Step1_decomposition.yaml"
MAX_THINKING_BUDGET = 8192


def _load_step1_prompt() -> Tuple[str, str]:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must be mapping with system/user keys: {PROMPT_PATH}")
    system = str(data.get("system", "")).strip()
    user = str(data.get("user", "")).strip()
    if not system or not user:
        raise ValueError(f"Prompt YAML missing non-empty system/user: {PROMPT_PATH}")
    return system, user


def _guess_video_mime(video_path: str) -> str:
    guessed, _ = mimetypes.guess_type(video_path)
    if guessed and guessed.startswith("video/"):
        return guessed
    return "video/mp4"


@dataclass(frozen=True)
class DecomposeConfig:
    model: str = "gemini-2.5-flash"
    project: str = "gemini-api-433301"
    location: str = "global"
    temperature: float = 0.0
    max_output_tokens: int = 16384
    thinking_budget: Optional[int] = None
    thinking_level: str = ""
    max_frames: int = 32
    enable_tools: bool = True
    max_tool_rounds: int = 8


class DecompositionAgent:
    """Calls Vertex with Step1 prompt and a video input."""

    def __init__(self, cfg: Optional[DecomposeConfig] = None):
        self.cfg = cfg or DecomposeConfig(
            project=os.getenv("GCP_PROJECT", "gemini-api-433301"),
            location=os.getenv("GCP_LOCATION", "global"),
        )
        self.system_prompt, self.user_prompt = _load_step1_prompt()

    def build_user_prompt(self, env_code: str, extra_context: str = "") -> str:
        env_block = (
            "Environment code:\n"
            "```python\n"
            f"{env_code}\n"
            "```"
        )
        marker = "Output format (STRICT):"
        if marker in self.user_prompt:
            head, tail = self.user_prompt.split(marker, 1)
            prompt = f"{head.rstrip()}\n\n{env_block}\n\n{marker}{tail}"
        else:
            prompt = f"{self.user_prompt}\n\n{env_block}"
        if extra_context.strip():
            prompt += f"\n\nAdditional context:\n{extra_context.strip()}"
        return prompt

    def run(
        self,
        video_path: str,
        env_code: str,
        pose_json_path: str = "",
        extra_context: str = "",
        retries: int = 3,
    ) -> Dict[str, Any]:
        user_prompt = self.build_user_prompt(env_code=env_code, extra_context=extra_context)
        raw, sampled_frame_indices, tool_trace, response_debug = self._call_vertex(
            video_path=video_path,
            user_prompt=user_prompt,
            pose_json_path=pose_json_path,
            retries=retries,
        )
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
            "video_path": video_path,
            "sampled_frame_indices": sampled_frame_indices,
            "tool_trace": tool_trace,
            "raw_response": raw,
            "response_debug": response_debug,
            "parsed_json": self._extract_json(raw),
            "python_code": self._extract_python(raw),
        }

    def _call_vertex(
        self,
        video_path: str,
        user_prompt: str,
        pose_json_path: str = "",
        retries: int = 3,
    ) -> Tuple[str, List[int], List[Dict[str, Any]], Dict[str, Any]]:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError("google-genai is not installed. Install it in qam/.venv.") from exc

        client = genai.Client(
            vertexai=True,
            project=self.cfg.project,
            location=self.cfg.location,
        )
        config_kwargs: Dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "max_output_tokens": self.cfg.max_output_tokens,
            "system_instruction": self.system_prompt,
        }
        if self.cfg.thinking_budget is not None or self.cfg.thinking_level:
            thinking_budget = self.cfg.thinking_budget
            if thinking_budget is not None and thinking_budget > MAX_THINKING_BUDGET:
                print(
                    f"{level_prefix('DecomposeStep1', level='warn')} "
                    f"thinking_budget={thinking_budget} exceeds max={MAX_THINKING_BUDGET}; capping."
                )
                thinking_budget = MAX_THINKING_BUDGET
            thinking_level = None
            if self.cfg.thinking_level:
                if self._supports_thinking_level(self.cfg.model):
                    thinking_level = getattr(types.ThinkingLevel, str(self.cfg.thinking_level).upper(), None)
                else:
                    print(
                        f"{level_prefix('DecomposeStep1', level='warn')} model {self.cfg.model!r} does not support thinking_level; "
                        "using thinking_budget only."
                    )
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
            )
        config = types.GenerateContentConfig(**config_kwargs)

        parts = [types.Part.from_text(text=user_prompt + f"\n\nVideo frame sample count: {self.cfg.max_frames}.")]
        sampled_frame_indices: List[int] = []
        if video_path.startswith("gs://"):
            video_mime = _guess_video_mime(video_path)
            parts.append(types.Part.from_uri(file_uri=video_path, mime_type=video_mime))
        else:
            path = Path(video_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Video not found: {path}")
            sampled = self._sample_video_frames_uniform(path, max_frames=self.cfg.max_frames)
            sampled_frame_indices = [int(i) for i, _ in sampled]
            for idx, (src_idx, fb) in enumerate(sampled):
                parts.append(
                    types.Part.from_text(
                        text=f"\nSampledFrame {idx + 1}/{len(sampled)} (source_frame_index={int(src_idx)})"
                    )
                )
                parts.append(types.Part.from_bytes(data=fb, mime_type="image/jpeg"))

        tool_runtime = None
        if self.cfg.enable_tools and pose_json_path:
            tool_runtime = FrameToolRuntime(pose_json_path=pose_json_path)
            object_names_text = ", ".join(tool_runtime.object_names) if tool_runtime.object_names else "(none detected)"
            env_src_text = "available" if tool_runtime.has_env_source else "not available in provided JSON"
            parts.append(
                types.Part.from_text(
                    text=(
                        "\nAvailable tools:\n"
                        "- get_position(src:str, object_name:str, frame_num:int optional, time_ref:str='curr')\n"
                        "  returns: {'ok': bool, 'position': [x, y, z]} or {'ok': false, 'error': str}\n"
                        "- get_dist_xy(a:array_like_2, b:array_like_2)\n"
                        "  a and b must be plain 2D numeric vectors: [x, y]\n"
                        "- get_dist_xyz(a:array_like_3, b:array_like_3)\n"
                        "  a and b must be plain 3D numeric vectors: [x, y, z]\n"
                        "Important:\n"
                        "- Do NOT pass the full get_position response object into distance tools.\n"
                        "- First read p = get_position(...), then pass p['position'] (or slices of it).\n"
                        "Examples:\n"
                        "- p1 = get_position(...); p2 = get_position(...)\n"
                        "- get_dist_xyz(a=p1['position'], b=p2['position'])\n"
                        "- get_dist_xy(a=p1['position'][:2], b=p2['position'][:2])\n"
                        "get_position source modes:\n"
                        "- src='demo': frame_num is required; use source_frame_index labels to inspect demo behavior.\n"
                        "- src='env': use env-state trace (pose/qpos based) for reward-time calls; frame_num optional.\n"
                        "- time_ref supports 'prev' or 'curr' when src='env' and frame_num is omitted.\n"
                        f"env source status: {env_src_text}\n"
                        f"Available object_name values from pose JSON: {object_names_text}"
                    )
                )
            )

        contents = [types.Content(role="user", parts=parts)]
        tools = None
        if tool_runtime is not None:
            tools = [
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="get_position",
                            description="Return object position (xyz) from demo/env source as {'ok': bool, 'position': [x,y,z]}",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "src": {"type": "string", "enum": ["demo", "env"]},
                                    "frame_num": {"type": "integer"},
                                    "time_ref": {"type": "string", "enum": ["prev", "curr"]},
                                    "object_name": {"type": "string"},
                                },
                                "required": ["object_name"],
                            },
                        ),
                        types.FunctionDeclaration(
                            name="get_dist_xy",
                            description="Compute 2D Euclidean distance. a and b must be numeric vectors [x, y], not objects.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "a": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                    "b": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                                },
                                "required": ["a", "b"],
                            },
                        ),
                        types.FunctionDeclaration(
                            name="get_dist_xyz",
                            description="Compute 3D Euclidean distance. a and b must be numeric vectors [x, y, z], not objects.",
                            parameters={
                                "type": "object",
                                "properties": {
                                    "a": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                                    "b": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                                },
                                "required": ["a", "b"],
                            },
                        ),
                    ]
                )
            ]

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response, tool_trace = self._generate_with_tools(
                    client=client,
                    model=self.cfg.model,
                    contents=contents,
                    config=config,
                    tools=tools,
                    tool_runtime=tool_runtime,
                )
                usage_md = getattr(response, "usage_metadata", None)
                if usage_md is not None:
                    usage = TokenUsage(
                        input_tokens=int(getattr(usage_md, "prompt_token_count", 0) or 0),
                        output_tokens=int(getattr(usage_md, "candidates_token_count", 0) or 0),
                        reasoning_tokens=int(getattr(usage_md, "thoughts_token_count", 0) or 0),
                        total_tokens=int(getattr(usage_md, "total_token_count", 0) or 0),
                    )
                    tracker = get_global_cost_tracker(self.cfg.model)
                    tracker.add_usage(usage)
                    tracker.print_update(usage, prefix="DecomposeStep1")
                raw_text = self._extract_response_text(response).strip()
                response_debug = self._summarize_response(response)
                finish_reason_text = " ".join(str(x) for x in response_debug.get("finish_reasons", []))
                if "MALFORMED_FUNCTION_CALL" in finish_reason_text:
                    snapshot = self._safe_response_snapshot(response)
                    response_debug["malformed_snapshot"] = snapshot
                    print(
                        f"{level_prefix('DecomposeStep1', level='warn')} malformed_function_call_snapshot "
                        f"{color(json.dumps(snapshot, ensure_ascii=False), 'dim')}"
                    )
                if not raw_text:
                    pairs = [
                        ("finish_reasons", response_debug.get("finish_reasons", [])),
                        ("prompt_block_reason", response_debug.get("prompt_feedback", {}).get("block_reason")),
                    ]
                    print(
                        f"{level_prefix('DecomposeStep1', level='warn')} empty_text_response "
                        f"{join_kv(pairs, tone='muted')}"
                    )
                return (raw_text, sampled_frame_indices, tool_trace, response_debug)
            except Exception as exc:
                last_err = exc
                text = str(exc)
                is_rate_limit = (
                    "429" in text or "ResourceExhausted" in text or "quota" in text.lower()
                )
                if is_rate_limit and attempt < retries:
                    time.sleep(2 * (2**attempt))
                    continue
                raise
        raise RuntimeError(f"Vertex call failed: {last_err}")

    @staticmethod
    def _supports_thinking_level(model_name: str) -> bool:
        name = str(model_name).lower()
        return "gemini-3" in name

    def _generate_with_tools(self, client, model, contents, config, tools, tool_runtime):
        from google.genai import types
        cfg = config
        tool_trace: List[Dict[str, Any]] = []
        if tools is not None:
            cfg = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                system_instruction=config.system_instruction,
                tools=tools,
                thinking_config=config.thinking_config,
            )
        response = client.models.generate_content(model=model, contents=contents, config=cfg)
        self._log_response_debug(response, tag="initial")
        if tools is None or tool_runtime is None:
            return response, tool_trace

        for _ in range(self.cfg.max_tool_rounds):
            calls = self._extract_function_calls(response)
            if not calls:
                return response, tool_trace
            contents.append(response.candidates[0].content)
            tool_parts = []
            for call in calls:
                name = call.get("name")
                args = call.get("args", {})
                result = tool_runtime.dispatch(name, args)
                tool_event = {
                    "name": name,
                    "args": args,
                    "result": result,
                }
                tool_trace.append(tool_event)
                print(
                    f"{level_prefix('DecomposeStep1Tool', level='info')} "
                    f"{color(json.dumps(tool_event, ensure_ascii=False), 'dim')}"
                )
                tool_parts.append(
                    types.Part.from_function_response(
                        name=name,
                        response={"result": result},
                    )
                )
            contents.append(types.Content(role="tool", parts=tool_parts))
            response = client.models.generate_content(model=model, contents=contents, config=cfg)
            self._log_response_debug(response, tag="post_tool")
        return response, tool_trace

    def _log_response_debug(self, response: Any, tag: str) -> None:
        direct_calls = getattr(response, "function_calls", None) or []
        if direct_calls:
            safe_calls = []
            for c in direct_calls:
                safe_calls.append(
                    {
                        "name": str(getattr(c, "name", "")),
                        "args": dict(getattr(c, "args", {}) or {}),
                    }
                )
            print(
                f"{level_prefix('DecomposeStep1Resp', level='info')} "
                f"tag={tag} direct_function_calls={color(json.dumps(safe_calls, ensure_ascii=False), 'dim')}"
            )
        cands = getattr(response, "candidates", None) or []
        for ci, cand in enumerate(cands):
            finish = self._coerce_debug_value(getattr(cand, "finish_reason", None))
            parts = getattr(getattr(cand, "content", None), "parts", None) or []
            items: List[Dict[str, Any]] = []
            for pi, p in enumerate(parts):
                item: Dict[str, Any] = {"part_index": pi}
                txt = getattr(p, "text", None)
                if isinstance(txt, str) and txt:
                    item["text_len"] = len(txt)
                    item["text_preview"] = txt[:160]
                fc = getattr(p, "function_call", None)
                if fc is not None:
                    item["function_call"] = {
                        "name": str(getattr(fc, "name", "")),
                        "args": dict(getattr(fc, "args", {}) or {}),
                    }
                items.append(item)
            if items or finish is not None:
                print(
                    f"{level_prefix('DecomposeStep1Resp', level='info')} "
                    f"tag={tag} candidate={ci} finish_reason={finish} "
                    f"parts={color(json.dumps(items, ensure_ascii=False), 'dim')}"
                )

    @staticmethod
    def _extract_function_calls(response) -> List[Dict[str, Any]]:
        calls = []
        direct_calls = getattr(response, "function_calls", None)
        if direct_calls:
            for c in direct_calls:
                calls.append({"name": getattr(c, "name", ""), "args": dict(getattr(c, "args", {}) or {})})
            return calls
        cands = getattr(response, "candidates", None) or []
        if not cands:
            return calls
        parts = getattr(getattr(cands[0], "content", None), "parts", None) or []
        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc is not None:
                calls.append({"name": getattr(fc, "name", ""), "args": dict(getattr(fc, "args", {}) or {})})
        return calls

    @staticmethod
    def _sample_video_frames_uniform(path: Path, max_frames: int = 20) -> List[Tuple[int, bytes]]:
        reader = imageio.get_reader(str(path))
        try:
            num_frames = reader.count_frames()
        except Exception:
            num_frames = 0
        if num_frames <= 0:
            # Fallback: stream through to count.
            num_frames = sum(1 for _ in reader)
            reader.close()
            reader = imageio.get_reader(str(path))

        if num_frames <= max_frames:
            indices = list(range(num_frames))
        else:
            indices = np.linspace(0, num_frames - 1, max_frames, dtype=int).tolist()

        frames: List[Tuple[int, bytes]] = []
        for i in indices:
            frame = reader.get_data(i)
            img = Image.fromarray(frame)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=90)
            frames.append((int(i), buf.getvalue()))
        reader.close()
        return frames

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                parsed = json.loads(match.group(1).strip())
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return None
        return None

    @staticmethod
    def _extract_python(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _coerce_debug_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        try:
            as_dict = dict(value)
            return {str(k): DecompositionAgent._coerce_debug_value(v) for k, v in as_dict.items()}
        except Exception:
            pass
        if isinstance(value, list):
            return [DecompositionAgent._coerce_debug_value(v) for v in value]
        text = str(value)
        if "." in text:
            text = text.split(".")[-1]
        return text

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        direct = getattr(response, "text", None)
        if isinstance(direct, str) and direct:
            return direct
        chunks: List[str] = []
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            parts = getattr(getattr(cand, "content", None), "parts", None) or []
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    chunks.append(t)
        return "\n".join(chunks)

    @staticmethod
    def _summarize_response(response: Any) -> Dict[str, Any]:
        usage_md = getattr(response, "usage_metadata", None)
        prompt_feedback = getattr(response, "prompt_feedback", None)
        summary: Dict[str, Any] = {
            "response_text_len": len((getattr(response, "text", "") or "")),
            "finish_reasons": [],
            "usage_metadata": {
                "prompt_token_count": int(getattr(usage_md, "prompt_token_count", 0) or 0),
                "candidates_token_count": int(getattr(usage_md, "candidates_token_count", 0) or 0),
                "thoughts_token_count": int(getattr(usage_md, "thoughts_token_count", 0) or 0),
                "total_token_count": int(getattr(usage_md, "total_token_count", 0) or 0),
            },
            "prompt_feedback": {
                "block_reason": DecompositionAgent._coerce_debug_value(getattr(prompt_feedback, "block_reason", None)),
                "safety_ratings": DecompositionAgent._coerce_debug_value(
                    getattr(prompt_feedback, "safety_ratings", None)
                ),
            },
            "candidates": [],
        }
        candidates = getattr(response, "candidates", None) or []
        for idx, cand in enumerate(candidates):
            finish_reason = DecompositionAgent._coerce_debug_value(getattr(cand, "finish_reason", None))
            summary["finish_reasons"].append(finish_reason)
            parts = getattr(getattr(cand, "content", None), "parts", None) or []
            text_lengths: List[int] = []
            function_calls = 0
            for p in parts:
                t = getattr(p, "text", None)
                if isinstance(t, str) and t:
                    text_lengths.append(len(t))
                if getattr(p, "function_call", None) is not None:
                    function_calls += 1
            summary["candidates"].append(
                {
                    "index": idx,
                    "finish_reason": finish_reason,
                    "safety_ratings": DecompositionAgent._coerce_debug_value(getattr(cand, "safety_ratings", None)),
                    "part_count": len(parts),
                    "function_call_count": function_calls,
                    "text_part_lengths": text_lengths,
                }
            )
        return summary

    @staticmethod
    def _safe_response_snapshot(response: Any) -> Dict[str, Any]:
        snap: Dict[str, Any] = {
            "response_text": (getattr(response, "text", "") or "")[:2000],
            "function_calls_count": len(getattr(response, "function_calls", None) or []),
            "candidate_count": len(getattr(response, "candidates", None) or []),
        }
        data = None
        for method_name in ("to_json_dict", "model_dump", "dict"):
            method = getattr(response, method_name, None)
            if callable(method):
                try:
                    data = method()
                    break
                except Exception:
                    continue
        if data is not None:
            try:
                raw = json.dumps(data, ensure_ascii=False)
                snap["response_obj_preview"] = raw[:4000]
            except Exception:
                snap["response_obj_preview"] = str(data)[:4000]
        return snap


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step1 decomposition with video input.")
    parser.add_argument("--video", type=str, required=True, help="Local path or gs:// URI to video.")
    parser.add_argument(
        "--env-code-path",
        type=str,
        required=True,
        help="Path to environment/source code to condition subtask code generation.",
    )
    parser.add_argument(
        "--pose-json-path",
        type=str,
        default="/rlwrld3/home/hyeonbin/RL/qam/artifacts/double_demo.json",
        help="Optional JSON file for tool access to object positions by frame (enables tool-calling).",
    )
    parser.add_argument("--output", type=str, default="", help="Optional JSON output file.")
    parser.add_argument("--write-raw-response", type=str, default="", help="Optional raw response text output file.")
    parser.add_argument("--write-prompt", type=str, default="", help="Optional rendered prompt output file.")
    parser.add_argument("--extra-context", type=str, default="", help="Optional additional context.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for rate limits.")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Vertex model name.")
    parser.add_argument("--project", type=str, default="", help="GCP project ID (overrides env).")
    parser.add_argument("--location", type=str, default="", help="GCP location (overrides env).")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum number of uniformly sampled frames sent to the model.",
    )
    parser.add_argument(
        "--enable-tools",
        type=str,
        default="true",
        help="Enable tool-calling (true/false).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=16384,
        help="Max output tokens for Vertex generation.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help=f"Model thinking token budget (max {MAX_THINKING_BUDGET}). 0 disables thinking; -1 automatic.",
    )
    parser.add_argument(
        "--thinking-level",
        type=str,
        default="",
        help="Thinking level: MINIMAL, LOW, MEDIUM, or HIGH.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = DecomposeConfig(
        model=args.model,
        project=args.project or os.getenv("GCP_PROJECT", "gemini-api-433301"),
        location=args.location or os.getenv("GCP_LOCATION", "global"),
        max_frames=args.max_frames,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
        thinking_level=args.thinking_level,
        enable_tools=str(args.enable_tools).lower() in {"1", "true", "yes"},
    )
    agent = DecompositionAgent(cfg=cfg)
    env_code_path = Path(args.env_code_path).expanduser().resolve()
    if not env_code_path.exists():
        raise FileNotFoundError(f"--env-code-path not found: {env_code_path}")
    env_code = env_code_path.read_text()
    result = agent.run(
        video_path=args.video,
        env_code=env_code,
        pose_json_path=args.pose_json_path,
        extra_context=args.extra_context,
        retries=args.retries,
    )

    if args.write_raw_response:
        raw_path = Path(args.write_raw_response)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(result.get("raw_response", ""))

    if args.write_prompt:
        prompt_path = Path(args.write_prompt)
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(f"[SYSTEM]\n{result['system_prompt']}\n\n[USER]\n{result['user_prompt']}")

    out_text = json.dumps(result, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(out_text)
        print(f"Wrote decomposition result to {out}")
        return

    print(out_text)


if __name__ == "__main__":
    main()
