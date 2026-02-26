"""Vertex-based reward code proposer for OGBench + QAM double-cube task2."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Tuple
import yaml

from qam.reward.utils import TokenUsage, get_global_cost_tracker


PROMPTS_DIR = Path(__file__).resolve().parent / "assets" / "prompts"
STEP_PROMPT_FILES = {
    1: "Step1_decomposition.yaml",
    2: "Step2_progress_metric.yaml",
    3: "Step3_active_assignment.yaml",
    4: "Step4_reward_terms.yaml",
    5: "Step5_success_anti_spoofing.yaml",
    6: "Step6_failure_audit.yaml",
    7: "Step7_invariants.yaml",
    8: "Step8_logging.yaml",
}


def _load_step_prompt(step: int) -> Tuple[str, str]:
    name = STEP_PROMPT_FILES.get(step)
    if name is None:
        raise ValueError(f"Unknown prompt step: {step}")
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must be a mapping with system/user keys: {path}")
    system = str(data.get("system", "")).strip()
    user = str(data.get("user", "")).strip()
    if not system or not user:
        raise ValueError(f"Prompt YAML missing non-empty system/user: {path}")
    return system, user


@dataclass(frozen=True)
class VertexConfig:
    model: str = "gemini-2.5-flash"
    project: str = "gemini-api-433301"
    location: str = "global"
    temperature: float = 0.0
    max_output_tokens: int = 8192
    infer_server_url: str = "http://127.0.0.1:8008"
    use_api: bool = False


class RewardProposingAgent:
    """Calls Vertex Gemini with the reward prompt and returns generated code."""

    def __init__(self, vertex: Optional[VertexConfig] = None):
        self.vertex = vertex or VertexConfig(
            project=os.getenv("GCP_PROJECT", "gemini-api-433301"),
            location=os.getenv("GCP_LOCATION", "global"),
            infer_server_url=os.getenv("INFER_SERVER_URL", "http://127.0.0.1:8008"),
            use_api=os.getenv("REWARD_PROPOSER_USE_API", "0").lower() in {"1", "true", "yes"},
        )
        self.system_prompt, _ = _load_step_prompt(1)

    def build_prompt(self, step: str = "all", extra_context: str = "") -> Tuple[str, str]:
        if step == "all":
            users = []
            systems = []
            for idx in sorted(STEP_PROMPT_FILES):
                system, user = _load_step_prompt(idx)
                systems.append(system)
                users.append(user)
            system_prompt = systems[0]
            if any(s != system_prompt for s in systems[1:]):
                # Keep deterministic behavior; first system prompt wins.
                system_prompt = systems[0]
            user_prompt = "\n\n".join(users)
        else:
            step_idx = int(step)
            system_prompt, user_prompt = _load_step_prompt(step_idx)
        if extra_context.strip():
            user_prompt = f"{user_prompt}\n\nAdditional context:\n{extra_context.strip()}"
        return system_prompt, user_prompt

    def generate(self, step: str = "all", extra_context: str = "", retries: int = 3) -> Dict[str, Any]:
        system_prompt, user_prompt = self.build_prompt(step=step, extra_context=extra_context)
        text = self._execute(system_prompt=system_prompt, user_prompt=user_prompt, retries=retries)
        return {
            "step": step,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": text,
            "reward_code": self.extract_python_code(text),
            "design_text": self.extract_design_text(text),
        }

    def _execute(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        if self.vertex.use_api:
            return self._execute_api(system_prompt=system_prompt, user_prompt=user_prompt, retries=retries)
        try:
            return self._execute_vertex(system_prompt=system_prompt, user_prompt=user_prompt, retries=retries)
        except RuntimeError as exc:
            if "google-genai is not installed" in str(exc):
                # Transparent fallback for environments that only have an inference server.
                return self._execute_api(system_prompt=system_prompt, user_prompt=user_prompt, retries=retries)
            raise

    def _execute_vertex(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            raise RuntimeError(
                "google-genai is not installed. Install it in qam/.venv first."
            ) from exc

        client = genai.Client(
            vertexai=True,
            project=self.vertex.project,
            location=self.vertex.location,
        )
        config = types.GenerateContentConfig(
            temperature=self.vertex.temperature,
            max_output_tokens=self.vertex.max_output_tokens,
            system_instruction=system_prompt,
        )

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.vertex.model,
                    contents=[user_prompt],
                    config=config,
                )
                usage_md = getattr(response, "usage_metadata", None)
                if usage_md is not None:
                    usage = TokenUsage(
                        input_tokens=int(getattr(usage_md, "prompt_token_count", 0) or 0),
                        output_tokens=int(getattr(usage_md, "candidates_token_count", 0) or 0),
                        reasoning_tokens=int(getattr(usage_md, "thoughts_token_count", 0) or 0),
                        total_tokens=int(getattr(usage_md, "total_token_count", 0) or 0),
                    )
                    tracker = get_global_cost_tracker(self.vertex.model)
                    tracker.add_usage(usage)
                    tracker.print_update(usage, prefix="RewardProposerVertex")
                return getattr(response, "text", "") or ""
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

    def _execute_api(self, system_prompt: str, user_prompt: str, retries: int = 3) -> str:
        try:
            import requests
        except Exception as exc:
            raise RuntimeError("requests is not installed; cannot use API fallback.") from exc

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                response = requests.post(
                    f"{self.vertex.infer_server_url}/v1/vlm_inference",
                    json={
                        "prompt": f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}",
                        "images": [],
                        "temperature": self.vertex.temperature,
                        "max_tokens": self.vertex.max_output_tokens,
                        "model_name": self.vertex.model,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
                text = data.get("text", "")
                if not isinstance(text, str):
                    raise RuntimeError("Inference server response missing 'text' field.")
                usage = TokenUsage(
                    input_tokens=int(data.get("prompt_tokens", 0) or 0),
                    output_tokens=int(data.get("completion_tokens", 0) or 0),
                    reasoning_tokens=0,
                    total_tokens=int(data.get("total_tokens", 0) or 0),
                )
                if usage.total_tokens > 0 or usage.input_tokens > 0 or usage.output_tokens > 0:
                    tracker = get_global_cost_tracker(self.vertex.model)
                    tracker.add_usage(usage)
                    tracker.print_update(usage, prefix="RewardProposerAPI")
                return text
            except Exception as exc:
                last_err = exc
                if attempt < retries:
                    time.sleep(2 * (2**attempt))
                    continue
                raise RuntimeError(f"Inference server call failed: {exc}") from exc
        raise RuntimeError(f"Inference server call failed: {last_err}")

    @staticmethod
    def extract_python_code(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def extract_design_text(text: str) -> str:
        if not text:
            return ""
        code_match = re.search(r"```python\s*.*?```", text, re.DOTALL | re.IGNORECASE)
        if code_match:
            start, end = code_match.span()
            return (text[:start] + text[end:]).strip()
        return text.strip()
