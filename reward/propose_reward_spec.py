#!/usr/bin/env python3
"""CLI for generating reward design and reward code via Vertex."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from qam.reward.double_cube_task2_proposer import RewardProposingAgent, VertexConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reward design + code for double-cube-play-task2."
    )
    parser.add_argument(
        "--format",
        choices=["json", "code", "design", "prompt"],
        default="json",
        help="Output format.",
    )
    parser.add_argument(
        "--extra-context",
        type=str,
        default="",
        help="Optional extra context appended to the base prompt.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "5", "6", "7", "8"],
        help="Prompt step to run. 'all' concatenates Step1..Step8 user prompts.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry count for Vertex rate-limit errors.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Model name for Vertex or inference server.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="",
        help="GCP project for Vertex. Defaults to env GCP_PROJECT.",
    )
    parser.add_argument(
        "--location",
        type=str,
        default="",
        help="GCP location for Vertex. Defaults to env GCP_LOCATION or global.",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use inference server API fallback instead of Vertex SDK.",
    )
    parser.add_argument(
        "--infer-server-url",
        type=str,
        default="http://127.0.0.1:8008",
        help="Inference server URL when --use-api is enabled.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional output path. Prints to stdout if omitted.",
    )
    parser.add_argument(
        "--write-raw-response",
        type=str,
        default="",
        help="Optional path to save full raw model response text.",
    )
    parser.add_argument(
        "--write-prompt",
        type=str,
        default="",
        help="Optional path to save the exact final prompt text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = VertexConfig(
        model=args.model,
        project=args.project or "gemini-api-433301",
        location=args.location or "global",
        infer_server_url=args.infer_server_url,
        use_api=args.use_api,
    )
    agent = RewardProposingAgent(vertex=cfg)
    if args.format == "prompt":
        system_prompt, user_prompt = agent.build_prompt(step=args.step, extra_context=args.extra_context)
        prompt_text = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"
        text = prompt_text
        if args.write_prompt:
            prompt_path = Path(args.write_prompt)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(prompt_text)
    else:
        result = agent.generate(step=args.step, extra_context=args.extra_context, retries=args.retries)
        if args.write_raw_response:
            raw_path = Path(args.write_raw_response)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(result.get("raw_response", ""))
        if args.write_prompt:
            prompt_path = Path(args.write_prompt)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            system_prompt = result.get("system_prompt", "")
            user_prompt = result.get("user_prompt", "")
            prompt_path.write_text(f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}")
        if args.format == "json":
            text = json.dumps(result, indent=2)
        elif args.format == "code":
            text = result.get("reward_code", "")
        else:
            text = result.get("design_text", "")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"Wrote reward proposal to {path}")
        return

    print(text)


if __name__ == "__main__":
    main()
