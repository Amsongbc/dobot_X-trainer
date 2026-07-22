#!/usr/bin/env python3
"""Launch an OpenPI 0.5 Dobot policy server.

This script is intended to be run on the GPU/server machine, not on the
robot-control client. It wraps OpenPI's official WebSocket policy server and
keeps deployment-specific paths out of the source file, so it can be published
in a public repository.

Example:
    python dobot_client/serve_openpi05_server.py \
        --openpi-dir /path/to/openpi \
        --checkpoint-dir /path/to/pi05_dobot_checkpoint/29999_pytorch \
        --port 8000
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the OpenPI 0.5 Dobot WebSocket policy server.",
    )
    parser.add_argument(
        "--openpi-dir",
        type=Path,
        default=os.environ.get("OPENPI_DIR"),
        required=os.environ.get("OPENPI_DIR") is None,
        help="Path to the OpenPI repository. Can also be set with OPENPI_DIR.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=os.environ.get("OPENPI_CHECKPOINT_DIR"),
        required=os.environ.get("OPENPI_CHECKPOINT_DIR") is None,
        help=(
            "Path to the OpenPI pi0.5 Dobot checkpoint directory. The directory "
            "should contain model.safetensors and assets/."
        ),
    )
    parser.add_argument(
        "--config",
        default="pi05_dobot_full",
        help="OpenPI training config name for the Dobot pi0.5 policy.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="WebSocket server port.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for launching OpenPI.",
    )
    parser.add_argument(
        "--default-prompt",
        default=None,
        help="Optional default prompt used when the client does not send one.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record policy requests/responses with OpenPI's PolicyRecorder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the command without starting the server.",
    )
    return parser.parse_args()


def validate_paths(openpi_dir: Path, checkpoint_dir: Path) -> tuple[Path, Path]:
    openpi_dir = openpi_dir.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()

    serve_policy = openpi_dir / "scripts" / "serve_policy.py"
    if not serve_policy.is_file():
        raise FileNotFoundError(f"OpenPI serve_policy.py not found: {serve_policy}")

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    weight_path = checkpoint_dir / "model.safetensors"
    assets_dir = checkpoint_dir / "assets"
    if not weight_path.is_file():
        raise FileNotFoundError(f"model.safetensors not found: {weight_path}")
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"normalization assets directory not found: {assets_dir}")

    return openpi_dir, checkpoint_dir


def build_command(args: argparse.Namespace, openpi_dir: Path, checkpoint_dir: Path) -> list[str]:
    command = [
        args.python,
        str(openpi_dir / "scripts" / "serve_policy.py"),
        "--port",
        str(args.port),
    ]
    if args.default_prompt:
        command.extend(["--default-prompt", args.default_prompt])
    if args.record:
        command.append("--record")
    command.extend(
        [
            "policy:checkpoint",
            "--policy.config",
            args.config,
            "--policy.dir",
            str(checkpoint_dir),
        ]
    )
    return command


def main() -> None:
    args = parse_args()
    openpi_dir, checkpoint_dir = validate_paths(args.openpi_dir, args.checkpoint_dir)
    command = build_command(args, openpi_dir, checkpoint_dir)

    env = os.environ.copy()
    openpi_src = str(openpi_dir / "src")
    env["PYTHONPATH"] = (
        openpi_src
        if not env.get("PYTHONPATH")
        else openpi_src + os.pathsep + env["PYTHONPATH"]
    )

    print("Launching OpenPI 0.5 Dobot server:")
    print(" ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return

    subprocess.run(command, cwd=openpi_dir, env=env, check=True)


if __name__ == "__main__":
    main()
