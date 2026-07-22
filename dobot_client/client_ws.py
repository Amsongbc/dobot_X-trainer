#!/usr/bin/env python3
"""WebSocket client for Motus inference server (dobot/server_vlm_mask.py /ws)."""

import argparse
import base64
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

try:
    from websocket import create_connection
except ImportError as exc:
    raise SystemExit(
        "websocket-client is required. Install with: pip install websocket-client"
    ) from exc


def _parse_csv_to_float_list(csv_str: str) -> List[float]:
    values = [x.strip() for x in csv_str.split(",") if x.strip()]
    return [float(x) for x in values]


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _create_random_image_base64(width: int = 384, height: int = 320) -> str:
    random_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(random_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _ws_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/ws"


def _send_and_recv(ws_url: str, payload: Dict[str, Any], timeout: int = 300) -> Dict[str, Any]:
    ws = create_connection(ws_url, timeout=timeout)
    try:
        ws.send(json.dumps(payload))
        raw = ws.recv()
        return json.loads(raw)
    finally:
        ws.close()


def check_only(host: str, port: int, timeout: int = 10) -> bool:
    ws_url = _ws_url(host, port)
    print(f"Checking WebSocket endpoint: {ws_url}")
    response = _send_and_recv(ws_url, {"type": "health"}, timeout=timeout)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    ok = response.get("type") == "health" and response.get("status") == "healthy"
    print("check-only:", "PASS" if ok else "FAIL")
    return ok


def run_once(
    host: str,
    port: int,
    instruction: Optional[str],
    image_path: Optional[str],
    images_b64: Optional[List[str]],
    state: Optional[List[float]],
    msg_type: str = "inference",
    timeout: int = 300,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": msg_type}
    if msg_type == "inference":
        if instruction is not None:
            payload["instruction"] = instruction
        if images_b64 is not None:
            payload["images"] = images_b64
        elif image_path is not None:
            payload["image"] = _encode_image_to_base64(image_path)
        if state is not None:
            payload["proprio_data"] = [state]
    return _send_and_recv(_ws_url(host, port), payload, timeout=timeout)


def run_loop(
    host: str,
    port: int,
    hz: float,
    instruction: Optional[str],
    image_path: Optional[str],
    images_b64: Optional[List[str]],
    state: Optional[List[float]],
    count: Optional[int],
    timeout: int = 300,
) -> None:
    interval = 1.0 / hz
    idx = 0
    while count is None or idx < count:
        start = time.time()
        try:
            result = run_once(
                host=host,
                port=port,
                instruction=instruction,
                image_path=image_path,
                images_b64=images_b64,
                state=state,
                timeout=timeout,
            )
            if result.get("type") == "error":
                print(f"[{idx}] ERROR: {result.get('detail')}")
            else:
                actions = result.get("predicted_actions", [])
                print(
                    f"[{idx}] ok action_shape={result.get('action_shape')} "
                    f"server_ms={result.get('processing_time_ms', 0):.1f} "
                    f"first_action={actions[0] if actions else None}"
                )
        except Exception as exc:
            print(f"[{idx}] request failed: {exc}")
        idx += 1
        elapsed = time.time() - start
        sleep_s = max(0.0, interval - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Motus WebSocket inference client")
    parser.add_argument("host", help="Server host, e.g. 127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--check-only", action="store_true", help="Only verify WebSocket health")
    parser.add_argument("--mock", action="store_true", help="Run mock inference once")
    parser.add_argument("--hz", type=float, default=None, help="Inference loop frequency")
    parser.add_argument("--count", type=int, default=1, help="Loop count when --hz is set")
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--image", type=str, default=None, help="Single image or pre-concatenated T-shape image")
    parser.add_argument("--top_image", type=str, default=None)
    parser.add_argument("--left_wrist_image", type=str, default=None)
    parser.add_argument("--right_wrist_image", type=str, default=None)
    parser.add_argument("--state_csv", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    state = _parse_csv_to_float_list(args.state_csv) if args.state_csv else None
    images_b64 = None
    if args.top_image or args.left_wrist_image or args.right_wrist_image:
        image_paths = [p for p in [args.top_image, args.left_wrist_image, args.right_wrist_image] if p]
        images_b64 = [_encode_image_to_base64(p) for p in image_paths]

    if args.check_only:
        return 0 if check_only(args.host, args.port, timeout=args.timeout) else 1

    if args.mock:
        result = run_once(args.host, args.port, None, None, None, None, msg_type="mock", timeout=args.timeout)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result.get("type") == "mock" else 1

    if args.hz is not None:
        run_loop(
            host=args.host,
            port=args.port,
            hz=args.hz,
            instruction=args.instruction,
            image_path=args.image,
            images_b64=images_b64,
            state=state,
            count=args.count,
            timeout=args.timeout,
        )
        return 0

    if images_b64 is None and args.image is None:
        images_b64 = [
            _create_random_image_base64(),
            _create_random_image_base64(),
            _create_random_image_base64(),
        ]

    result = run_once(
        host=args.host,
        port=args.port,
        instruction=args.instruction,
        image_path=args.image,
        images_b64=images_b64,
        state=state,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("type") == "inference" else 1


if __name__ == "__main__":
    raise SystemExit(main())
