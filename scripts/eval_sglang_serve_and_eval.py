#!/usr/bin/env python
import argparse
import shlex
import subprocess
import time
from typing import List, Optional

import requests

from scripts.eval_sglang_server import build_parser, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve sglang and run evaluation")
    parser.add_argument("--serve_cmd", default="python -m sglang.launch_server")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout_s", type=int, default=300)
    parser.add_argument("--serve_args", nargs="*", default=None)
    args, remaining = parser.parse_known_args()

    eval_parser = build_parser()
    eval_namespace = eval_parser.parse_args(remaining)
    return argparse.Namespace(serve=args, eval=eval_namespace)


def wait_for_server(base_url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            response = requests.get(f"{base_url}/models", timeout=5)
            if response.status_code == 200:
                return
        except requests.RequestException:
            time.sleep(2)
    raise TimeoutError("sglang server did not start in time")


def build_serve_cmd(args: argparse.Namespace) -> List[str]:
    cmd = shlex.split(args.serve_cmd)
    cmd += [
        "--model-path",
        args.model_name_or_path,
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.serve_args:
        cmd += args.serve_args
    return cmd


def main() -> None:
    args = parse_args()
    base_url = f"http://{args.serve.host}:{args.serve.port}/v1"
    serve_cmd = build_serve_cmd(args.serve)

    process = subprocess.Popen(serve_cmd)
    try:
        wait_for_server(base_url, args.serve.timeout_s)
        args.eval.base_url = base_url
        run_eval(args.eval)
    finally:
        process.terminate()
        process.wait(timeout=30)


if __name__ == "__main__":
    main()
