#!/usr/bin/env python3
"""Compare base Qwen3.8-27B-FP8 vLLM latency with native MTP-1 and MTP-3.

Runs both configurations sequentially on one explicitly selected GPU. Model
startup and one warm-up request per mode are excluded from request timings.
All artifacts, including vLLM logs and raw request/response data, stay in this
workspace directory.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


OUT = Path(os.environ.get("LATENCY_OUTPUT_DIR", Path(__file__).resolve().parent / "run-output")).resolve()
MODEL = Path(os.environ["QWEN38_MODEL"]) if os.environ.get("QWEN38_MODEL") else None
SOURCE = Path(os.environ.get("LATENCY_SOURCE_ROOT", Path(__file__).resolve().parent / "assets")).resolve()
MANIFEST: dict[str, Any] = {}
SAMPLED: list[dict[str, Any]] = []
PORT = int(os.environ.get("LATENCY_PORT", "8018"))
GPU = os.environ.get("LATENCY_GPU", "0")
VLLM = os.environ.get("VLLM_BIN") or shutil.which("vllm")
MODEL_ALIAS = "qwen38-base"
MAX_TOKENS = 192
WINDOWS = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
WARMUP_WINDOW = (18.0, 20.0)


def resolve_visible_gpu() -> dict[str, str]:
    import torch

    try:
        logical_index = int(GPU)
    except ValueError as exc:
        raise RuntimeError(f"LATENCY_GPU must be a logical GPU index, got {GPU!r}") from exc
    if logical_index < 0 or logical_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"LATENCY_GPU={logical_index} is out of range; this process sees "
            f"{torch.cuda.device_count()} CUDA device(s)"
        )
    device_uuid = str(torch.cuda.get_device_properties(logical_index).uuid)
    if not device_uuid.startswith("GPU-"):
        device_uuid = "GPU-" + device_uuid
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], text=True).splitlines()
    for line in rows:
        values = [value.strip() for value in line.split(",", 5)]
        if len(values) == 6 and values[1] == device_uuid:
            return dict(zip(("physical_index", "uuid", "name", "driver", "memory_free_mib", "utilization"), values))
    raise RuntimeError(f"Could not map visible CUDA device {logical_index} ({device_uuid}) to nvidia-smi")


def assert_gpu_idle() -> dict[str, str]:
    device = resolve_visible_gpu()
    if int(device["memory_free_mib"]) < 45000 or int(device["utilization"]) >= 5:
        raise RuntimeError(
            f"Visible GPU {GPU} (physical index {device['physical_index']}) is not idle "
            "with >=45000 MiB free; refusing to start vLLM"
        )
    return device


def collect_environment() -> dict[str, Any]:
    import platform
    import torch
    import transformers
    import vllm

    selected = resolve_visible_gpu()
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": selected["driver"],
        "gpu": selected["name"],
        "gpu_index_visible_to_process": GPU,
        "gpu_index_physical": selected["physical_index"],
        "vllm": vllm.__version__,
        "transformers": transformers.__version__,
    }


def check_prerequisites() -> None:
    global MANIFEST, SAMPLED
    if MODEL is None:
        raise RuntimeError("Set QWEN38_MODEL to the downloaded local Qwen3.8-27B-FP8 checkpoint directory")
    if VLLM is None:
        raise RuntimeError("Set VLLM_BIN to the vLLM executable, or activate an environment where vllm is on PATH")
    if not MODEL.is_dir():
        raise FileNotFoundError(f"Base model not found: {MODEL}")
    if not Path(VLLM).is_file():
        raise FileNotFoundError(f"vLLM executable not found: {VLLM}")
    if not (SOURCE / "input_manifest.json").is_file():
        raise FileNotFoundError(f"Input manifest not found under {SOURCE}")
    MANIFEST = json.loads((SOURCE / "input_manifest.json").read_text())
    SAMPLED = MANIFEST["sampled_frames"]
    assert_gpu_idle()
    try:
        r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
    except requests.RequestException:
        return
    raise RuntimeError(f"Port {PORT} already has a responding service; refusing to replace it")


def build_prompt(start_s: float, end_s: float) -> str:
    return (
        f"Task instruction: {MANIFEST['task_instruction']}\n"
        f"Observation interval: {start_s:.2f}s through {end_s:.2f}s.\n"
        "Earlier memory: No events recorded yet.\n\n"
        "Maintain a chronological text event log. Write only information visibly "
        "supported by these images that is new or shows a change. Keep each event "
        "concise and in observed order. Describe actions as in progress unless the "
        "images show their outcome. Do not include timestamps; the system adds the "
        "interval time. If no new event is visible, return an empty string.\n"
        'Return exactly one JSON object and no Markdown: {"memory_update": string}'
    )


def make_body(start_s: float, end_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = [r for r in SAMPLED if start_s - 1e-6 <= float(r["time_s"]) < end_s - 1e-6]
    if not selected:
        raise RuntimeError(f"No sampled frames for interval {start_s}-{end_s}")
    paths = [SOURCE / "sampled_images" / r["file"] for r in selected]
    raw_frames = [p.read_bytes() for p in paths]
    encoded = ",".join(base64.b64encode(b).decode("ascii") for b in raw_frames)
    frame_ids = [int(r["frame_id"]) for r in selected]
    prompt = build_prompt(start_s, end_s)
    body = {
        "model": MODEL_ALIAS,
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/jpeg;base64," + encoded}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
        "media_io_kwargs": {"video": {
            "fps": float(MANIFEST["source_fps"]),
            "frames_indices": frame_ids,
            "total_num_frames": int(MANIFEST["clip_frames"]),
            "duration": float(MANIFEST["demo_interval_s"][1] - MANIFEST["demo_interval_s"][0]),
            "do_sample_frames": False,
        }},
    }
    metadata = {
        "interval_s": [start_s, end_s],
        "frame_count": len(selected),
        "frame_ids": frame_ids,
        "frame_timestamps_s": [round(float(r["time_s"]), 4) for r in selected],
        "frame_sha256": [hashlib.sha256(b).hexdigest() for b in raw_frames],
        "prompt": prompt,
    }
    return body, metadata


def stream_request(body: dict[str, Any], metadata: dict[str, Any], kind: str) -> dict[str, Any]:
    body = dict(body)
    body["model"] = MODEL_ALIAS
    started = time.perf_counter()
    first_text_at = None
    chunks: list[str] = []
    usage = metrics = None
    with requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json=body,
                       stream=True, timeout=(30, 900)) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:]
            if raw == "[DONE]":
                break
            event = json.loads(raw)
            if event.get("usage") is not None:
                usage = event["usage"]
            if event.get("metrics") is not None:
                metrics = event["metrics"]
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    chunks.append(piece)
    ended = time.perf_counter()
    output = "".join(chunks)
    total_s = ended - started
    first_s = first_text_at - started if first_text_at is not None else None
    completion_tokens = (usage or {}).get("completion_tokens")
    return {
        **metadata,
        "request_kind": kind,
        "client_ttft_nonempty_text_s": first_s,
        "client_total_s": total_s,
        "client_after_first_text_s": ended - first_text_at if first_text_at else None,
        "completion_tokens": completion_tokens,
        "client_tokens_per_second": completion_tokens / total_s if completion_tokens and total_s else None,
        "usage": usage,
        "server_metrics": metrics,
        "output": output,
    }


def wait_healthy(proc: subprocess.Popen[str], log_path: Path, timeout_s: int = 900) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with status {proc.returncode}; see {log_path}")
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become healthy within {timeout_s}s; see {log_path}")


def stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)


def run_mode(depth: int) -> dict[str, Any]:
    assert_gpu_idle()
    log_path = OUT / f"vllm_mtp{depth}.log"
    log_file = log_path.open("w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = resolve_visible_gpu()["physical_index"]
    env["VLLM_CACHE_ROOT"] = str(OUT / "cache" / f"mtp{depth}")
    env["TRITON_CACHE_DIR"] = str(OUT / "cache" / f"triton_mtp{depth}")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(OUT / "cache" / f"inductor_mtp{depth}")
    env["HF_HOME"] = str(OUT / "cache" / f"hf_mtp{depth}")
    env["HF_HUB_CACHE"] = str(OUT / "cache" / f"hf_hub_mtp{depth}")
    env["TRANSFORMERS_CACHE"] = str(OUT / "cache" / f"transformers_mtp{depth}")
    env["XDG_CACHE_HOME"] = str(OUT / "cache" / f"xdg_mtp{depth}")
    env["VLLM_ENGINE_READY_TIMEOUT_S"] = "900"
    cmd = [
        VLLM, "serve", str(MODEL),
        "--host", "127.0.0.1", "--port", str(PORT),
        "--served-model-name", MODEL_ALIAS,
        "--trust-remote-code",
        "--trust-request-mm-kwargs",
        "--enable-per-request-metrics",
        "--per-request-spec-decode-metrics", "summary",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": depth}),
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True,
                            env=env, start_new_session=True)
    try:
        wait_healthy(proc, log_path)
        log_text = log_path.read_text(errors="replace")
        if ("speculative_config=SpeculativeConfig(method='mtp'" not in log_text
                or f"num_spec_tokens={depth}" not in log_text):
            raise RuntimeError(f"Server log does not confirm MTP-{depth} configuration; see {log_path}")

        bodies = [(make_body(start, end)) for start, end in WINDOWS]
        warm_body, warm_meta = make_body(*WARMUP_WINDOW)
        warm = stream_request(warm_body, warm_meta, kind="warmup")
        rows = [stream_request(body, meta, kind="measured") for body, meta in bodies]
        log_text = log_path.read_text(errors="replace")
        return {
            "mtp_depth": depth,
            "model": str(MODEL),
            "checkpoint_type": "unfinetuned base FP8",
            "vllm_version": subprocess.check_output([sys.executable, "-c", "import vllm; print(vllm.__version__)"], text=True).strip(),
            "visible_gpu": GPU,
            "server_args": cmd,
            "warmup_excluded": warm,
            "samples": rows,
            "server_log": str(log_path),
            "mtp_config_log_excerpt": [line[-2500:] for line in log_text.splitlines() if "speculative_config=SpeculativeConfig" in line],
        }
    finally:
        stop_server(proc)
        log_file.close()


def main() -> None:
    check_prerequisites()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    run_meta = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": subprocess.check_output(["hostname"], text=True).strip(),
        "gpu_index_visible_to_process": GPU,
        "environment": collect_environment(),
        "setup": "single RTX A6000, Qwen3.8-27B-FP8, vLLM 0.32.0, enforce_eager, max_model_len=4096",
        "input_protocol": "same three 2-second timestamped JPEG-video windows and same prompt for MTP-1 and MTP-3; separate 18-20s video warm-up window per server is excluded",
        "timer_definition": "client timings start immediately before POST and include HTTP/service overhead; server metrics separately expose TTFT, generation time, queue and ITL",
        "outputs": [],
    }
    (OUT / "benchmark_config.json").write_text(json.dumps({
        **run_meta,
        "model_path": str(MODEL),
        "source_manifest": str(SOURCE / "input_manifest.json"),
        "windows": WINDOWS,
        "warmup_window": WARMUP_WINDOW,
        "max_tokens": MAX_TOKENS,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "thinking": False},
    }, ensure_ascii=False, indent=2) + "\n")
    for depth in (1, 3):
        print(f"Starting MTP-{depth} on visible GPU {GPU}", flush=True)
        run = run_mode(depth)
        run_meta["outputs"].append(run)
        target = OUT / f"mtp{depth}_results.json"
        target.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n")
        concise = [{
            "interval_s": r["interval_s"], "completion_tokens": r["completion_tokens"],
            "client_ttft_s": round(r["client_ttft_nonempty_text_s"], 4) if r["client_ttft_nonempty_text_s"] is not None else None,
            "client_total_s": round(r["client_total_s"], 4),
            "server_metrics": r["server_metrics"],
        } for r in run["samples"]]
        print(json.dumps({"mtp_depth": depth, "samples": concise}, ensure_ascii=False), flush=True)
    run_meta["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "results.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
