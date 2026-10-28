#!/usr/bin/env python3
"""Count Qwen visual placeholder tokens for the exact benchmark frame groups.

This runs the checkpoint's installed Transformers processor on CPU; it does not
load model weights or use a GPU. Counts are derived from image_grid_thw and
video_grid_thw, using the same merge-size formula that expands image/video
placeholders in Qwen3VLProcessor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

OUT = Path(os.environ.get("LATENCY_OUTPUT_DIR", Path(__file__).resolve().parent / "run-output")).resolve()
for name, subdir in (
    ("HF_HOME", "hf_visual_count"),
    ("HF_HUB_CACHE", "hf_hub_visual_count"),
    ("TRANSFORMERS_CACHE", "transformers_visual_count"),
    ("XDG_CACHE_HOME", "xdg_visual_count"),
):
    os.environ.setdefault(name, str(OUT / "cache" / subdir))

from PIL import Image
from transformers import AutoProcessor, __version__ as TRANSFORMERS_VERSION


MODEL = Path(os.environ["QWEN38_MODEL"]) if os.environ.get("QWEN38_MODEL") else None
SOURCE = Path(os.environ.get("LATENCY_SOURCE_ROOT", Path(__file__).resolve().parent / "assets")).resolve()
RESULTS = OUT


def visual_count(grid: Any, merge_size: int) -> int:
    return int((grid.prod(dim=-1) // (merge_size**2)).sum().item())


def main() -> None:
    if MODEL is None:
        raise RuntimeError("Set QWEN38_MODEL to the downloaded local Qwen3.8-27B-FP8 checkpoint directory")
    manifest_path = SOURCE / "input_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Input manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    video_processor = processor.video_processor
    rows = manifest["sampled_frames"]
    result_rows = []

    for start, end in ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0)):
        selected = [r for r in rows if start - 1e-6 <= float(r["time_s"]) < end - 1e-6]
        paths = [SOURCE / "sampled_images" / r["file"] for r in selected]
        frames = [Image.open(path).convert("RGB") for path in paths]
        processed = video_processor(
            videos=[frames],
            fps=float(manifest["source_fps"]),
            do_sample_frames=False,
            cap_pixels_per_frame=False,
            return_tensors="pt",
        )
        grid = processed["video_grid_thw"]
        visual_tokens = visual_count(grid, int(video_processor.merge_size))
        mtp1 = json.loads((RESULTS / "mtp1_results.json").read_text())
        sample = next(r for r in mtp1["samples"] if r["interval_s"] == [start, end])
        result_rows.append({
            "input": "timestamped_video_frames",
            "interval_s": [start, end],
            "frame_count": len(selected),
            "frame_timestamps_s": [float(r["time_s"]) for r in selected],
            "original_frame_sizes_px": [list(img.size) for img in frames],
            "video_grid_thw": grid.tolist(),
            "merge_size": int(video_processor.merge_size),
            "visual_token_count": visual_tokens,
            "vllm_total_prompt_tokens": sample["usage"]["prompt_tokens"],
            "nonvisual_prompt_token_remainder": sample["usage"]["prompt_tokens"] - visual_tokens,
            "output_tokens_mtp1": sample["completion_tokens"],
            "output_tokens_mtp3": next(
                r["completion_tokens"] for r in json.loads((RESULTS / "mtp3_results.json").read_text())["samples"]
                if r["interval_s"] == [start, end]
            ),
        })

    output = {
        "model": str(MODEL),
        "processor": type(processor).__name__,
        "transformers_version": TRANSFORMERS_VERSION,
        "processor_source": "local checkpoint preprocessor_config.json and video_preprocessor_config.json",
        "count_definition": "sum(prod(grid_thw) / merge_size^2); exact Qwen image/video placeholder count after processor resize/patchify, not a direct model hidden-state hook",
        "video_windows": result_rows,
    }
    (OUT / "visual_token_counts.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
