# Qwen3.8 inference latency: vLLM MTP-1 vs MTP-3

Reproduce the measured vLLM MTP-1/MTP-3 comparison. The sampled video frames
and manifest are included in `assets/`; the only large external artifact is the
local Qwen3.8-27B-FP8 base checkpoint. See [REPORT.md](REPORT.md) for the measured
results and interpretation.

## Quick start

```bash
git clone <repository-url> qwen38-inference-latency
cd qwen38-inference-latency

# Activate an environment with the tested vLLM/CUDA/PyTorch stack.
source /path/to/vllm-environment/bin/activate
pip install -r requirements.txt

export QWEN38_MODEL="/path/to/downloaded/Qwen3.8-27B-FP8"
export LATENCY_GPU=0  # optional; defaults to GPU 0
python benchmark_mtp_latency.py
```

The script uses `assets/` by default, checks that the selected GPU has at least
45,000 MiB free and under 5% utilization, and writes new output to `run-output/`
so it does not overwrite the committed reference results. Set
`LATENCY_SOURCE_ROOT` only if you want to use a different copy of the input
frames. Set `LATENCY_OUTPUT_DIR` to choose another output directory.
`LATENCY_GPU` is the zero-based GPU index visible to the current process; the
script maps that logical index to its physical device even when
`CUDA_VISIBLE_DEVICES` remaps GPU numbering. It defaults to visible GPU 0.

## Model and environment

Download the **unfinetuned Qwen3.8-27B-FP8 base checkpoint**, including its
processor files, from the model's official source and set `QWEN38_MODEL` to its
local directory. The benchmark does not use the fine-tuned LoRA adapter.

`requirements.txt` pins the Python-side dependencies used by the benchmark and
visual-token counter. It intentionally excludes `torch`, `vllm`, and CUDA-related
packages. vLLM 0.32.0 and its PyTorch/CUDA-compatible build are intentionally
installed separately: use vLLM's installation instructions for
the target CUDA/PyTorch environment instead of letting a generic pip command
replace the GPU stack. The measured environment was:

| Component | Version / device |
|---|---|
| Python | 3.12.4 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime (PyTorch build) | 13.0 |
| NVIDIA driver | 580.119.02 |
| vLLM | 0.32.0 |
| Transformers | 5.19.0 |
| GPU | NVIDIA RTX A6000, 48 GB |

A rerun records Python, PyTorch, CUDA runtime, NVIDIA driver, GPU, vLLM, and
Transformers versions in `benchmark_config.json` and `results.json`.

## What the benchmark records

The launcher explicitly configures vLLM speculative decoding as
`{"method":"mtp","num_speculative_tokens":1}` and then `3`. It verifies the
MTP depth in the server startup log before sending requests. Startup and warm-up
are excluded from request latency. It records client TTFT and total request
time, server TTFT/generation/queue/inter-token metrics when available, exact
completion token usage, prompts and outputs, and source-frame hashes.

- [`REPORT.md`](REPORT.md): measured results, visual-token counts, and interpretation.
- `assets/input_manifest.json` and `assets/sampled_images/`: exact sampled inputs.
- `benchmark_config.json`: environment, host, GPU, inputs, and timer scope for the recorded run.
- `vllm_mtp1.log`, `vllm_mtp3.log`: server logs and effective MTP configuration.
- `mtp1_results.json`, `mtp3_results.json`, `results.json`: raw and combined records.
- `visual_token_counts.json`: processor-derived visual-token counts.
- `count_visual_tokens.py`: CPU-only visual-token counter; it loads the processor,
  not model weights, and does not need a GPU. Run it after a benchmark with the
  same environment and output directory to regenerate the counts.

The benchmark measures high-level memory generation only; it does not run the
robot simulation or low-level policy. Each interval is sent independently with
the same fixed prompt and no previous output carried forward. This is a
controlled latency comparison, not a closed-loop task evaluation.

An earlier 2026-10-22 run was described as MTP-1, but its vLLM log shows
`speculative_config=None`. Those older timings are not verified MTP results.
The committed MTP-1/MTP-3 runs explicitly configure and verify speculative MTP
in the logs.
