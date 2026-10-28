# Qwen3.8 inference latency: MTP-1 vs MTP-3

## Result

On one RTX A6000 on `l2.mit.edu`, the unfinetuned `Qwen3.8-27B-FP8` served
through vLLM 0.32 produced roughly 30-token memory updates in 1.35–1.97 seconds.
The 40-token MTP-1 request took 1.975 seconds, so the reported “about 40 tokens
in 1–2 seconds” is consistent with this configuration. MTP-3 reduced the mean
request time across the same three inputs by about 19%, with similar output
lengths.

| Mode | Output tokens (three requests) | Mean client request | Mean client TTFT | Mean server generation | Mean inter-token latency | Mean draft acceptance |
|---|---:|---:|---:|---:|---:|---:|
| MTP-1 | 29 / 27 / 40 | 1.807 s | 0.756 s | 1.051 s | 34.0 ms | 95.2% |
| MTP-3 | 29 / 27 / 44 | 1.457 s | 0.771 s | 0.686 s | 21.7 ms | 81.2% |

Decode-only throughput, calculated as `(output_tokens - 1) / vLLM generation_time`
for each request, was **29.5 token/s for MTP-1** and **47.2 token/s for MTP-3**
(aggregate over the three requests). This excludes the first token; end-to-end
throughput including TTFT was 17.7 and 22.9 token/s respectively.

The visual input counts were:

| Interval | Sampled frames | Visual tokens | Total prompt tokens (vLLM) |
|---|---:|---:|---:|
| 0–2 s | 5 | 900 | 1,078 |
| 2–4 s | 4 | 600 | 770 |
| 4–6 s | 4 | 600 | 770 |

Visual-token counts come from the local Qwen3VL processor `video_grid_thw` after resizing and patchification. Total prompt tokens also include text, timestamps, and multimodal markers. The processor grids and calculation are recorded in [`visual_token_counts.json`](visual_token_counts.json) and can be regenerated with [`count_visual_tokens.py`](count_visual_tokens.py).

Client request time starts immediately before the HTTP request and ends after the streaming
response completes. It includes video processing, multimodal prefill, decoding,
and HTTP overhead, but excludes model startup. Queue time was about 0.02 ms, so
the request was not waiting behind other vLLM work. TTFT stayed around 0.7–0.9
seconds; most of the remaining time was text generation. MTP-3 proposed more
tokens per step and accepted a smaller fraction of them, but still reduced
mean inter-token latency by about 36%.

The decoded instructions were materially the same across modes. The first two
outputs matched exactly; the third MTP-3 output added a short location phrase.
This is a small latency sample (three measured requests per mode), not a
statistical performance claim or a full task evaluation.

## Inference setup and review points

The new MTP-1/MTP-3 benchmark uses the **vLLM backend**. MTP is enabled with vLLM speculative decoding (`method=mtp`, one or three speculative tokens), confirmed in startup logs and request metrics. 

- Base checkpoint only; no LoRA adapter loaded.
- One RTX A6000, vLLM 0.32.0, FP8 weights, `max_model_len=4096`.
- Same prompts, timestamped video frame groups, temperature 0, thinking off,
  and token limit for both modes. Requests are run sequentially.
- One different 18–20s video warm-up per server is excluded from timing.
- The server log explicitly confirms `method='mtp'` and `num_spec_tokens=1` or
  `3`; request responses include speculative acceptance metrics.
- The run uses `--enforce-eager`, which disables CUDA graphs and torch.compile.
  This matched the previous conservative/debug serving setup, but may leave
  throughput optimizations unused. These results do not measure best-case
  tuned vLLM performance; that should be a separate controlled comparison.

## Correction to the earlier MTP-1 report

The 2026-10-22 file `qwen38_memory_2s_timing_20261022/stage_timing_mtp.json`
was described as MTP-1 and reported 31/33/56 tokens in about 1.8–2.5 seconds.
However, its associated `stage_server.log` initializes vLLM with
`speculative_config=None`. Those measurements are **not verified MTP-1** and
must not be cited as such. The current run explicitly configures and verifies
MTP-1 and MTP-3 in the server logs.

## Code and evidence

- Benchmark implementation: [`benchmark_mtp_latency.py`](benchmark_mtp_latency.py)
- Visual-token counter: [`count_visual_tokens.py`](count_visual_tokens.py)
- Exact launch settings and inputs: [`benchmark_config.json`](benchmark_config.json)
- Processor-derived visual token counts: [`visual_token_counts.json`](visual_token_counts.json)
- MTP-1 requests and outputs: [`mtp1_results.json`](mtp1_results.json)
- MTP-3 requests and outputs: [`mtp3_results.json`](mtp3_results.json)
- Combined machine-readable record: [`results.json`](results.json)
- Full vLLM logs: [`vllm_mtp1.log`](vllm_mtp1.log), [`vllm_mtp3.log`](vllm_mtp3.log)

To rerun after cloning, activate the compatible vLLM environment, set
`QWEN38_MODEL` to the local base checkpoint, and run `python benchmark_mtp_latency.py` from the repository root. The bundled `assets/` are
used by default; see [`README.md`](README.md) for the complete setup.
