# StatsD Metrics Implementation

## Summary

Added a generic StatsD metrics logger that emits all vLLM metrics (matching Prometheus), including vision encoding times.

## Changes Made

### 1. Generic StatsD Logger (`vllm/v1/metrics/statsd.py`)
- Created `StatsDStatLogger` class alongside other metric loggers
- Emits all standard metrics to StatsD:
  - Gauges: running/waiting requests, KV cache usage
  - Counters: preemptions, tokens, cache hits/queries
  - Timings: TTFT, inter-token latency, e2e latency, queue/prefill/decode times, vision encoding
- Generic `record_metric()` function for any metric collection

### 2. Vision Encoding Histogram Support
- Added `vision_encoding_times_iter` to `IterationStats`
- Wired up Prometheus histogram to observe vision encoding times
- Thread-local metric buffering for collection from model forward passes
- Automatic collection in engine before metrics recording

### 3. Integration
- Auto-registers when `VLLM_STATSD_HOST` is set
- Works alongside Prometheus (both emit metrics)
- Zero overhead when not configured

## Usage

```bash
export VLLM_STATSD_HOST=localhost
export VLLM_STATSD_PORT=8125  # optional, defaults to 8125
```

## Metrics Emitted

All metrics prefixed with `vllm.`:

**Gauges:**
- `num_requests_running`
- `num_requests_waiting`
- `kv_cache_usage_perc`

**Counters:**
- `prompt_tokens`
- `generation_tokens`
- `num_preemptions`
- `prefix_cache_queries/hits`
- `external_prefix_cache_queries/hits`
- `mm_cache_queries/hits`
- `request_success.<finish_reason>`

**Timings (in ms):**
- `time_to_first_token_seconds`
- `inter_token_latency_seconds`
- `vision_encoding_seconds`
- `e2e_request_latency_seconds`
- `request_queue_time_seconds`
- `request_inference_time_seconds`
- `request_prefill_time_seconds`
- `request_decode_time_seconds`

## Files Modified

1. `vllm/v1/metrics/statsd.py` - New generic StatsD logger
2. `vllm/v1/metrics/loggers.py` - Auto-register StatsD logger
3. `vllm/v1/metrics/stats.py` - Added vision_encoding_times_iter field
4. `vllm/v1/engine/llm_engine.py` - Collect metrics before recording
5. `vllm/model_executor/models/{llava,qwen2_vl,qwen3_vl}.py` - Use generic record_metric()
6. `vllm/model_executor/vision_statsd.py` - Removed (replaced by generic statsd.py)

