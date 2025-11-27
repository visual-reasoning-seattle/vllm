# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""StatsD metrics logger for vLLM."""

import os
import socket
from typing import Optional

from vllm.config import VllmConfig
from vllm.v1.metrics.stats import (
    IterationStats,
    MultiModalCacheStats,
    SchedulerStats,
    record_vision_encoding_time,
)

_statsd_client: Optional["StatsDClient"] = None


class StatsDClient:
    """StatsD UDP client."""

    def __init__(self, host: str, port: int, prefix: str = "vllm"):
        self.host = host
        self.port = port
        self.prefix = prefix
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send(self, metric: str, value: float, metric_type: str):
        try:
            message = f"{self.prefix}.{metric}:{value}|{metric_type}"
            self.sock.sendto(message.encode(), (self.host, self.port))
        except Exception:
            pass

    def timing(self, metric: str, value_ms: float):
        self._send(metric, value_ms, "ms")

    def gauge(self, metric: str, value: float):
        self._send(metric, value, "g")

    def counter(self, metric: str, value: float = 1):
        self._send(metric, value, "c")


class StatsDStatLogger:
    """StatsD logger for vLLM metrics."""

    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]):
        self.engine_indexes = engine_indexes
        host = os.getenv("VLLM_STATSD_HOST")
        port = int(os.getenv("VLLM_STATSD_PORT", "8125"))
        self.client = StatsDClient(host, port) if host else None

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ):
        if not self.client:
            return

        if scheduler_stats:
            self.client.gauge("num_requests_running", scheduler_stats.num_running_reqs)
            self.client.gauge("num_requests_waiting", scheduler_stats.num_waiting_reqs)
            self.client.gauge(
                "kv_cache_usage_perc", scheduler_stats.kv_cache_usage * 100
            )
            self.client.counter(
                "prefix_cache_queries", scheduler_stats.prefix_cache_stats.queries
            )
            self.client.counter(
                "prefix_cache_hits", scheduler_stats.prefix_cache_stats.hits
            )

            if scheduler_stats.connector_prefix_cache_stats:
                self.client.counter(
                    "external_prefix_cache_queries",
                    scheduler_stats.connector_prefix_cache_stats.queries,
                )
                self.client.counter(
                    "external_prefix_cache_hits",
                    scheduler_stats.connector_prefix_cache_stats.hits,
                )

        if mm_cache_stats:
            self.client.counter("mm_cache_queries", mm_cache_stats.queries)
            self.client.counter("mm_cache_hits", mm_cache_stats.hits)

        if not iteration_stats:
            return

        self.client.counter("num_preemptions", iteration_stats.num_preempted_reqs)
        self.client.counter("prompt_tokens", iteration_stats.num_prompt_tokens)
        self.client.counter("generation_tokens", iteration_stats.num_generation_tokens)

        for ttft in iteration_stats.time_to_first_tokens_iter:
            self.client.timing("time_to_first_token_seconds", ttft * 1000)

        for itl in iteration_stats.inter_token_latencies_iter:
            self.client.timing("inter_token_latency_seconds", itl * 1000)

        for vision_time in iteration_stats.vision_encoding_times_iter:
            self.client.timing("vision_encoding_seconds", vision_time * 1000)

        for req in iteration_stats.finished_requests:
            self.client.counter(f"request_success.{req.finish_reason}")
            self.client.timing("e2e_request_latency_seconds", req.e2e_latency * 1000)
            self.client.timing("request_queue_time_seconds", req.queued_time * 1000)
            self.client.timing(
                "request_inference_time_seconds", req.inference_time * 1000
            )
            self.client.timing("request_prefill_time_seconds", req.prefill_time * 1000)
            self.client.timing("request_decode_time_seconds", req.decode_time * 1000)

    def log(self):
        pass

    def log_engine_initialized(self):
        pass


def get_statsd_client() -> StatsDClient | None:
    """Get global StatsD client if configured."""
    global _statsd_client
    if _statsd_client is None:
        host = os.getenv("VLLM_STATSD_HOST")
        port = os.getenv("VLLM_STATSD_PORT", "8125")
        if host:
            try:
                _statsd_client = StatsDClient(host, int(port))
            except Exception:
                pass
    return _statsd_client


def record_metric(metric: str, value: float, metric_type: str = "timing"):
    """Record a metric value and send to StatsD if configured.

    For vision encoding metrics, also records to thread-local buffer.
    """
    # Record vision encoding times to buffer for collection
    if metric == "vision_encoding_seconds" and metric_type == "timing":
        record_vision_encoding_time(value)

    # Also send to StatsD immediately if configured
    client = get_statsd_client()
    if client:
        if metric_type == "timing":
            client.timing(metric, value * 1000)
        elif metric_type == "counter":
            client.counter(metric, value)
        elif metric_type == "gauge":
            client.gauge(metric, value)
