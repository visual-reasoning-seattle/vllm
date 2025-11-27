# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""StatsD metrics logger for vLLM."""

import socket

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.metrics.loggers import AggregateStatLoggerBase
from vllm.v1.metrics.stats import (
    IterationStats,
    MultiModalCacheStats,
    SchedulerStats,
)

logger = init_logger(__name__)


class StatsDClient:
    """StatsD UDP client."""

    def __init__(self, host: str, port: int, prefix: str = "vllm") -> None:
        self._host = host
        self._port = port
        self._prefix = prefix
        self._sock: socket.socket | None = None
        self._connect()

    def _connect(self) -> None:
        """Initialize the UDP socket."""
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        except OSError as e:
            logger.debug("Failed to create StatsD UDP socket: %s", e)
            self._sock = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def prefix(self) -> str:
        return self._prefix

    def __del__(self) -> None:
        """Clean up the socket on destruction."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError as e:
                logger.debug("Error closing StatsD socket: %s", e)

    def _send(self, metric: str, value: float, metric_type: str) -> None:
        if self._sock is None:
            return
        try:
            message = f"{self._prefix}.{metric}:{value}|{metric_type}"
            self._sock.sendto(message.encode(), (self._host, self._port))
        except OSError as e:
            logger.debug("Failed to send StatsD metric %s: %s", metric, e)

    def timing(self, metric: str, value_ms: float) -> None:
        self._send(metric, value_ms, "ms")

    def gauge(self, metric: str, value: float) -> None:
        self._send(metric, value, "g")

    def counter(self, metric: str, value: float = 1) -> None:
        self._send(metric, value, "c")


class StatsDStatLogger(AggregateStatLoggerBase):
    """StatsD logger for vLLM metrics."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int],
        host: str,
        port_str: str,
    ) -> None:
        self._engine_indexes = engine_indexes
        self._client: StatsDClient | None = None

        # Validate host
        host = host.strip()
        if not host:
            logger.warning("VLLM_STATSD_HOST is empty. StatsD metrics disabled.")
            return

        # Validate port
        try:
            port = int(port_str)
        except ValueError:
            logger.warning(
                "VLLM_STATSD_PORT must be an integer, got '%s'. "
                "StatsD metrics disabled.",
                port_str,
            )
            return

        if port < 1 or port > 65535:
            logger.warning(
                "VLLM_STATSD_PORT must be between 1 and 65535, got %d. "
                "StatsD metrics disabled.",
                port,
            )
            return

        self._client = StatsDClient(host, port)
        logger.info(
            "StatsD metrics enabled: sending to %s:%d with prefix 'vllm'",
            host,
            port,
        )

    @property
    def client(self) -> StatsDClient | None:
        return self._client

    @property
    def engine_indexes(self) -> list[int]:
        return self._engine_indexes

    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ) -> None:
        if not self._client:
            return

        if scheduler_stats:
            self._client.gauge(
                "num_requests_running", scheduler_stats.num_running_reqs
            )
            self._client.gauge(
                "num_requests_waiting", scheduler_stats.num_waiting_reqs
            )
            self._client.gauge(
                "kv_cache_usage_perc", scheduler_stats.kv_cache_usage * 100
            )
            self._client.counter(
                "prefix_cache_queries", scheduler_stats.prefix_cache_stats.queries
            )
            self._client.counter(
                "prefix_cache_hits", scheduler_stats.prefix_cache_stats.hits
            )

            if scheduler_stats.connector_prefix_cache_stats:
                self._client.counter(
                    "external_prefix_cache_queries",
                    scheduler_stats.connector_prefix_cache_stats.queries,
                )
                self._client.counter(
                    "external_prefix_cache_hits",
                    scheduler_stats.connector_prefix_cache_stats.hits,
                )

        if mm_cache_stats:
            self._client.counter("mm_cache_queries", mm_cache_stats.queries)
            self._client.counter("mm_cache_hits", mm_cache_stats.hits)

        if not iteration_stats:
            return

        self._client.counter("num_preemptions", iteration_stats.num_preempted_reqs)
        self._client.counter("prompt_tokens", iteration_stats.num_prompt_tokens)
        self._client.counter(
            "generation_tokens", iteration_stats.num_generation_tokens
        )

        for ttft in iteration_stats.time_to_first_tokens_iter:
            self._client.timing("time_to_first_token_seconds", ttft * 1000)

        for itl in iteration_stats.inter_token_latencies_iter:
            self._client.timing("inter_token_latency_seconds", itl * 1000)

        for req in iteration_stats.finished_requests:
            self._client.counter(f"request_success.{req.finish_reason}")
            self._client.timing("e2e_request_latency_seconds", req.e2e_latency * 1000)
            self._client.timing("request_queue_time_seconds", req.queued_time * 1000)
            self._client.timing(
                "request_inference_time_seconds", req.inference_time * 1000
            )
            self._client.timing(
                "request_prefill_time_seconds", req.prefill_time * 1000
            )
            self._client.timing("request_decode_time_seconds", req.decode_time * 1000)

    def log(self) -> None:
        pass

    def log_engine_initialized(self) -> None:
        pass
