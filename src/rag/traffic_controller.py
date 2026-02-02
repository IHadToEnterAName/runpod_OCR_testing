"""
Traffic Controller Module
=========================
Traffic management for single Qwen3-VL model endpoint.
Rate limiting, circuit breaker, and health monitoring.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
import httpx
from openai import AsyncOpenAI

from config.settings import get_config

config = get_config()


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ServerStatus(Enum):
    """Server health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RequestPriority(Enum):
    """Request priority levels."""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class ServerHealth:
    """Health information for the vLLM server."""
    status: ServerStatus = ServerStatus.UNKNOWN
    last_check: float = 0.0
    response_time_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None
    consecutive_failures: int = 0


@dataclass
class RateLimitState:
    """Rate limiting state."""
    requests_per_minute: int = 60
    current_count: int = 0
    window_start: float = field(default_factory=time.time)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for failover handling."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    open_until: float = 0.0
    failure_threshold: int = 5
    recovery_timeout: float = 20.0


@dataclass
class TrafficStats:
    """Traffic statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    rate_limited: int = 0
    circuit_breaks: int = 0
    avg_response_time_ms: float = 0.0
    errors: int = 0


# =============================================================================
# TRAFFIC CONTROLLER
# =============================================================================

class TrafficController:
    """
    Traffic controller for single Qwen3-VL endpoint.

    Features:
    - Rate limiting
    - Circuit breaker for failures
    - Health monitoring
    - Concurrency control
    """

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

        self._health = ServerHealth()

        self._rate_limit = RateLimitState(
            requests_per_minute=config.traffic.model_rpm
        )

        self._circuit_breaker = CircuitBreakerState(
            failure_threshold=config.traffic.failure_threshold,
            recovery_timeout=config.traffic.recovery_timeout
        )

        self._semaphore = asyncio.Semaphore(config.performance.model_concurrent_limit)

        self._stats = TrafficStats()

        self._health_check_interval = config.traffic.health_check_interval
        self._last_health_check = 0.0

    # =========================================================================
    # CLIENT MANAGEMENT
    # =========================================================================

    def get_client(self) -> AsyncOpenAI:
        """Get or create Qwen3-VL client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=config.models.base_url,
                api_key="EMPTY",
                timeout=config.traffic.model_timeout
            )
        return self._client

    # =========================================================================
    # HEALTH MONITORING
    # =========================================================================

    async def check_health(self) -> ServerHealth:
        """Check health of the vLLM server."""
        try:
            url = f"{config.models.base_url.rstrip('/v1')}/v1/models"
            start_time = time.time()

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)

            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                self._health.status = ServerStatus.HEALTHY
                self._health.response_time_ms = response_time
                self._health.success_count += 1
                self._health.consecutive_failures = 0
                self._health.last_error = None

                # Reset circuit breaker on success
                cb = self._circuit_breaker
                if cb.is_open and time.time() > cb.open_until:
                    cb.is_open = False
                    cb.failure_count = 0
                    print(f"Circuit breaker CLOSED")
            else:
                self._health.status = ServerStatus.DEGRADED
                self._health.error_count += 1
                self._health.consecutive_failures += 1
                self._health.last_error = f"HTTP {response.status_code}"

        except Exception as e:
            self._health.status = ServerStatus.UNHEALTHY
            self._health.error_count += 1
            self._health.consecutive_failures += 1
            self._health.last_error = str(e)
            self._update_circuit_breaker(error=True)

        self._health.last_check = time.time()
        return self._health

    async def periodic_health_check(self):
        """Run periodic health check."""
        current_time = time.time()
        if current_time - self._last_health_check >= self._health_check_interval:
            await self.check_health()
            self._last_health_check = current_time

    def is_healthy(self) -> bool:
        """Check if the vLLM server is healthy."""
        cb = self._circuit_breaker
        if cb.is_open and time.time() < cb.open_until:
            return False
        return self._health.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED, ServerStatus.UNKNOWN]

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _update_circuit_breaker(self, error: bool = False):
        """Update circuit breaker state."""
        cb = self._circuit_breaker

        if error:
            cb.failure_count += 1
            cb.last_failure_time = time.time()

            if cb.failure_count >= cb.failure_threshold and not cb.is_open:
                cb.is_open = True
                cb.open_until = time.time() + cb.recovery_timeout
                self._stats.circuit_breaks += 1
                print(f"Circuit breaker OPEN (failures: {cb.failure_count})")
        else:
            if cb.failure_count > 0:
                cb.failure_count = max(0, cb.failure_count - 1)

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limit."""
        rl = self._rate_limit
        current_time = time.time()

        if current_time - rl.window_start >= 60.0:
            rl.window_start = current_time
            rl.current_count = 0

        if rl.current_count >= rl.requests_per_minute:
            self._stats.rate_limited += 1
            return False

        rl.current_count += 1
        return True

    async def wait_for_rate_limit(self):
        """Wait until rate limit allows request."""
        rl = self._rate_limit

        while not self._check_rate_limit():
            wait_time = 60.0 - (time.time() - rl.window_start)
            if wait_time > 0:
                print(f"Rate limited, waiting {wait_time:.1f}s")
                await asyncio.sleep(min(wait_time, 5.0))

    # =========================================================================
    # REQUEST EXECUTION
    # =========================================================================

    async def execute_request(
        self,
        request_func: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a request with traffic control.

        Args:
            request_func: Async function to execute
            priority: Request priority
            timeout: Request timeout in seconds

        Returns:
            Result from request_func
        """
        if timeout is None:
            timeout = config.traffic.model_timeout

        self._stats.total_requests += 1

        # Check circuit breaker
        cb = self._circuit_breaker
        if cb.is_open:
            if time.time() < cb.open_until:
                raise RuntimeError("Circuit breaker open for vLLM server")
            else:
                cb.is_open = False
                print("Circuit breaker half-open, attempting request")

        # Wait for rate limit
        await self.wait_for_rate_limit()

        start_time = time.time()

        try:
            async with self._semaphore:
                result = await asyncio.wait_for(request_func(), timeout=timeout)

            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)
            self._update_circuit_breaker(error=False)

            return result

        except asyncio.TimeoutError:
            self._stats.errors += 1
            self._update_circuit_breaker(error=True)
            raise RuntimeError(f"Request timeout ({timeout}s)")

        except Exception:
            self._stats.errors += 1
            self._update_circuit_breaker(error=True)
            raise

    def _update_avg_response_time(self, response_time_ms: float):
        """Update running average response time."""
        total = self._stats.total_requests
        if total == 1:
            self._stats.avg_response_time_ms = response_time_ms
        else:
            self._stats.avg_response_time_ms = (
                (self._stats.avg_response_time_ms * (total - 1) + response_time_ms) / total
            )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get traffic statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "cache_hits": self._stats.cache_hits,
            "rate_limited": self._stats.rate_limited,
            "circuit_breaks": self._stats.circuit_breaks,
            "avg_response_time_ms": round(self._stats.avg_response_time_ms, 2),
            "errors": self._stats.errors,
            "health": self._health.status.value,
            "circuit_breaker_open": self._circuit_breaker.is_open,
        }

    def reset_stats(self):
        """Reset traffic statistics."""
        self._stats = TrafficStats()

    def get_health_summary(self) -> str:
        """Get a formatted health summary."""
        status_map = {
            ServerStatus.HEALTHY: "HEALTHY",
            ServerStatus.DEGRADED: "DEGRADED",
            ServerStatus.UNHEALTHY: "UNHEALTHY",
            ServerStatus.UNKNOWN: "UNKNOWN"
        }

        cb_status = "OPEN" if self._circuit_breaker.is_open else "CLOSED"

        lines = [
            "Traffic Controller Status:",
            f"  Server: {status_map.get(self._health.status, 'UNKNOWN')}",
            f"  Circuit breaker: {cb_status}",
            f"  Latency: {self._health.response_time_ms:.0f}ms",
            f"  Total requests: {self._stats.total_requests}",
            f"  Errors: {self._stats.errors}",
        ]

        return "\n".join(lines)


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_controller_instance: Optional[TrafficController] = None


def get_traffic_controller() -> TrafficController:
    """Get the global traffic controller instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = TrafficController()
    return _controller_instance
