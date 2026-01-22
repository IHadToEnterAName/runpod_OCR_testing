"""
Traffic Controller Module
=========================
Intelligent traffic routing, load management, and failover handling.
Routes requests to appropriate models with rate limiting and health checks.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import httpx
from openai import AsyncOpenAI

from config.settings import get_config

config = get_config()


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ModelType(Enum):
    """Available model types."""
    VISION = "vision"
    REASONING = "reasoning"
    EMBEDDING = "embedding"


class ServerStatus(Enum):
    """Server health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class RequestPriority(Enum):
    """Request priority levels."""
    HIGH = 1      # Interactive user queries
    NORMAL = 2    # Standard requests
    LOW = 3       # Background tasks, batch processing
    BATCH = 4     # Bulk operations


@dataclass
class ServerHealth:
    """Health information for a server."""
    status: ServerStatus = ServerStatus.UNKNOWN
    last_check: float = 0.0
    response_time_ms: float = 0.0
    error_count: int = 0
    success_count: int = 0
    last_error: Optional[str] = None
    consecutive_failures: int = 0


@dataclass
class RateLimitState:
    """Rate limiting state for a model."""
    requests_per_minute: int = 60
    current_count: int = 0
    window_start: float = field(default_factory=time.time)
    queue: deque = field(default_factory=deque)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for failover handling."""
    is_open: bool = False
    failure_count: int = 0
    last_failure_time: float = 0.0
    open_until: float = 0.0
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds


@dataclass
class TrafficStats:
    """Traffic statistics."""
    total_requests: int = 0
    vision_requests: int = 0
    reasoning_requests: int = 0
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
    Centralized traffic routing and control.

    Features:
    - Model routing (vision vs reasoning)
    - Rate limiting per model
    - Circuit breaker for failures
    - Health monitoring
    - Request prioritization
    - Load balancing
    """

    def __init__(self):
        """Initialize the traffic controller."""
        # Model clients
        self._vision_client: Optional[AsyncOpenAI] = None
        self._reasoning_client: Optional[AsyncOpenAI] = None

        # Health status
        self._health: Dict[ModelType, ServerHealth] = {
            ModelType.VISION: ServerHealth(),
            ModelType.REASONING: ServerHealth(),
        }

        # Rate limiting
        self._rate_limits: Dict[ModelType, RateLimitState] = {
            ModelType.VISION: RateLimitState(requests_per_minute=30),  # Vision is slower
            ModelType.REASONING: RateLimitState(requests_per_minute=60),
        }

        # Circuit breakers
        self._circuit_breakers: Dict[ModelType, CircuitBreakerState] = {
            ModelType.VISION: CircuitBreakerState(failure_threshold=3, recovery_timeout=30),
            ModelType.REASONING: CircuitBreakerState(failure_threshold=5, recovery_timeout=20),
        }

        # Semaphores for concurrency control
        self._semaphores: Dict[ModelType, asyncio.Semaphore] = {
            ModelType.VISION: asyncio.Semaphore(config.performance.vision_concurrent_limit),
            ModelType.REASONING: asyncio.Semaphore(config.performance.llm_concurrent_limit),
        }

        # Statistics
        self._stats = TrafficStats()

        # Request queues for prioritization
        self._priority_queues: Dict[ModelType, Dict[RequestPriority, deque]] = {
            ModelType.VISION: {p: deque() for p in RequestPriority},
            ModelType.REASONING: {p: deque() for p in RequestPriority},
        }

        # Health check interval
        self._health_check_interval = 30.0  # seconds
        self._last_health_check = 0.0

    # =========================================================================
    # CLIENT MANAGEMENT
    # =========================================================================

    def get_vision_client(self) -> AsyncOpenAI:
        """Get or create vision model client."""
        if self._vision_client is None:
            self._vision_client = AsyncOpenAI(
                base_url=config.models.vision_base_url,
                api_key="EMPTY",
                timeout=60.0
            )
        return self._vision_client

    def get_reasoning_client(self) -> AsyncOpenAI:
        """Get or create reasoning model client."""
        if self._reasoning_client is None:
            self._reasoning_client = AsyncOpenAI(
                base_url=config.models.reasoning_base_url,
                api_key="EMPTY",
                timeout=120.0
            )
        return self._reasoning_client

    def get_client(self, model_type: ModelType) -> AsyncOpenAI:
        """Get client for specified model type."""
        if model_type == ModelType.VISION:
            return self.get_vision_client()
        elif model_type == ModelType.REASONING:
            return self.get_reasoning_client()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # =========================================================================
    # HEALTH MONITORING
    # =========================================================================

    async def check_health(self, model_type: ModelType) -> ServerHealth:
        """Check health of a specific model server."""
        health = self._health[model_type]

        try:
            if model_type == ModelType.VISION:
                url = f"{config.models.vision_base_url.rstrip('/v1')}/v1/models"
            else:
                url = f"{config.models.reasoning_base_url.rstrip('/v1')}/v1/models"

            start_time = time.time()

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url)

            response_time = (time.time() - start_time) * 1000

            if response.status_code == 200:
                health.status = ServerStatus.HEALTHY
                health.response_time_ms = response_time
                health.success_count += 1
                health.consecutive_failures = 0
                health.last_error = None

                # Reset circuit breaker on success
                cb = self._circuit_breakers[model_type]
                if cb.is_open and time.time() > cb.open_until:
                    cb.is_open = False
                    cb.failure_count = 0
                    print(f"✅ Circuit breaker CLOSED for {model_type.value}")
            else:
                health.status = ServerStatus.DEGRADED
                health.error_count += 1
                health.consecutive_failures += 1
                health.last_error = f"HTTP {response.status_code}"

        except Exception as e:
            health.status = ServerStatus.UNHEALTHY
            health.error_count += 1
            health.consecutive_failures += 1
            health.last_error = str(e)

            # Update circuit breaker
            self._update_circuit_breaker(model_type, error=True)

        health.last_check = time.time()
        return health

    async def check_all_health(self) -> Dict[ModelType, ServerHealth]:
        """Check health of all model servers."""
        results = {}
        for model_type in [ModelType.VISION, ModelType.REASONING]:
            results[model_type] = await self.check_health(model_type)
        return results

    async def periodic_health_check(self):
        """Run periodic health checks (call from background task)."""
        current_time = time.time()
        if current_time - self._last_health_check >= self._health_check_interval:
            await self.check_all_health()
            self._last_health_check = current_time

    def is_healthy(self, model_type: ModelType) -> bool:
        """Check if a model server is healthy."""
        health = self._health[model_type]
        cb = self._circuit_breakers[model_type]

        # Check circuit breaker
        if cb.is_open and time.time() < cb.open_until:
            return False

        # Check health status
        return health.status in [ServerStatus.HEALTHY, ServerStatus.DEGRADED, ServerStatus.UNKNOWN]

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _update_circuit_breaker(self, model_type: ModelType, error: bool = False):
        """Update circuit breaker state."""
        cb = self._circuit_breakers[model_type]

        if error:
            cb.failure_count += 1
            cb.last_failure_time = time.time()

            if cb.failure_count >= cb.failure_threshold and not cb.is_open:
                cb.is_open = True
                cb.open_until = time.time() + cb.recovery_timeout
                self._stats.circuit_breaks += 1
                print(f"⚠️ Circuit breaker OPEN for {model_type.value} (failures: {cb.failure_count})")
        else:
            # Successful request - reset on consecutive successes
            if cb.failure_count > 0:
                cb.failure_count = max(0, cb.failure_count - 1)

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    def _check_rate_limit(self, model_type: ModelType) -> bool:
        """Check if request is within rate limit."""
        rl = self._rate_limits[model_type]
        current_time = time.time()

        # Reset window if expired
        if current_time - rl.window_start >= 60.0:
            rl.window_start = current_time
            rl.current_count = 0

        # Check limit
        if rl.current_count >= rl.requests_per_minute:
            self._stats.rate_limited += 1
            return False

        rl.current_count += 1
        return True

    async def wait_for_rate_limit(self, model_type: ModelType):
        """Wait until rate limit allows request."""
        rl = self._rate_limits[model_type]

        while not self._check_rate_limit(model_type):
            # Calculate wait time
            wait_time = 60.0 - (time.time() - rl.window_start)
            if wait_time > 0:
                print(f"⏳ Rate limited for {model_type.value}, waiting {wait_time:.1f}s")
                await asyncio.sleep(min(wait_time, 5.0))  # Check every 5s max

    # =========================================================================
    # MODEL ROUTING
    # =========================================================================

    def determine_model(
        self,
        query: str,
        has_images: bool = False,
        force_model: Optional[ModelType] = None
    ) -> ModelType:
        """
        Determine which model to use based on query characteristics.

        Args:
            query: The user query
            has_images: Whether the request includes images
            force_model: Force a specific model (overrides auto-detection)

        Returns:
            ModelType to use
        """
        if force_model:
            return force_model

        # If images are present, use vision model
        if has_images:
            if self.is_healthy(ModelType.VISION):
                return ModelType.VISION
            else:
                print("⚠️ Vision model unhealthy, falling back to reasoning")
                return ModelType.REASONING

        # Check for vision-related keywords
        vision_keywords = [
            "image", "picture", "photo", "chart", "graph", "diagram",
            "figure", "illustration", "screenshot", "visual", "see",
            "look at", "shown in", "display", "depict"
        ]

        query_lower = query.lower()
        is_vision_query = any(kw in query_lower for kw in vision_keywords)

        if is_vision_query and self.is_healthy(ModelType.VISION):
            return ModelType.VISION

        # Default to reasoning model
        return ModelType.REASONING

    # =========================================================================
    # REQUEST EXECUTION
    # =========================================================================

    async def execute_request(
        self,
        model_type: ModelType,
        request_func: Callable,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 120.0
    ) -> Any:
        """
        Execute a request with traffic control.

        Args:
            model_type: Which model to use
            request_func: Async function to execute
            priority: Request priority
            timeout: Request timeout in seconds

        Returns:
            Result from request_func
        """
        self._stats.total_requests += 1

        if model_type == ModelType.VISION:
            self._stats.vision_requests += 1
        else:
            self._stats.reasoning_requests += 1

        # Check circuit breaker
        cb = self._circuit_breakers[model_type]
        if cb.is_open:
            if time.time() < cb.open_until:
                raise RuntimeError(f"Circuit breaker open for {model_type.value}")
            else:
                # Try to close circuit breaker (half-open state)
                cb.is_open = False
                print(f"🔄 Circuit breaker half-open for {model_type.value}, attempting request")

        # Wait for rate limit
        await self.wait_for_rate_limit(model_type)

        # Acquire semaphore for concurrency control
        semaphore = self._semaphores[model_type]

        start_time = time.time()

        try:
            async with semaphore:
                result = await asyncio.wait_for(request_func(), timeout=timeout)

            # Update stats
            response_time = (time.time() - start_time) * 1000
            self._update_avg_response_time(response_time)

            # Update circuit breaker (success)
            self._update_circuit_breaker(model_type, error=False)

            return result

        except asyncio.TimeoutError:
            self._stats.errors += 1
            self._update_circuit_breaker(model_type, error=True)
            raise RuntimeError(f"Request timeout for {model_type.value}")

        except Exception as e:
            self._stats.errors += 1
            self._update_circuit_breaker(model_type, error=True)
            raise

    def _update_avg_response_time(self, response_time_ms: float):
        """Update average response time."""
        total = self._stats.total_requests
        if total == 1:
            self._stats.avg_response_time_ms = response_time_ms
        else:
            # Running average
            self._stats.avg_response_time_ms = (
                (self._stats.avg_response_time_ms * (total - 1) + response_time_ms) / total
            )

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    async def generate_with_vision(
        self,
        messages: List[Dict],
        max_tokens: int = 512,
        temperature: float = 0.3,
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> str:
        """Generate response using vision model."""
        client = self.get_vision_client()

        async def request():
            response = await client.chat.completions.create(
                model=config.models.vision_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content

        return await self.execute_request(
            ModelType.VISION,
            request,
            priority=priority,
            timeout=60.0
        )

    async def generate_with_reasoning(
        self,
        messages: List[Dict],
        max_tokens: int = 2048,
        temperature: float = 0.3,
        stream: bool = False,
        priority: RequestPriority = RequestPriority.NORMAL
    ):
        """Generate response using reasoning model."""
        client = self.get_reasoning_client()

        async def request():
            return await client.chat.completions.create(
                model=config.models.reasoning_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream
            )

        return await self.execute_request(
            ModelType.REASONING,
            request,
            priority=priority,
            timeout=120.0
        )

    # =========================================================================
    # STATISTICS AND MONITORING
    # =========================================================================

    def get_stats(self) -> Dict:
        """Get traffic statistics."""
        return {
            "total_requests": self._stats.total_requests,
            "vision_requests": self._stats.vision_requests,
            "reasoning_requests": self._stats.reasoning_requests,
            "cache_hits": self._stats.cache_hits,
            "rate_limited": self._stats.rate_limited,
            "circuit_breaks": self._stats.circuit_breaks,
            "avg_response_time_ms": round(self._stats.avg_response_time_ms, 2),
            "errors": self._stats.errors,
            "health": {
                "vision": self._health[ModelType.VISION].status.value,
                "reasoning": self._health[ModelType.REASONING].status.value,
            },
            "circuit_breakers": {
                "vision_open": self._circuit_breakers[ModelType.VISION].is_open,
                "reasoning_open": self._circuit_breakers[ModelType.REASONING].is_open,
            }
        }

    def reset_stats(self):
        """Reset traffic statistics."""
        self._stats = TrafficStats()

    def get_health_summary(self) -> str:
        """Get a formatted health summary."""
        lines = ["📊 Traffic Controller Status:"]

        for model_type in [ModelType.VISION, ModelType.REASONING]:
            health = self._health[model_type]
            cb = self._circuit_breakers[model_type]

            status_emoji = {
                ServerStatus.HEALTHY: "✅",
                ServerStatus.DEGRADED: "⚠️",
                ServerStatus.UNHEALTHY: "❌",
                ServerStatus.UNKNOWN: "❓"
            }.get(health.status, "❓")

            cb_status = "🔴 OPEN" if cb.is_open else "🟢 CLOSED"

            lines.append(
                f"  {model_type.value}: {status_emoji} {health.status.value} | "
                f"Circuit: {cb_status} | "
                f"Latency: {health.response_time_ms:.0f}ms"
            )

        lines.append(f"  Total requests: {self._stats.total_requests}")
        lines.append(f"  Errors: {self._stats.errors}")

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


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def route_to_model(
    query: str,
    messages: List[Dict],
    has_images: bool = False,
    max_tokens: int = 2048,
    temperature: float = 0.3,
    stream: bool = False,
    priority: RequestPriority = RequestPriority.NORMAL
):
    """
    Automatically route request to appropriate model.

    Args:
        query: User query for model selection
        messages: Chat messages
        has_images: Whether request includes images
        max_tokens: Max tokens for generation
        temperature: Generation temperature
        stream: Whether to stream response
        priority: Request priority

    Returns:
        Model response (stream if stream=True)
    """
    controller = get_traffic_controller()

    # Determine model
    model_type = controller.determine_model(query, has_images)
    print(f"🔀 Routing to: {model_type.value}")

    if model_type == ModelType.VISION:
        return await controller.generate_with_vision(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority
        )
    else:
        return await controller.generate_with_reasoning(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            priority=priority
        )


async def check_servers_health() -> Dict:
    """Check health of all model servers."""
    controller = get_traffic_controller()
    await controller.check_all_health()
    return controller.get_stats()
