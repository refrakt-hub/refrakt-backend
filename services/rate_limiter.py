"""Rate limiting and admission-control helpers."""

from typing import Callable, List, Optional

from fastapi import Depends, Request, Response
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
from redis.asyncio import Redis

from config import Settings, get_settings

DEFAULT_TIER_HEADER = "X-RateLimit-Tier"
PREMIUM_TIER_VALUE = "premium"
USER_ID_HEADER = "X-User-ID"

_settings: Settings = get_settings()
_rate_limiter_initialized = False
_redis: Optional[Redis] = None
_dependencies: dict[str, List[Depends]] = {"run": [], "assistant": []}


async def init_rate_limiter(settings: Settings) -> None:
    """Initialise FastAPI-Limiter with shared Redis connection."""
    global _rate_limiter_initialized, _redis, _settings, _dependencies
    _settings = settings
    _dependencies = _build_rate_limiters(settings)
    if not settings.RATE_LIMIT_ENABLED:
        return
    if not settings.QUEUE_URL:
        raise RuntimeError("RATE_LIMIT_ENABLED requires QUEUE_URL to be configured")

    identifier = _client_identifier_factory(settings)
    _redis = Redis.from_url(
        settings.QUEUE_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    await FastAPILimiter.init(
        _redis,
        prefix="refrakt-rate-limit",
        identifier=identifier,
    )
    _rate_limiter_initialized = True


async def shutdown_rate_limiter() -> None:
    """Tear down FastAPI-Limiter connection."""
    global _rate_limiter_initialized, _redis
    if not _rate_limiter_initialized:
        return
    await FastAPILimiter.close()
    _rate_limiter_initialized = False
    _redis = None


def _client_identifier_factory(settings: Settings) -> Callable[[Request], str]:
    async def identifier(request: Request) -> str:
        return _extract_ip(settings, request)

    return identifier


def _user_identifier_factory(settings: Settings) -> Callable[[Request], str]:
    async def identifier(request: Request) -> str:
        user_id = (
            request.headers.get(USER_ID_HEADER)
            or request.headers.get("X-UserID")
            or request.query_params.get("user_id")
        )
        if user_id:
            return user_id
        return _extract_ip(settings, request)

    return identifier


def _extract_ip(settings: Settings, request: Request) -> str:
    header = request.headers.get(settings.CLIENT_IP_HEADER)
    if header:
        return header.split(",")[0].strip()

    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    client = request.client
    return client.host if client else "unknown"


class TieredRateLimiter:
    """Wrapper that applies premium multiplier when tier header is present."""

    def __init__(
        self,
        standard: RateLimiter,
        premium: RateLimiter,
        enabled: bool,
        header_name: str = DEFAULT_TIER_HEADER,
    ):
        self.standard = standard
        self.premium = premium
        self.enabled = enabled
        self.header_name = header_name

    async def __call__(self, request: Request, response: Response) -> None:
        if not self.enabled:
            return
        tier = (request.headers.get(self.header_name) or "").strip().lower()
        limiter = self.premium if tier == PREMIUM_TIER_VALUE else self.standard
        await limiter(request, response)


def _build_rate_limiters(settings: Settings) -> dict[str, List[Depends]]:
    dependencies: dict[str, List[Depends]] = {
        "run": [],
        "assistant": [],
    }
    if not settings.RATE_LIMIT_ENABLED:
        return dependencies

    ip_identifier = _client_identifier_factory(settings)
    user_identifier = _user_identifier_factory(settings)
    premium_multiplier = max(settings.RATE_LIMIT_PREMIUM_MULTIPLIER, 1.0)

    def _tiered(times: int, seconds: int, identifier: Callable[[Request], str]) -> TieredRateLimiter:
        premium_times = max(int(times * premium_multiplier), times)
        standard = RateLimiter(times=times, seconds=seconds, identifier=identifier)
        premium = RateLimiter(times=premium_times, seconds=seconds, identifier=identifier)
        return TieredRateLimiter(
            standard=standard,
            premium=premium,
            enabled=settings.RATE_LIMIT_ENABLED,
        )

    run_ip_limiter = _tiered(
        times=settings.RATE_LIMIT_RUN_BURST,
        seconds=60,
        identifier=ip_identifier,
    )
    run_user_limiter = _tiered(
        times=settings.RATE_LIMIT_RUN_PER_MINUTE,
        seconds=60,
        identifier=user_identifier,
    )
    assistant_limiter = _tiered(
        times=settings.RATE_LIMIT_ASSISTANT_PER_MINUTE,
        seconds=60,
        identifier=user_identifier,
    )

    dependencies["run"] = [Depends(run_ip_limiter), Depends(run_user_limiter)]
    dependencies["assistant"] = [Depends(assistant_limiter)]
    return dependencies


_dependencies = _build_rate_limiters(_settings)


def get_run_rate_limit_dependencies() -> List[Depends]:
    """Return dependency list for the /run endpoint."""
    return list(_dependencies["run"])


def get_assistant_rate_limit_dependencies() -> List[Depends]:
    """Return dependency list for assistant endpoints."""
    return list(_dependencies["assistant"])

