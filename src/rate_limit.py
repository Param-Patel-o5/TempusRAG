"""Shared rate limiter, retry helper, and circuit breaker per provider.

Coordinates API calls across modules for:
- gemini_generate (LLM Call 1, promise extraction) — 15 RPM, 4s interval
- groq (LLM Call 2, cross-year judge) — 30 RPM, 2.5s interval

Thread-safe and synchronous. Safe inside Streamlit.
"""

import logging
import random
import re
import threading
import time
from typing import Any, Callable

from src.config import (
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    GEMINI_GENERATE_RPM,
    GEMINI_JUDGE_REQUEST_INTERVAL,
    GEMINI_JUDGE_RPM,
    GEMINI_REQUEST_INTERVAL,
    GROQ_REQUEST_INTERVAL,
    GROQ_RPM,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)

MAX_SERVER_RETRY_DELAY = 120.0
"""If server requests a retry delay longer than this, fail fast instead of sleeping."""


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DailyQuotaExhaustedError(RuntimeError):
    """Raised when a provider daily quota is exhausted. Retrying cannot fix this."""


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a provider circuit breaker is open due to recent fatal errors."""


# ============================================================================
# ERROR CLASSIFICATION HELPERS
# ============================================================================

def get_error_status_code(error: Exception) -> int | None:
    """Extract HTTP status code from an exception if available."""
    for attr in ("status_code", "code", "http_status", "status"):
        val = getattr(error, attr, None)
        if isinstance(val, int):
            return val

    resp = getattr(error, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val

    return None


def is_daily_quota_error(error: Exception | str) -> bool:
    """Return True if error indicates daily quota exhaustion (not per-minute)."""
    err_text = str(error).lower()

    has_daily_keywords = (
        "per day" in err_text
        or "per_day" in err_text
        or "rpd" in err_text
        or "day limit" in err_text
        or "per calendar day" in err_text
        or "reset tomorrow" in err_text
        or (
            "daily" in err_text
            and any(k in err_text for k in ("limit", "quota", "exceeded", "exhausted"))
        )
    )

    if not has_daily_keywords:
        return False

    # Guard: message mentions per-minute, not per-day
    if any(k in err_text for k in ("per minute", "per_minute", "tpm", "rpm")):
        if "requests per minute" in err_text and "requests per day" not in err_text:
            return False
        if "tokens per minute" in err_text and "requests per day" not in err_text:
            return False

    return True


def is_retryable_error(error: Exception) -> bool:
    """Return True if error is transient and should be retried.

    Retry: 429, 5xx, timeouts, connection errors, DNS hiccups (getaddrinfo failed).
    Fail fast: daily quota, 400/401/403/404, safety blocks, plain RuntimeError.
    """
    import socket

    if is_daily_quota_error(error):
        return False

    if isinstance(error, (CircuitBreakerOpenError, DailyQuotaExhaustedError)):
        return False

    status_code = get_error_status_code(error)
    if status_code is not None:
        if status_code in (400, 401, 403, 404):
            return False
        if status_code == 429:
            return True
        if 500 <= status_code < 600:
            return True
        return False

    # Network-level errors: DNS failures (errno 11001/11002 on Windows), TCP resets, etc.
    if isinstance(error, (TimeoutError, ConnectionError, socket.gaierror, OSError)):
        # gaierror = DNS resolution failure (getaddrinfo failed) — always transient
        # OSError covers other low-level socket errors
        return True

    err_lower = str(error).lower()

    if any(k in err_lower for k in (
        "blocked", "safety", "content policy",
        "prompt was blocked", "finish_reason: safety"
    )):
        return False

    if type(error) is RuntimeError:
        if not any(k in err_lower for k in (
            "timeout", "timed out", "connection", "rate limit", "429"
        )):
            return False

    if any(k in err_lower for k in (
        "429", "rate limit", "resource_exhausted", "too many requests",
        "500", "502", "503", "504", "server error", "bad gateway",
        "service unavailable", "gateway timeout",
        "timeout", "timed out", "connection error",
        "connection reset", "connection refused",
        "getaddrinfo", "name or service not known",  # DNS failures as strings
        "temporary failure in name resolution",
    )):
        return True

    return False


def extract_retry_delay(error: Exception | str) -> float | None:
    """Extract server-recommended retry delay in seconds from error message.

    Supports protobuf format, key-value strings, and natural language.
    """
    err_text = str(error)

    # Protobuf: retry_delay { seconds: 18 nanos: 500000000 }
    pb_match = re.search(
        r"retry_delay\s*\{\s*seconds:\s*(\d+)(?:\s*nanos:\s*(\d+))?\s*\}",
        err_text, re.IGNORECASE,
    )
    if pb_match:
        try:
            sec = float(pb_match.group(1))
            if pb_match.group(2):
                sec += float(pb_match.group(2)) / 1e9
            return sec
        except (ValueError, TypeError):
            pass

    # Key-value: retryDelay: '18s'
    kv_match = re.search(
        r"retry[_-]?delay\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s?['\"]?",
        err_text, re.IGNORECASE,
    )
    if kv_match:
        try:
            return float(kv_match.group(1))
        except (ValueError, TypeError):
            pass

    # Natural language: retry in 18.5s / retry-after: 30
    nl_match = re.search(
        r"retry[-_\s]*(?:in|after|delay)?[:\s]+['\"]?(\d+(?:\.\d+)?)['\"]?\s*s?",
        err_text, re.IGNORECASE,
    )
    if nl_match:
        try:
            return float(nl_match.group(1))
        except (ValueError, TypeError):
            pass

    return None


# ============================================================================
# RATE LIMITER
# ============================================================================

class ProviderRateLimiter:
    """Thread-safe rate limiter with circuit breaker for a single API provider."""

    def __init__(
        self,
        name: str,
        requests_per_minute: int,
        min_interval_override: float | None = None,
        cooldown_seconds: float = CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    ):
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.cooldown_seconds = cooldown_seconds

        # Allow explicit interval override (e.g. GROQ_REQUEST_INTERVAL or GEMINI_REQUEST_INTERVAL)
        if min_interval_override is not None:
            self.min_interval = min_interval_override
        else:
            self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0

        self.last_request_time = 0.0
        self._lock = threading.Lock()

        # Circuit breaker
        self._is_broken = False
        self._broken_at = 0.0
        self._breaker_reason: str = ""

    def reset_circuit_breaker(self) -> None:
        with self._lock:
            self._is_broken = False
            self._broken_at = 0.0
            self._breaker_reason = ""

    def trip_circuit_breaker(self, reason: str) -> None:
        with self._lock:
            self._is_broken = True
            self._broken_at = time.time()
            self._breaker_reason = reason
        logger.error("Circuit breaker tripped for %s: %s", self.name, reason)

    def _check_breaker(self) -> None:
        if not self._is_broken:
            return
        elapsed = time.time() - self._broken_at
        if elapsed >= self.cooldown_seconds:
            logger.info(
                "Circuit breaker for %s auto-reset after %.1fs", self.name, elapsed
            )
            self._is_broken = False
            self._broken_at = 0.0
            self._breaker_reason = ""
            return
        remaining = self.cooldown_seconds - elapsed
        raise CircuitBreakerOpenError(
            f"Provider '{self.name}' circuit breaker open "
            f"({remaining:.1f}s remaining): {self._breaker_reason}"
        )

    def enforce_interval(self) -> None:
        """Enforce minimum interval between requests.

        Thread-safe slot reservation: each thread computes its sleep duration
        and advances last_request_time while holding the lock, then sleeps
        AFTER releasing the lock. Concurrent threads can reserve separate
        future time slots and sleep independently — no thread blocks others.
        """
        sleep_for = 0.0
        with self._lock:
            self._check_breaker()
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_for = self.min_interval - elapsed
                # Reserve the next time slot by advancing last_request_time
                self.last_request_time = now + sleep_for
            else:
                self.last_request_time = now
        # Sleep OUTSIDE the lock — other threads can reserve their own slots concurrently
        if sleep_for > 0:
            logger.debug("Pacing [%s]: sleeping %.2fs", self.name, sleep_for)
            time.sleep(sleep_for)

    def execute_with_retry(
        self,
        call_fn: Callable[[], Any],
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_BASE_DELAY,
        max_delay: float = RETRY_MAX_DELAY,
    ) -> Any:
        """Execute callable with rate limiting, retry, and circuit breaking.

        - Daily quota → trip breaker, raise DailyQuotaExhaustedError immediately
        - Non-retryable → fail fast, no retry
        - Server retry delay > 120s → fail fast
        - 429 / 5xx / timeout → exponential backoff, retry up to max_retries
        """
        for attempt in range(max_retries + 1):
            try:
                self.enforce_interval()
                return call_fn()

            except (DailyQuotaExhaustedError, CircuitBreakerOpenError):
                raise

            except Exception as e:
                if is_daily_quota_error(e):
                    msg = f"Daily quota exhausted for {self.name}: {e}"
                    self.trip_circuit_breaker(msg)
                    raise DailyQuotaExhaustedError(msg) from e

                if not is_retryable_error(e):
                    logger.error("Non-retryable error [%s], failing fast: %s", self.name, e)
                    raise

                if attempt >= max_retries:
                    logger.error(
                        "All %d attempts failed for [%s]: %s",
                        max_retries + 1, self.name, e
                    )
                    raise

                server_delay = extract_retry_delay(e)
                if server_delay is not None:
                    if server_delay > MAX_SERVER_RETRY_DELAY:
                        msg = (
                            f"Server requested {server_delay:.1f}s retry delay for "
                            f"[{self.name}], exceeds {MAX_SERVER_RETRY_DELAY:.0f}s cap"
                        )
                        logger.error(msg)
                        raise RuntimeError(msg) from e
                    delay = max(server_delay, 1.0)
                    logger.warning(
                        "Rate limit [%s] attempt %d/%d, server wait: %.1fs",
                        self.name, attempt + 1, max_retries + 1, delay
                    )
                else:
                    err_lower = str(e).lower()
                    is_429 = any(k in err_lower for k in (
                        "429", "resource_exhausted", "rate limit", "too many requests"
                    ))
                    if is_429:
                        backoff = min(base_delay * (2 ** attempt), max_delay)
                        delay = max(backoff, 15.0 * (attempt + 1)) + random.uniform(0.5, 2.0)
                        logger.warning(
                            "Rate limit [%s] attempt %d/%d, backoff %.1fs: %s",
                            self.name, attempt + 1, max_retries + 1, delay, e
                        )
                    else:
                        delay = min(base_delay * (attempt + 1), max_delay) + random.uniform(0.1, 0.5)
                        logger.warning(
                            "Transient error [%s] attempt %d/%d, retry in %.1fs: %s",
                            self.name, attempt + 1, max_retries + 1, delay, e
                        )

                time.sleep(delay)


# ============================================================================
# GLOBAL SINGLETONS
# ============================================================================

_LIMITERS: dict[str, ProviderRateLimiter] = {
    "gemini_generate": ProviderRateLimiter(
        name="gemini_generate",
        requests_per_minute=GEMINI_GENERATE_RPM,
        min_interval_override=GEMINI_REQUEST_INTERVAL,  # 4.0s between extraction calls
    ),
    "gemini_judge": ProviderRateLimiter(
        name="gemini_judge",
        requests_per_minute=GEMINI_JUDGE_RPM,
        min_interval_override=GEMINI_JUDGE_REQUEST_INTERVAL,  # 2.0s between judge calls
    ),
    "groq": ProviderRateLimiter(
        name="groq",
        requests_per_minute=GROQ_RPM,
        min_interval_override=GROQ_REQUEST_INTERVAL,  # 7.5s between calls (8K TPM safety)
    ),
}


def get_rate_limiter(provider: str) -> ProviderRateLimiter:
    """Get shared rate limiter singleton for a provider.

    Creates a default 30 RPM limiter if provider not registered.
    """
    if provider not in _LIMITERS:
        logger.warning("Unknown provider '%s', creating default 30 RPM limiter", provider)
        _LIMITERS[provider] = ProviderRateLimiter(provider, requests_per_minute=30)
    return _LIMITERS[provider]