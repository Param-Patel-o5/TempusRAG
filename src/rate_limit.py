"""Shared rate limiter, retry helper, and circuit breaker per provider.

Coordinates API calls across modules for:
- gemini_generate (LLM Call 1, promise extraction)
- groq (LLM Call 2, cross-year judge)

Thread-safe and synchronous: no asyncio required, safe inside Streamlit.
"""

import logging
import random
import re
import threading
import time
from typing import Any, Callable

try:
    from src.config import CIRCUIT_BREAKER_COOLDOWN_SECONDS
except ImportError:
    CIRCUIT_BREAKER_COOLDOWN_SECONDS = 1800.0  # 30 minutes default

from src.config import (
    GEMINI_GENERATE_RPM,
    GROQ_RPM,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)

# Maximum server retry delay allowed in seconds. If exceeded, fail fast instead of sleeping.
MAX_SERVER_RETRY_DELAY = 120.0


class DailyQuotaExhaustedError(RuntimeError):
    """Raised when a provider's daily quota is exhausted (retrying cannot fix it)."""


class CircuitBreakerOpenError(RuntimeError):
    """Raised when a provider circuit breaker is open due to recent fatal errors."""


def get_error_status_code(error: Exception) -> int | None:
    """Extract HTTP status code from an exception if available."""
    for attr in ("status_code", "code", "http_status", "status"):
        val = getattr(error, attr, None)
        if isinstance(val, int):
            return val
        if callable(val):
            try:
                res = val()
                if isinstance(res, int):
                    return res
            except Exception:
                pass

    resp = getattr(error, "response", None)
    if resp is not None:
        val = getattr(resp, "status_code", None)
        if isinstance(val, int):
            return val

    return None


def is_daily_quota_error(error: Exception | str) -> bool:
    """Detect whether an error indicates a daily quota limit exhaustion.

    Strictly differentiates between daily quotas (fatal, breaker trips) and per-minute
    or per-second quotas (transient, retryable with backoff).
    """
    err_text = str(error).lower()

    has_daily_keywords = (
        "per day" in err_text
        or "per_day" in err_text
        or "perday" in err_text
        or "rpd" in err_text
        or "day limit" in err_text
        or "per calendar day" in err_text
        or "reset tomorrow" in err_text
        or "until tomorrow" in err_text
        or ("daily" in err_text and any(k in err_text for k in ("limit", "quota", "request", "exceeded", "exhausted")))
    )

    if not has_daily_keywords:
        return False

    # Guard against messages mentioning both per-minute and per-day in descriptive text
    if any(k in err_text for k in ("per minute", "per_minute", "perminute", "tpm", "rpm")):
        if "quota metric" in err_text:
            metric_part = err_text.split("quota metric")[-1]
            if any(m in metric_part for m in ("minute", "perminute", "per_minute")):
                return False
        if "requests per minute" in err_text and "requests per day" not in err_text:
            return False
        if "tokens per minute" in err_text and "requests per day" not in err_text:
            return False

    return True


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is transient and should be retried.

    Retry only on:
    - HTTP 429 (Rate limit / Too Many Requests)
    - HTTP 5xx (Server errors: 500, 502, 503, 504)
    - Timeouts and connection errors

    Fail fast on:
    - Daily quota exhaustion (trips circuit breaker)
    - HTTP 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 404 (Not Found)
    - Blocked prompts and safety filter violations
    - Generic RuntimeError and client programming errors
    """
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

    if isinstance(error, (TimeoutError, ConnectionError)):
        return True

    err_str = str(error)
    err_lower = err_str.lower()

    if any(k in err_lower for k in ("blocked", "safety", "content policy", "prompt was blocked", "finish_reason: safety")):
        return False

    # Check for explicit client status codes in message text
    for code in (400, 401, 403, 404):
        if f"{code} " in err_str or f"status {code}" in err_lower or f"code {code}" in err_lower:
            return False

    # Fail fast on generic RuntimeError without transient markers
    if type(error) is RuntimeError:
        if not any(k in err_lower for k in ("timeout", "timed out", "connection", "rate limit", "429")):
            return False

    # Retry on transient server/rate-limiting indicators
    if any(k in err_lower for k in (
        "429", "rate limit", "resource_exhausted", "too many requests",
        "500", "502", "503", "504", "server error", "bad gateway",
        "service unavailable", "gateway timeout",
        "timeout", "timed out", "connection error", "connection reset",
        "connection refused", "remotedisconnected"
    )):
        return True

    return False


def extract_retry_delay(error: Exception | str) -> float | None:
    """Extract server-recommended retry delay in seconds from an error message if present.

    Supports:
    - Protobuf format: retry_delay { seconds: 18 nanos: 500000000 } or retry_delay { seconds: 18 }
    - Key-value string format: retryDelay: '18s' or retryDelay: "18.5s" or retry_delay: 25
    - Natural language: "Please retry in 18.5s" or "retry after 30s" or "retry-after: 25"
    """
    err_text = str(error)

    # 1. Protobuf format: retry_delay { seconds: 18 (nanos: 123)? }
    pb_match = re.search(
        r"retry_delay\s*\{\s*seconds:\s*(\d+)(?:\s*nanos:\s*(\d+))?\s*\}",
        err_text,
        re.IGNORECASE,
    )
    if pb_match:
        try:
            sec = float(pb_match.group(1))
            if pb_match.group(2):
                sec += float(pb_match.group(2)) / 1e9
            return sec
        except (ValueError, TypeError):
            pass

    # 2. Key-value string format: retryDelay: '18s' or retryDelay: "18.5s" or retry_delay: 25
    kv_match = re.search(
        r"retry[_-]?delay\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)\s*s?['\"]?",
        err_text,
        re.IGNORECASE,
    )
    if kv_match:
        try:
            return float(kv_match.group(1))
        except (ValueError, TypeError):
            pass

    # 3. Natural language or header: retry in 18.5s / retry after 30s / retry-after: 30 / retry after: 30
    nl_match = re.search(
        r"retry[-_\s]*(?:in|after|delay)?[:\s]+['\"]?(\d+(?:\.\d+)?)['\"]?\s*s?",
        err_text,
        re.IGNORECASE,
    )
    if nl_match:
        try:
            return float(nl_match.group(1))
        except (ValueError, TypeError):
            pass

    return None


class ProviderRateLimiter:
    """Thread-safe rate limiter, circuit breaker, and retry coordinator for a single provider."""

    def __init__(
        self,
        name: str,
        requests_per_minute: int,
        cooldown_seconds: float = CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    ):
        self.name = name
        self.requests_per_minute = requests_per_minute
        self.cooldown_seconds = cooldown_seconds
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.last_request_time = 0.0
        self._lock = threading.Lock()

        # Circuit breaker state
        self._is_broken = False
        self._broken_at = 0.0
        self._breaker_reason: str = ""

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._is_broken = False
            self._broken_at = 0.0
            self._breaker_reason = ""

    def trip_circuit_breaker(self, reason: str) -> None:
        """Trip circuit breaker so subsequent calls fail immediately.

        Logs the full error text when tripped.
        """
        with self._lock:
            self._is_broken = True
            self._broken_at = time.time()
            self._breaker_reason = reason
        logger.error(f"Circuit breaker tripped for {self.name}. Full error: {reason}")

    def _check_breaker(self) -> None:
        """Check circuit breaker state and auto-reset if cooldown elapsed."""
        if not self._is_broken:
            return

        elapsed = time.time() - self._broken_at
        if elapsed >= self.cooldown_seconds:
            logger.info(
                f"Circuit breaker for {self.name} auto-reset after {elapsed:.1f}s "
                f"(cooldown threshold: {self.cooldown_seconds}s)"
            )
            self._is_broken = False
            self._broken_at = 0.0
            self._breaker_reason = ""
            return

        remaining = self.cooldown_seconds - elapsed
        raise CircuitBreakerOpenError(
            f"Provider '{self.name}' circuit breaker is open (cooling down for {remaining:.1f}s more): "
            f"{self._breaker_reason}"
        )

    def enforce_interval(self) -> None:
        """Enforce minimum time interval between consecutive requests."""
        with self._lock:
            self._check_breaker()

            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval:
                sleep_sec = self.min_interval - time_since_last
                logger.debug(f"Rate limiting [{self.name}]: sleeping {sleep_sec:.2f}s")
                time.sleep(sleep_sec)

            self.last_request_time = time.time()

    def execute_with_retry(
        self,
        call_fn: Callable[[], Any],
        max_retries: int = MAX_RETRIES,
        base_delay: float = RETRY_BASE_DELAY,
        max_delay: float = RETRY_MAX_DELAY,
    ) -> Any:
        """Execute a callable with rate limiting, server retry delays, and circuit breaking.

        Args:
            call_fn: Zero-argument callable making the network request.
            max_retries: Maximum retry attempts for transient errors.
            base_delay: Initial exponential backoff delay in seconds.
            max_delay: Maximum exponential backoff delay in seconds.

        Returns:
            The return value of call_fn().

        Raises:
            DailyQuotaExhaustedError: If daily quota is hit (trips circuit breaker).
            CircuitBreakerOpenError: If the circuit breaker is currently open.
            RuntimeError: If server retry delay exceeds 120s or non-retryable error occurs.
            Exception: If all retries are exhausted.
        """
        for attempt in range(max_retries + 1):
            try:
                self.enforce_interval()
                return call_fn()

            except (DailyQuotaExhaustedError, CircuitBreakerOpenError):
                raise

            except Exception as e:
                # 1. Daily quota exhausted -> trip breaker immediately, never retry
                if is_daily_quota_error(e):
                    msg = f"Daily quota exhausted for {self.name}: {e}"
                    self.trip_circuit_breaker(msg)
                    raise DailyQuotaExhaustedError(msg) from e

                # 2. Non-retryable error -> fail fast immediately
                if not is_retryable_error(e):
                    logger.error(f"Non-retryable error for [{self.name}], failing fast: {e}")
                    raise

                if attempt >= max_retries:
                    logger.error(f"All {max_retries + 1} attempts failed for [{self.name}]: {e}")
                    raise

                # 3. Check server-requested delay and cap at 120s
                server_delay = extract_retry_delay(e)
                if server_delay is not None:
                    if server_delay > MAX_SERVER_RETRY_DELAY:
                        msg = (
                            f"Server requested retry delay of {server_delay:.1f}s for [{self.name}], "
                            f"which exceeds maximum allowable limit of {MAX_SERVER_RETRY_DELAY:.0f}s"
                        )
                        logger.error(msg)
                        raise RuntimeError(msg) from e

                    delay = max(server_delay, 1.0)
                    logger.warning(
                        f"Rate limit for [{self.name}] on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Server requested wait: {delay:.1f}s"
                    )
                else:
                    err_str = str(e).lower()
                    is_429 = "429" in err_str or "resource_exhausted" in err_str or "rate limit" in err_str
                    if is_429:
                        # Allow provider per-minute quota window (60s) to reset between retries
                        backoff = min(base_delay * (2 ** attempt), max_delay)
                        delay = max(backoff, 15.0 * (attempt + 1)) + random.uniform(0.5, 2.0)
                        logger.warning(
                            f"Rate limit hit for [{self.name}] on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Backing off {delay:.1f}s to allow quota window reset: {e}"
                        )
                    else:
                        delay = min(base_delay * (attempt + 1), max_delay) + random.uniform(0.1, 0.5)
                        logger.warning(
                            f"Transient error for [{self.name}] on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {delay:.1f}s: {e}"
                        )

                time.sleep(delay)


# Global singleton limiters per provider
_LIMITERS: dict[str, ProviderRateLimiter] = {
    "gemini_generate": ProviderRateLimiter("gemini_generate", GEMINI_GENERATE_RPM),
}


def get_rate_limiter(provider: str) -> ProviderRateLimiter:
    """Retrieve the shared rate limiter singleton for a provider.

    Args:
        provider: Provider identifier string (e.g. 'gemini_generate').

    Returns:
        ProviderRateLimiter instance.
    """
    if provider not in _LIMITERS:
        _LIMITERS[provider] = ProviderRateLimiter(provider, 30)
    return _LIMITERS[provider]


def _run_tests() -> None:
    print("--- 1. Testing is_daily_quota_error (strict per-day vs per-minute) ---")
    # Per-minute cases (must NOT be treated as daily quota)
    gemini_rpm = "ResourceExhausted: 429 Quota exceeded for quota metric 'GenerateContentRequestsPerMinutePerProject' and limit 'GenerateContentRequestsPerMinutePerProject' of service 'generativelanguage.googleapis.com'"
    assert not is_daily_quota_error(gemini_rpm), "Failed: Gemini RPM flagged as daily"

    groq_tpm = "Rate limit reached for model 'llama-3.1-8b-instant' on tokens per minute (TPM): Limit 6000, Used 5800, Requested 500. Please try again in 12s."
    assert not is_daily_quota_error(groq_tpm), "Failed: Groq TPM flagged as daily"

    groq_rpm = "Rate limit reached for model 'llama-3.1-8b-instant' on requests per minute (RPM): Limit 30, Used 30. Please try again in 2s."
    assert not is_daily_quota_error(groq_rpm), "Failed: Groq RPM flagged as daily"

    generic_rpm = "Rate limit exceeded: 15 requests per minute. Try again in 4s."
    assert not is_daily_quota_error(generic_rpm), "Failed: Generic RPM flagged as daily"

    # Per-day cases (must be detected as daily quota)
    gemini_rpd = "ResourceExhausted: 429 Quota exceeded for quota metric 'GenerateContentRequestsPerDayPerProject' and limit 'GenerateContentRequestsPerDayPerProject' of service 'generativelanguage.googleapis.com'"
    assert is_daily_quota_error(gemini_rpd), "Failed: Gemini RPD not flagged as daily"

    groq_rpd = "Rate limit reached for model 'llama-3.1-8b-instant' on requests per day (RPD): Limit 14400, Used 14400. Please try again in 8h20m."
    assert is_daily_quota_error(groq_rpd), "Failed: Groq RPD not flagged as daily"

    generic_rpd = "Free tier daily request limit reached for today. Resets at 00:00 UTC."
    assert is_daily_quota_error(generic_rpd), "Failed: Generic RPD not flagged as daily"

    print("OK: Strict daily quota error detection verified")

    print("\n--- 2. Testing extract_retry_delay parser variations ---")
    d1 = extract_retry_delay("ResourceExhausted: retry_delay { seconds: 18 }")
    assert d1 == 18.0, f"Expected 18.0, got {d1}"

    d2 = extract_retry_delay("ResourceExhausted: retry_delay { seconds: 18 nanos: 500000000 }")
    assert d2 == 18.5, f"Expected 18.5, got {d2}"

    d3 = extract_retry_delay("Rate limit reached. retryDelay: '18s'")
    assert d3 == 18.0, f"Expected 18.0, got {d3}"

    d4 = extract_retry_delay('Rate limit reached. retryDelay: "22.5s"')
    assert d4 == 22.5, f"Expected 22.5, got {d4}"

    d5 = extract_retry_delay("Please retry in 18.5s.")
    assert d5 == 18.5, f"Expected 18.5, got {d5}"

    d6 = extract_retry_delay("retry-after: 30")
    assert d6 == 30.0, f"Expected 30.0, got {d6}"

    print("OK: All retry delay formats parsed correctly")

    print("\n--- 3. Testing retry delay cap (> 120s raises) ---")
    limiter = ProviderRateLimiter("test_cap", requests_per_minute=600)

    class MockExcessiveDelayError(Exception):
        pass

    def call_excessive():
        raise MockExcessiveDelayError("Rate limit exceeded. Please retry in 150s.")

    try:
        limiter.execute_with_retry(call_excessive, max_retries=1)
        assert False, "Failed: Did not raise on delay > 120s"
    except RuntimeError as re_err:
        assert "exceeds maximum allowable limit" in str(re_err)
        print("OK: Delay > 120s raised RuntimeError immediately")

    print("\n--- 4. Testing fail-fast on 400, 401, 403, 404, blocked prompts, and RuntimeError ---")
    # Test 400 with status_code attribute
    class MockHttpError(Exception):
        def __init__(self, code: int, msg: str):
            super().__init__(msg)
            self.status_code = code

    call_count = 0
    def call_bad_request():
        nonlocal call_count
        call_count += 1
        raise MockHttpError(400, "Bad Request: Invalid argument")

    try:
        limiter.execute_with_retry(call_bad_request, max_retries=2)
    except MockHttpError:
        assert call_count == 1, f"Expected 1 call (fail fast), got {call_count}"
        print("OK: HTTP 400 failed fast without retries")

    # Test 403 with code attribute
    class MockGoogleError(Exception):
        def __init__(self, code: int, msg: str):
            super().__init__(msg)
            self.code = code

    call_count = 0
    def call_forbidden():
        nonlocal call_count
        call_count += 1
        raise MockGoogleError(403, "Permission Denied")

    try:
        limiter.execute_with_retry(call_forbidden, max_retries=2)
    except MockGoogleError:
        assert call_count == 1, f"Expected 1 call (fail fast), got {call_count}"
        print("OK: HTTP 403 failed fast without retries")

    # Test Blocked Prompt
    call_count = 0
    def call_blocked():
        nonlocal call_count
        call_count += 1
        raise Exception("Candidate was blocked due to SAFETY policy")

    try:
        limiter.execute_with_retry(call_blocked, max_retries=2)
    except Exception:
        assert call_count == 1, f"Expected 1 call (fail fast), got {call_count}"
        print("OK: Blocked safety prompt failed fast without retries")

    # Test plain RuntimeError
    call_count = 0
    def call_runtime():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Non-transient internal error")

    try:
        limiter.execute_with_retry(call_runtime, max_retries=2)
    except RuntimeError:
        assert call_count == 1, f"Expected 1 call (fail fast), got {call_count}"
        print("OK: Plain RuntimeError failed fast without retries")

    print("\n--- 5. Testing retry behavior on 503 Service Unavailable ---")
    call_count = 0
    def call_503():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise MockHttpError(503, "Service Unavailable")
        return "success"

    res = limiter.execute_with_retry(call_503, max_retries=3, base_delay=0.01)
    assert res == "success"
    assert call_count == 3
    print("OK: HTTP 503 retried and succeeded on attempt 3")

    print("\n--- 6. Testing circuit breaker trip, logging, and auto-reset after cooldown ---")
    cooldown_limiter = ProviderRateLimiter("test_cooldown", requests_per_minute=600, cooldown_seconds=0.05)

    def call_daily_limit():
        raise RuntimeError("Quota exceeded: PerDay limit reached")

    try:
        cooldown_limiter.execute_with_retry(call_daily_limit, max_retries=1)
    except DailyQuotaExhaustedError:
        print("OK: Daily quota raised DailyQuotaExhaustedError")

    # Verify breaker is open immediately
    try:
        cooldown_limiter.enforce_interval()
        assert False, "Failed: Breaker should be open"
    except CircuitBreakerOpenError as cb_err:
        assert "cooling down" in str(cb_err)
        print("OK: CircuitBreakerOpenError blocked immediate subsequent call")

    # Wait for cooldown to expire
    time.sleep(0.06)
    # Breaker must auto-reset on next enforce_interval call
    cooldown_limiter.enforce_interval()
    assert cooldown_limiter._is_broken is False
    print("OK: Circuit breaker auto-reset after cooldown expired")

    print("\nAll rate_limit.py tests passed successfully!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    print("=== Testing src/rate_limit.py ===")
    _run_tests()
