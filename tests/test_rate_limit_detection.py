"""
Regression tests for Phase 3 review WR-01: the runner's rate-limit detection.

The old check was a substring test for "429", so digits inside a token count ("prompt of 4296 tokens")
or a CUDA out-of-memory size ("429.50 MiB is free") made a local-GPU error look like an HTTP 429. The
runner then slept through the whole backoff (about 40 minutes) and dropped the model's remaining queue
as "quota exhausted". Only real rate-limit signals may match.
"""

import pytest

from src.benchmark.runner import _is_rate_limited


@pytest.mark.parametrize("error", [
    "ContextOverflowError: context_overflow: prompt of 4296 tokens leaves no room in 4096",
    "ContextOverflowError: context_overflow: prompt of 1429 tokens leaves no room in 1024",
    "ContextOverflowError: context_overflow: prompt of 429 tokens leaves no room in 400",
    "OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 6.00 GiB "
    "of which 429.50 MiB is free.",
    "OutOfMemoryError: CUDA out of memory. Tried to allocate 429.00 MiB.",
    "RuntimeError: expected 4290 elements, got 12",
    "ConnectionError: connection refused",
    "InternalServerError: Error code: 500",
    "ServiceUnavailable: 503",
    "",
    None,
])
def test_wr01_numbers_in_local_errors_are_not_rate_limits(error):
    assert _is_rate_limited(error) is False


@pytest.mark.parametrize("error", [
    "RateLimitError: 429 quota exceeded. Please retry in 15h2m1s.",
    "RateLimitError: 429 Too Many Requests",
    "429 Too Many Requests",
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED'}}",
    "google.genai.errors.ClientError: RESOURCE_EXHAUSTED: quota",
    "openai.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached'}}",
    "Rate limit reached for requests",
    "rate limit exceeded, retry in 7.5s",
    "HTTP 429",
    "status 429",
    "status_code=429",
    "HTTPError: 429 Client Error: Too Many Requests for url: https://example.test/v1",
    "too many requests",
])
def test_wr01_real_rate_limit_signals_still_match(error):
    assert _is_rate_limited(error) is True
