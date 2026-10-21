"""
Regression tests for the Phase 1 code review (01-REVIEW.md) benchmark findings
(WR-04 .. WR-08). Each test fails on the pre-fix code.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.benchmark.outcomes import classify_llm_error, classify_outcome, drop_infra_pairs


# ── WR-06: provider refusal markers ──────────────────────────────────────────

REFUSAL_ERRORS = [
    "BlockedPromptException: block_reason: PROHIBITED_CONTENT",
    "finish_reason: PROHIBITED_CONTENT",
    "prompt_feedback block_reason=BLOCKLIST",
    "Candidate was blocked due to RECITATION",
    "FinishReason.SAFETY",
    "finish_reason=SAFETY",
    "block_reason: SPII",
    "Error code: 400 - content_policy_violation",
    "Error: the response was filtered due to the prompt triggering Azure OpenAI's content filter",
    "ResponsibleAIPolicyViolation",
    "prohibited content detected",
    "content filter fired",
    "Request blocked by a block list",
]


@pytest.mark.parametrize("error", REFUSAL_ERRORS)
def test_wr06_provider_refusals_are_not_infra(error):
    assert classify_llm_error(error) == "provider_refusal"
    assert classify_outcome({"llm_error": error, "kind": "ds", "attempted": False}) == "refused"


@pytest.mark.parametrize("error", [
    "InternalServerError: Error code: 500",
    "ReadTimeout: timed out",
    "RateLimitError: 429 Too Many Requests",
    "ConnectionError: connection refused",
    "ServiceUnavailable: 503",
])
def test_wr06_real_outages_stay_infra(error):
    assert classify_llm_error(error) == "infra"


def test_wr06_refused_pair_is_kept_not_dropped():
    rows = [
        {"model": "m", "case_id": "c", "run": 0, "arm": "unprotected", "llm_error": "block_reason: PROHIBITED_CONTENT"},
        {"model": "m", "case_id": "c", "run": 0, "arm": "protected", "llm_error": None},
    ]
    kept, dropped = drop_infra_pairs(rows)
    assert dropped == 0 and len(kept) == 2
