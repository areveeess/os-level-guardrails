// Results-view wording (src/dashboard/static/labels.js). Run: node --test tests/test_dashboard_labels.mjs
// (tests/test_dashboard_labels.py runs it under pytest when node is installed).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const L = require('../src/dashboard/static/labels.js');

const at_freeze = { model: 'groq-gpt-oss-120b', complete: false, status: 'partial', episodes_done: 289,
                    episodes_expected: 342, hosted_state: { state: 'stopped_at_freeze' } };
const error_dominated = { model: 'groq-gpt-oss-20b', complete: false, status: 'error-dominated', episodes_done: 132,
                          episodes_expected: 342, hosted_state: { state: 'stopped_at_freeze' } };
const by_quota = { model: 'groq-qwen3.8-27b', complete: false, status: 'partial', episodes_done: 322,
                   episodes_expected: 342, hosted_state: { state: 'stopped_by_quota' } };
const complete = { model: 'scratch-gpt', complete: true, status: 'complete', episodes_done: 342, episodes_expected: 342 };

test('final data: incomplete runs say where they stopped, never "not final"', () => {
    assert.equal(L.runState(at_freeze, 'final'), 'incomplete, stopped at the freeze (289/342)');
    assert.equal(L.runState(by_quota, 'final'), 'incomplete, stopped by quota (322/342)');
    assert.equal(L.runState(error_dominated, 'final'), 'incomplete, stopped at the freeze (132/342), error-dominated');
    for (const m of [at_freeze, by_quota, error_dominated]) {
        assert.doesNotMatch(L.runState(m, 'final'), /not final|partial/);
        assert.doesNotMatch(L.stopText(m, 'final'), /not final/);
    }
    assert.equal(L.stopText(error_dominated, 'final'), 'stopped at the freeze, error-dominated');
});

test('final data: heading and description', () => {
    assert.equal(L.incompleteHeading('final'), 'Incomplete runs (not ranked against complete runs)');
    const desc = L.incompleteDescription('final');
    assert.match(desc, /stopped at the data freeze or on the free-tier quota/);
    assert.match(desc, /numbers are final/);
    assert.doesNotMatch(desc, /not final/);
});

test('snapshot data keeps the "not final" wording', () => {
    assert.equal(L.runState(at_freeze, 'snapshot'), 'partial, not final (289/342)');
    assert.equal(L.incompleteHeading('snapshot'), 'Partial runs, not final and not ranked');
    assert.match(L.incompleteDescription('snapshot'), /not final/);
    assert.equal(L.stopText(by_quota, 'snapshot'), 'still running, not final');
});

test('an unknown stop state falls back, a complete run has no tag', () => {
    assert.equal(L.runState({ ...at_freeze, hosted_state: null }, 'final'), 'incomplete, stopped early (289/342)');
    assert.equal(L.runState(complete, 'final'), '');
    assert.equal(L.stopText(complete, 'final'), '');
});

test('arm values always name the arm and show k/n, including the protected 0/n', () => {
    const off = { pct: 5.7, k: 6, n: 105 }, on = { pct: 0, k: 0, n: 105 };
    assert.equal(L.armValue('off', off), 'no guardrails 5.7% (6/105)');
    assert.equal(L.armValue('on', on), 'guardrails 0% (0/105)');
    assert.equal(L.armPair(off, on), 'no guardrails 5.7% (6/105) → guardrails 0% (0/105)');
    assert.equal(L.armValue('off', { pct: 100, k: 120, n: 120 }), 'no guardrails 100% (120/120)');
    assert.equal(L.armValue('on', null), 'guardrails no data');
});
