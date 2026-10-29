/*
 * Wording for run state and arm values on the results view, kept pure so it can be tested
 * (tests/test_dashboard_labels.mjs). A run that did not finish every episode is "partial, not
 * final" only while the data is a snapshot; once numbers.json is final its numbers are final too,
 * and the run is "incomplete", labelled with where it stopped.
 */
'use strict';

(function (root) {
    const STOPS = {
        stopped_at_freeze: 'stopped at the freeze',
        stopped_by_quota: 'stopped by quota',
        waiting_quota: 'waiting for quota',
    };

    function isFinal(dataState) { return dataState === 'final'; }

    function counts(m) {
        return m.episodes_done != null && m.episodes_expected != null ? `${m.episodes_done}/${m.episodes_expected}` : '';
    }

    // Where an incomplete run stopped, without counts: "stopped at the freeze, error-dominated".
    function stopText(m, dataState) {
        if (m.complete) return '';
        const parts = [];
        if (isFinal(dataState)) {
            const state = m.hosted_state && m.hosted_state.state;
            parts.push(STOPS[state] || 'stopped early');
        } else {
            parts.push('still running, not final');
        }
        if (m.status === 'error-dominated') parts.push('error-dominated');
        return parts.join(', ');
    }

    // The full tag: "incomplete, stopped at the freeze (289/342)" or "partial, not final (251/342)".
    function runState(m, dataState) {
        if (m.complete) return '';
        const n = counts(m);
        const err = m.status === 'error-dominated' ? ', error-dominated' : '';
        if (isFinal(dataState)) {
            const state = m.hosted_state && m.hosted_state.state;
            return `incomplete, ${STOPS[state] || 'stopped early'}${n ? ` (${n})` : ''}${err}`;
        }
        return `partial, not final${n ? ` (${n})` : ''}${err}`;
    }

    function incompleteHeading(dataState) {
        return isFinal(dataState) ? 'Incomplete runs (not ranked against complete runs)'
                                  : 'Partial runs, not final and not ranked';
    }

    function incompleteDescription(dataState) {
        return isFinal(dataState)
            ? 'These hosted runs stopped at the data freeze or on the free-tier quota before every episode finished. ' +
              'Their numbers are final but rest on fewer episodes, so they are not ranked against the complete runs.'
            : 'Hosted runs still limited by the free-tier quota. Their rates are not final and must not be ranked ' +
              'against the complete runs.';
    }

    function pct(x) {
        if (x == null) return '';
        const v = Number(x);
        return (v === 0 || v === 100 ? v.toFixed(0) : v.toFixed(1)) + '%';
    }

    // One arm's value with what it measures: "no guardrails 5.7% (6/105)".
    function armValue(arm, metric) {
        const name = arm === 'off' ? 'no guardrails' : 'guardrails';
        if (!metric || metric.n == null) return `${name} no data`;
        return `${name} ${pct(metric.pct)} (${metric.k}/${metric.n})`;
    }

    // Both arms: "no guardrails 5.7% (6/105) → guardrails 0% (0/105)".
    function armPair(off, on) {
        return `${armValue('off', off)} → ${armValue('on', on)}`;
    }

    const api = { stopText, runState, incompleteHeading, incompleteDescription, armValue, armPair, pct };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.GuardrailsLabels = api;
})(typeof window !== 'undefined' ? window : globalThis);
