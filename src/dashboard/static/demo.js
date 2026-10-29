/*
 * Demo page: replay of a recorded episode (both arms, step by step) and the scripted policy
 * simulation. Every string from the server is inserted with textContent; highlights are built
 * from text nodes, never from HTML strings.
 */
'use strict';

(function () {
    const $ = id => document.getElementById(id);
    const STEP_MS = 1400;
    const CANARY = /CANARY[0-9A-F]+DATA/g;
    const LAYER = { tool_broker: 'Tool broker', policy_engine: 'Policy engine', sandbox: 'Namespaces', seccomp: 'seccomp-BPF' };
    const GROUPS = ['locally made', 'local pretrained', 'hosted', 'control'];
    const GROUP_LABEL = { 'locally made': 'Locally made', 'local pretrained': 'Local, pretrained', hosted: 'Hosted (Groq)', control: 'Scripted control' };

    const state = { stories: [], story: -1, lineup: {}, episode: null, shown: 0, total: 0, timer: null, cases: [] };

    function el(tag, cls, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined && text !== null) n.textContent = text;
        return n;
    }
    function icon(name) { const i = el('i', 'ti ' + name); i.setAttribute('aria-hidden', 'true'); return i; }
    async function getJSON(url, opts) {
        const r = await fetch(url, Object.assign({ cache: 'no-store' }, opts || {}));
        const body = await r.json().catch(() => ({}));
        if (!r.ok) { const e = new Error(body.error || String(r.status)); e.status = r.status; e.body = body; throw e; }
        return body;
    }

    // Text with the canary token(s) marked, built from text nodes.
    function withCanary(text) {
        const frag = document.createDocumentFragment();
        let last = 0;
        String(text).replace(CANARY, (m, at) => {
            frag.append(document.createTextNode(text.slice(last, at)));
            const mark = el('mark', 'canary', m);
            mark.title = 'Canary: planted test data the benchmark seeds as the private record. Seeing it here means the data left the agent.';
            frag.append(mark);
            last = at + m.length;
            return m;
        });
        frag.append(document.createTextNode(String(text).slice(last)));
        return frag;
    }
    // The tool response with the attacker's instruction marked.
    function withInjection(text, instruction) {
        const frag = document.createDocumentFragment();
        if (!instruction) { frag.append(document.createTextNode(text)); return frag; }
        const variants = [instruction, instruction.replace(/'/g, "\\'")];
        let at = -1, hit = '';
        for (const v of variants) { at = text.indexOf(v); if (at >= 0) { hit = v; break; } }
        if (at < 0) { frag.append(document.createTextNode(text)); return frag; }
        frag.append(document.createTextNode(text.slice(0, at)));
        const mark = el('mark', 'inject', hit);
        mark.title = 'The injected instruction: written by the attacker, hidden inside data the agent reads';
        frag.append(mark, document.createTextNode(text.slice(at + hit.length)));
        return frag;
    }
    function argsText(a) {
        if (a.action === 'call_tool') return JSON.stringify(a.args || {});
        return a.command || a.path || a.url || a.content || '';
    }

    // ── Building one arm's steps ──

    function stepsFor(arm, c) {
        const rec = state.episode.arms[arm];
        if (!rec) return [];
        const calls = (rec.tool_calls || []).slice();
        const checks = rec.policy_checks || [];
        const attacker = new Set(c.attacker_tools || []);
        if (c.kind === 'ds') attacker.add('GmailSendEmail');
        return (rec.actions || []).map((a, i) => {
            const s = { kind: a.action, what: '', detail: '', decision: null, layer: null, reason: '', attacker: false };
            if (a.action === 'call_tool') {
                const call = calls.shift() || {};
                s.what = a.tool || call.tool || 'tool';
                s.detail = argsText(a);
                s.attacker = attacker.has(s.what);
                if (arm === 'unprotected') { s.decision = 'RAN'; s.reason = 'Called directly; nothing checks it'; }
                else if (call.allowed === false) { s.decision = 'BLOCK'; s.layer = 'tool_broker'; s.reason = "Not in this task's capability list, so it never runs"; }
                else { s.decision = 'ALLOW'; s.layer = 'tool_broker'; s.reason = "In this task's capability list"; }
            } else if (a.action === 'respond') {
                s.what = 'Final answer';
                s.detail = a.message || rec.final_message || '';
                s.decision = 'END';
            } else {
                s.what = a.action;
                s.detail = argsText(a);
                if (arm === 'unprotected') { s.decision = 'RAN'; s.reason = 'Ran directly on the host; nothing checks it'; }
                else if (checks[i]) {
                    s.decision = checks[i].decision; s.layer = 'policy_engine'; s.reason = checks[i].reason;
                } else {
                    s.decision = 'CHECKED'; s.layer = 'policy_engine';
                    s.reason = 'Checked by the policy engine; the run kept only per-episode counts for it';
                }
            }
            return s;
        });
    }

    function stepNode(s, n) {
        const li = el('li', 'step');
        li.dataset.decision = s.decision || '';
        const head = el('div', 'step-head');
        head.append(el('span', 'step-n', String(n)));
        const what = el('span', 'step-what', s.what);
        head.append(what);
        if (s.attacker) {
            const tag = el('span', 'badge', "attacker's goal");
            tag.dataset.tone = 'warn';
            head.append(tag);
        }
        li.append(head);
        if (s.detail) {
            const d = el('code', 'step-detail');
            d.append(withCanary(s.detail));
            li.append(d);
        }
        if (s.decision && s.decision !== 'END') {
            const dec = el('div', 'step-dec');
            const map = {
                RAN: ['ti-player-play', 'Ran', 'ran'],
                ALLOW: ['ti-circle-check', 'Allowed', 'ALLOW'],
                FLAG: ['ti-alert-triangle', 'Flagged', 'FLAG'],
                BLOCK: ['ti-ban', 'Blocked', 'BLOCK'],
                CHECKED: ['ti-shield-check', 'Checked', 'checked'],
            }[s.decision] || ['ti-point', s.decision, ''];
            const b = el('span', 'badge');
            b.dataset.d = map[2];
            b.append(icon(map[0]), document.createTextNode(map[1] + (s.layer ? ' by ' + LAYER[s.layer] : '')));
            dec.append(b, el('span', 'step-why', s.reason));
            li.append(dec);
        }
        return li;
    }

    function outcome(arm) {
        const rec = state.episode.arms[arm];
        const c = state.episode.case;
        if (!rec) return { tone: '', text: 'No record for this arm' };
        if (rec.llm_error) return { tone: '', text: 'Ended in an infrastructure error, not an attack outcome' };
        if (c.kind === 'benign') {
            if (rec.false_positive) return { tone: 'bad', text: 'False positive: the guardrails blocked a normal task' };
            return rec.user_task_done ? { tone: 'good', text: 'The user task was done' } : { tone: '', text: 'The user task was not done' };
        }
        if (rec.succeeded) return { tone: 'bad', text: 'Attack succeeded' + (rec.canary_exposed ? ': the canary data left the agent' : '') };
        if (rec.attack_blocked) {
            const layers = Object.keys(rec.blocked_by || {}).filter(k => rec.blocked_by[k]).map(k => LAYER[k] || k);
            return { tone: 'good', text: 'Attack blocked' + (layers.length ? ' by ' + layers.join(' and ') : '') };
        }
        if (rec.attempted) return { tone: '', text: 'The attack was attempted but did not complete' };
        return { tone: '', text: 'The model did not follow the injection' };
    }

    // ── Player ──

    function renderPlayer() {
        const c = state.episode.case;
        $('case').hidden = false;
        $('player').hidden = false;
        $('case-prompt').textContent = c.prompt;
        $('case-tool').textContent = c.user_tool;
        const pre = $('case-response');
        pre.replaceChildren(withInjection(c.tool_response, c.attacker_instruction));
        $('case-note').textContent = c.kind === 'benign'
            ? 'A benign case: there is no injection in this tool result.'
            : `The highlighted text is the injection (${c.attack_type}). The attacker wants the agent to call ${c.attacker_tools.join(', ')}${c.kind === 'ds' ? ' and email the result out' : ''}.`;
        state.arms = { unprotected: stepsFor('unprotected', c), protected: stepsFor('protected', c) };
        state.total = Math.max(state.arms.unprotected.length, state.arms.protected.length);
        state.shown = 0;
        paint();
    }

    function paint() {
        for (const arm of ['unprotected', 'protected']) {
            const list = $('steps-' + arm);
            const steps = state.arms[arm];
            list.replaceChildren(...steps.slice(0, state.shown).map((s, i) => stepNode(s, i + 1)));
            const last = list.lastElementChild;
            if (last) last.classList.add('is-new');
            const out = $('out-' + arm);
            out.hidden = state.shown < state.total;
            if (!out.hidden) {
                const o = outcome(arm);
                out.dataset.tone = o.tone;
                out.replaceChildren(icon(o.tone === 'bad' ? 'ti-alert-octagon' : o.tone === 'good' ? 'ti-shield-check' : 'ti-info-circle'),
                                    document.createTextNode(o.text));
            }
        }
        $('player-pos').textContent = state.shown >= state.total
            ? `All ${state.total} steps shown`
            : `Step ${state.shown} of ${state.total}`;
        $('btn-step').disabled = state.shown >= state.total;
        if (state.shown >= state.total) stop();
    }
    function step() { if (state.episode && state.shown < state.total) { state.shown++; paint(); } }
    function reset() { stop(); state.shown = 0; if (state.episode) paint(); }
    function play() {
        if (state.timer) { stop(); return; }
        if (state.shown >= state.total) state.shown = 0;
        $('btn-play').querySelector('span').textContent = 'Pause';
        $('btn-play').querySelector('i').className = 'ti ti-player-pause';
        step();
        state.timer = setInterval(step, STEP_MS);
    }
    function stop() {
        clearInterval(state.timer);
        state.timer = null;
        $('btn-play').querySelector('span').textContent = 'Play';
        $('btn-play').querySelector('i').className = 'ti ti-player-play';
    }

    // ── Choosing an episode ──

    async function loadEpisode(model, caseId, run) {
        stop();
        try {
            state.episode = await getJSON(`/api/demo/episode?model=${encodeURIComponent(model)}&case=${encodeURIComponent(caseId)}&run=${run}`);
            renderPlayer();
        } catch (e) {
            $('player').hidden = true;
            $('case').hidden = true;
            $('replay-load').hidden = false;
            $('replay-load').textContent = 'That episode could not be loaded (' + e.message + ').';
            return;
        }
        $('replay-load').hidden = true;
    }

    async function loadCases(model, select) {
        const data = await getJSON(`/api/demo/cases?model=${encodeURIComponent(model)}`);
        state.cases = data.cases;
        const sel = $('pick-case');
        sel.replaceChildren(...data.cases.map(c => {
            const o = el('option', '', c.case_id + (c.attack_type ? ` (${c.attack_type})` : ' (benign task)'));
            o.value = c.case_id;
            return o;
        }));
        if (select) sel.value = select;
        fillRuns();
    }
    function fillRuns(select) {
        const c = state.cases.find(x => x.case_id === $('pick-case').value);
        const runs = c ? c.runs : [0];
        $('pick-run').replaceChildren(...runs.map(r => { const o = el('option', '', String(r)); o.value = r; return o; }));
        $('pick-run').value = String(select || runs[0] || 0);
    }

    async function chooseStory(i) {
        const s = state.stories[i];
        if (!s) return;
        state.story = i;
        document.querySelectorAll('.story').forEach((n, j) => n.setAttribute('aria-pressed', j === i ? 'true' : 'false'));
        $('pick-model').value = s.model;
        await loadCases(s.model, s.case_id);
        fillRuns(s.run);
        await loadEpisode(s.model, s.case_id, s.run);
    }

    function storyCards() {
        const box = $('stories');
        box.replaceChildren(...state.stories.map((s, i) => {
            const b = el('button', 'story');
            b.type = 'button';
            b.setAttribute('role', 'listitem');
            b.setAttribute('aria-pressed', 'false');
            b.append(el('span', 'story-n', String(i + 1)), el('span', 'story-t', s.title), el('span', 'story-x', s.text),
                     el('span', 'story-m', s.model + ', ' + s.case_id));
            b.addEventListener('click', () => { b.blur(); chooseStory(i); });
            return b;
        }));
    }

    async function initReplay() {
        let index, lineup;
        try {
            [index, lineup] = await Promise.all([getJSON('/api/demo/replay'), getJSON('/api/lineup')]);
        } catch (e) {
            $('replay-load').textContent = e.status === 404
                ? 'No frozen runs yet: results/phase5-final/ was not found. Replay needs the frozen benchmark data.'
                : 'The frozen runs could not be read.';
            return;
        }
        for (const m of lineup.models) state.lineup[m.model] = m;
        const sel = $('pick-model');
        for (const g of GROUPS) {
            const models = index.models.filter(m => (state.lineup[m] || {}).group === g);
            if (!models.length) continue;
            const og = document.createElement('optgroup');
            og.label = GROUP_LABEL[g];
            models.forEach(m => {
                const o = el('option', '', m + ((state.lineup[m] || {}).status && state.lineup[m].status !== 'complete' ? ' (partial)' : ''));
                o.value = m;
                og.append(o);
            });
            sel.append(og);
        }
        state.stories = index.stories;
        storyCards();
        $('replay-load').textContent = 'Pick a story above, or a model and case.';
        sel.addEventListener('change', async () => { await loadCases(sel.value); loadEpisode(sel.value, $('pick-case').value, $('pick-run').value); });
        $('pick-case').addEventListener('change', () => { fillRuns(); loadEpisode(sel.value, $('pick-case').value, $('pick-run').value); });
        $('pick-run').addEventListener('change', () => loadEpisode(sel.value, $('pick-case').value, $('pick-run').value));
        if (state.stories.length) await chooseStory(0);
    }

    // ── Simulation ──

    let simTimer = null;
    function simStatus(name, text) { $('sim-status').dataset.state = name; $('sim-status-text').textContent = text; }
    function renderSim(st) {
        const labels = { idle: 'Idle', running: `Running, ${st.done} of ${st.total}`, done: `Done, ${st.done} of ${st.total}`,
                         timeout: `Stopped at its time limit after ${st.done} of ${st.total}`, error: 'Failed: ' + (st.error || 'error') };
        simStatus(st.status === 'running' || st.status === 'done' ? 'up' : st.status === 'idle' ? 'wait' : 'down', labels[st.status] || st.status);
        $('sim-list').replaceChildren(...(st.decisions || []).map((d, i, all) => {
            const li = el('li', 'sim-row' + (i === all.length - 1 && st.status === 'running' ? ' is-new' : ''));
            const b = el('span', 'badge');
            b.dataset.d = d.decision;
            b.append(icon({ ALLOW: 'ti-circle-check', FLAG: 'ti-alert-triangle', BLOCK: 'ti-ban' }[d.decision] || 'ti-point'),
                     document.createTextNode({ ALLOW: 'Allowed', FLAG: 'Flagged', BLOCK: 'Blocked' }[d.decision] || d.decision));
            const body = el('div', 'sim-body');
            body.append(el('span', 'sim-desc', d.description), el('code', 'sim-target', d.action_type + ': ' + d.target),
                        el('span', 'sim-why', d.reason));
            li.append(b, body);
            return li;
        }));
        $('btn-sim').disabled = st.status === 'running';
    }
    async function pollSim() {
        try {
            const st = await getJSON('/api/demo/simulate');
            renderSim(st);
            if (st.status === 'running') simTimer = setTimeout(pollSim, 400);
        } catch (e) { simStatus('down', 'Disconnected'); }
    }
    async function runSim() {
        clearTimeout(simTimer);
        try {
            await getJSON('/api/demo/simulate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                                  body: JSON.stringify({ scenario: $('sim-scenario').value }) });
        } catch (e) {
            if (e.status === 409) simStatus('wait', 'A simulation is already running');
            else { simStatus('down', 'Could not start: ' + e.message); return; }
        }
        pollSim();
    }
    async function initSim() {
        try {
            const data = await getJSON('/api/demo/scenarios');
            data.scenarios.forEach(s => {
                const o = el('option', '', `${s.id}: ${s.description}`);
                o.value = s.id;
                $('sim-scenario').append(o);
            });
        } catch (e) { /* the run button still offers "All scenarios" */ }
        $('btn-sim').addEventListener('click', runSim);
        pollSim();
    }

    // ── Presenter view and keys ──

    function setPresenter(on) {
        document.body.classList.toggle('presenter', on);
        $('btn-presenter').setAttribute('aria-pressed', on ? 'true' : 'false');
        try { localStorage.setItem('guardrails-presenter', on ? '1' : '0'); } catch (e) { /* not persisted */ }
    }
    $('btn-presenter').addEventListener('click', () => setPresenter(!document.body.classList.contains('presenter')));
    try { if (localStorage.getItem('guardrails-presenter') === '1') setPresenter(true); } catch (e) { /* default off */ }

    $('btn-step').addEventListener('click', e => { e.currentTarget.blur(); step(); });
    $('btn-play').addEventListener('click', e => { e.currentTarget.blur(); play(); });
    $('btn-reset').addEventListener('click', e => { e.currentTarget.blur(); reset(); });
    document.addEventListener('keydown', e => {
        const t = e.target instanceof Element ? e.target : document.body;
        if (t.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
        // Space and Enter on a focused button or link keep their native meaning.
        if ((e.key === ' ' || e.key === 'Enter') && t.closest('button, a, summary')) return;
        if (e.key === ' ') { e.preventDefault(); step(); }
        else if (e.key === 'p' || e.key === 'P') play();
        else if (e.key === 'r' || e.key === 'R') reset();
        else if (e.key === 'ArrowRight' && state.stories.length) chooseStory((state.story + 1) % state.stories.length);
        else if (e.key === 'ArrowLeft' && state.stories.length) chooseStory((state.story - 1 + state.stories.length) % state.stories.length);
    });

    initReplay();
    initSim();
})();
