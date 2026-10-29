/*
 * Live monitor: polls /api/stats and /api/logs and renders the summary, the defense rings and
 * the decision feed. All log text is inserted with textContent (log fields are untrusted: they
 * hold whatever the agent tried to run).
 */
'use strict';

(function () {
    const POLL_MS = 2000;
    const MAX_ROWS = 400;
    const SVG = 'http://www.w3.org/2000/svg';

    // Outermost first: the order an action meets the layers. Keys match the "layer" field written
    // by ToolBroker, PolicyEngine._log_decision and SandboxExecutor.
    const LAYERS = [
        { key: 'tool_broker',   name: 'Tool broker',     mech: 'Per-task tool allow list', kernel: false },
        { key: 'policy_engine', name: 'Policy engine',   mech: 'Access-matrix rules in YAML', kernel: false },
        { key: 'sandbox',       name: 'Namespaces',      mech: 'Private user, mount, PID and network view', kernel: true },
        { key: 'seccomp',       name: 'seccomp-BPF',     mech: 'Kills chmod, mount, ptrace, setuid', kernel: true },
    ];
    const LAYER_NAME = Object.fromEntries(LAYERS.map(l => [l.key, l.name]));
    const DECISION = {
        ALLOW: { icon: 'ti-circle-check', word: 'Allowed' },
        FLAG: { icon: 'ti-alert-triangle', word: 'Flagged' },
        BLOCK: { icon: 'ti-ban', word: 'Blocked' },
    };
    const ACTION_ICON = {
        run_shell: 'ti-terminal-2', read_file: 'ti-file-text', write_file: 'ti-pencil',
        http_request: 'ti-world', call_tool: 'ti-tool',
    };

    const $ = id => document.getElementById(id);
    const state = {
        logs: [], filter: 'ALL', query: '', signature: null, prevCount: null,
        expanded: new Set(), loaded: false, isLinux: !!(window.GUARDRAILS && window.GUARDRAILS.isLinux),
    };

    function el(tag, cls, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined) n.textContent = text;
        return n;
    }
    function icon(name) {
        const i = el('i', 'ti ' + name);
        i.setAttribute('aria-hidden', 'true');
        return i;
    }
    function pct(n, total) {
        if (!total) return '0%';
        const p = 100 * n / total;
        return (p === 100 || p === 0 ? p.toFixed(0) : p.toFixed(1)) + '%';
    }
    function fmtTime(ts) {
        const d = new Date(ts);
        if (!ts || isNaN(d)) return ts ? String(ts).slice(11, 19) : '';
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    }
    function ago(ts) {
        const d = new Date(ts);
        if (!ts || isNaN(d)) return '';
        const s = Math.max(0, Math.round((Date.now() - d.getTime()) / 1000));
        if (s < 60) return s + ' s ago';
        if (s < 3600) return Math.round(s / 60) + ' min ago';
        if (s < 86400) return Math.round(s / 3600) + ' h ago';
        return d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + ' at ' + fmtTime(ts);
    }
    function announce(msg) { $('sr-announcer').textContent = msg; }
    function setLive(name, text) {
        $('live').dataset.state = name;
        $('live-text').textContent = text;
    }

    // ── Summary ──

    function renderSummary(s) {
        const total = s.total || 0;
        $('s-total').textContent = total.toLocaleString();
        $('s-block').textContent = (s.blocked || 0).toLocaleString();
        $('s-block').dataset.zero = s.blocked ? 'false' : 'true';
        $('s-block-2').textContent = (s.blocked || 0).toLocaleString();
        $('s-allow').textContent = (s.allowed || 0).toLocaleString();
        $('s-flag').textContent = (s.flagged || 0).toLocaleString();
        $('p-allow').textContent = pct(s.allowed || 0, total);
        $('p-flag').textContent = pct(s.flagged || 0, total);
        $('p-block').textContent = pct(s.blocked || 0, total);
        $('f-all').textContent = total;
        $('f-block').textContent = s.blocked || 0;
        $('f-flag').textContent = s.flagged || 0;
        $('f-allow').textContent = s.allowed || 0;
        $('btn-clear').setAttribute('aria-disabled', total ? 'false' : 'true');
        const first = state.logs[0];
        $('last').textContent = first ? 'Last decision ' + ago(first.timestamp) : 'No decisions yet';
    }

    // ── Defense rings ──

    function renderDepth() {
        // /api/stats counts every entry per layer (ALLOW rows too), so blocks per layer are
        // counted from the entries themselves.
        const blocks = {};
        let totalBlocks = 0;
        for (const l of state.logs) {
            if (l.decision !== 'BLOCK') continue;
            const k = l.layer || 'policy_engine';
            blocks[k] = (blocks[k] || 0) + 1;
            totalBlocks++;
        }
        const layers = LAYERS.slice();
        for (const k of Object.keys(blocks)) {
            if (!LAYER_NAME[k]) layers.push({ key: k, name: k, mech: 'Layer named in the log', kernel: false });
        }

        const svg = $('rings');
        svg.querySelectorAll('circle, text').forEach(n => n.remove());
        const step = 70 / Math.max(layers.length, 1);
        layers.forEach((ly, i) => {
            const n = blocks[ly.key] || 0;
            const share = totalBlocks ? n / totalBlocks : 0;
            const r = 102 - i * step;
            const track = document.createElementNS(SVG, 'circle');
            track.setAttribute('class', 'ring');
            for (const [k, v] of [['cx', 110], ['cy', 110], ['r', r], ['stroke-width', 7]]) track.setAttribute(k, v);
            track.dataset.layer = ly.key;
            svg.append(track);
            if (!n) return;
            // The arc's length is this layer's share of all blocks, drawn clockwise from the top.
            const arc = document.createElementNS(SVG, 'circle');
            const c = 2 * Math.PI * r;
            arc.setAttribute('class', 'ring-arc');
            for (const [k, v] of [['cx', 110], ['cy', 110], ['r', r], ['stroke-width', 7],
                                  ['stroke-dasharray', `${Math.max(4, share * c).toFixed(1)} ${c.toFixed(1)}`],
                                  ['transform', 'rotate(-90 110 110)']]) arc.setAttribute(k, v);
            svg.append(arc);
        });
        const core = document.createElementNS(SVG, 'circle');
        core.setAttribute('class', 'core');
        core.setAttribute('cx', 110);
        core.setAttribute('cy', 110);
        core.setAttribute('r', Math.max(18, 102 - layers.length * step - 6));
        svg.append(core);
        const t = document.createElementNS(SVG, 'text');
        t.setAttribute('x', 110);
        t.setAttribute('y', 113.5);
        t.setAttribute('text-anchor', 'middle');
        t.textContent = 'kernel';
        svg.append(t);

        const list = $('layers');
        list.replaceChildren(...layers.map((ly, i) => {
            const n = blocks[ly.key] || 0;
            const li = el('li');
            li.tabIndex = 0;
            li.dataset.layer = ly.key;
            li.append(el('span', 'layer-ix', String(i + 1)));
            li.append(el('span', 'layer-name', ly.name));
            const num = el('span', 'layer-n');
            const b = el('b', '', n.toLocaleString());
            b.dataset.zero = n ? 'false' : 'true';
            num.append(b, el('span', '', totalBlocks ? pct(n, totalBlocks) + ' of blocks' : 'blocked'));
            li.append(num);
            li.append(el('span', 'layer-mech', ly.mech));
            if (ly.kernel && !state.isLinux) {
                const note = el('span', 'layer-note');
                note.append(icon('ti-alert-triangle'), document.createTextNode('This host is not Linux, so this layer is not running'));
                li.append(note);
            }
            li.setAttribute('aria-label', `Layer ${i + 1}, ${ly.name}: ${n} blocked`);
            return li;
        }));
    }

    function hotRing(key) {
        $('rings').querySelectorAll('circle.ring').forEach(c => { c.dataset.hot = c.dataset.layer === key ? 'true' : 'false'; });
    }
    $('layers').addEventListener('pointerover', e => { const li = e.target.closest('li'); hotRing(li ? li.dataset.layer : null); });
    $('layers').addEventListener('pointerleave', () => hotRing(null));
    $('layers').addEventListener('focusin', e => { const li = e.target.closest('li'); hotRing(li ? li.dataset.layer : null); });
    $('layers').addEventListener('focusout', () => hotRing(null));

    // ── Feed ──

    function matches(l) {
        if (state.filter !== 'ALL' && l.decision !== state.filter) return false;
        if (!state.query) return true;
        return [l.target, l.reason, l.action, l.layer, LAYER_NAME[l.layer]]
            .some(v => v && String(v).toLowerCase().includes(state.query));
    }

    function buildRow(l, i, isNew) {
        const d = DECISION[l.decision] ? l.decision : 'OTHER';
        // Keyed from the oldest entry so a key survives new entries arriving on top.
        const key = (l.timestamp || '') + '|' + (state.logs.length - i);
        const tr = el('tr');
        tr.dataset.key = key;
        tr.tabIndex = 0;
        tr.setAttribute('aria-expanded', state.expanded.has(key) ? 'true' : 'false');
        if (isNew) tr.classList.add('is-new');

        const time = el('td', 'f-time', fmtTime(l.timestamp));
        if (l.timestamp) time.title = l.timestamp;

        const glyph = el('td', 'f-glyph');
        const g = el('span');
        g.dataset.d = d;
        g.append(icon(d === 'OTHER' ? 'ti-help' : DECISION[d].icon));
        glyph.append(g, el('span', 'sr-only', d === 'OTHER' ? String(l.decision || 'Unknown') : DECISION[d].word));

        const target = el('td', 'f-target', l.target == null ? '' : String(l.target));
        const reason = el('td', 'f-reason', l.reason || '');

        const meta = el('td', 'f-meta');
        const dec = el('span', 'f-dec', d === 'OTHER' ? String(l.decision || '?') : DECISION[d].word);
        dec.dataset.d = d;
        dec.setAttribute('aria-hidden', 'true');
        const act = el('span', 'f-act');
        act.append(icon(ACTION_ICON[l.action] || 'ti-point'),
                   document.createTextNode((LAYER_NAME[l.layer] || l.layer || 'Policy engine') + ', ' + (l.action || 'action')));
        meta.append(dec, act);

        tr.append(time, glyph, target, reason, meta);
        return tr;
    }

    function renderFeed(newCount) {
        const rows = [];
        let matched = 0;
        state.logs.forEach((l, i) => {
            if (!matches(l)) return;
            matched++;
            if (rows.length < MAX_ROWS) rows.push(buildRow(l, i, i < newCount));
        });
        $('tbody').replaceChildren(...rows);

        const total = state.logs.length;
        const empty = $('empty');
        $('feed-list').hidden = !matched;
        empty.hidden = !!matched;
        if (!total) {
            $('empty-h').textContent = 'Nothing mediated yet';
            $('empty-p').textContent = 'Run one of these from the project root. Each action the agent attempts appears here with the layer that decided it.';
            $('empty-runs').hidden = false;
        } else if (!matched) {
            $('empty-h').textContent = 'No decisions match';
            $('empty-p').textContent = 'Clear the search or choose All to see every decision.';
            $('empty-runs').hidden = true;
        }
        const shown = $('shown');
        shown.hidden = !total || (state.filter === 'ALL' && !state.query && matched <= MAX_ROWS);
        shown.textContent = matched > MAX_ROWS
            ? `Showing the newest ${MAX_ROWS} of ${matched} matching decisions`
            : `${matched} of ${total} decisions match`;
    }

    function renderGhosts() {
        const ghosts = $('ghosts');
        for (let i = 0; i < 4; i++) {
            const tr = el('tr', 'ghost');
            ['f-time', 'f-glyph', 'f-target', 'f-reason', 'f-meta'].forEach((c, j) => {
                const td = el('td', c);
                const s = el('span');
                s.style.width = [ '60px', '26px', (70 - i * 8) + '%', (55 + i * 6) + '%', '90px' ][j];
                td.append(s);
                tr.append(td);
            });
            ghosts.append(tr);
        }
    }

    // ── Polling ──

    async function getJSON(url) {
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) throw new Error(url + ' ' + r.status);
        return r.json();
    }

    async function refresh() {
        try {
            const [stats, logs] = await Promise.all([getJSON('/api/stats'), getJSON('/api/logs')]);
            setLive('up', 'Live');
            const list = Array.isArray(logs) ? logs : [];
            const sig = list.length + '|' + (list[0] && list[0].timestamp) + '|' + stats.total;
            if (sig === state.signature) {
                if (list[0]) $('last').textContent = 'Last decision ' + ago(list[0].timestamp);
                return;
            }
            let newCount = 0;
            if (state.prevCount !== null && list.length > state.prevCount) {
                newCount = list.length - state.prevCount;
                const newBlocks = list.slice(0, newCount).filter(l => l.decision === 'BLOCK').length;
                announce(`${newCount} new ${newCount === 1 ? 'decision' : 'decisions'}` + (newBlocks ? `, ${newBlocks} blocked` : ''));
            }
            state.signature = sig;
            state.prevCount = list.length;
            state.logs = list;
            if (!state.loaded) { $('ghosts').replaceChildren(); state.loaded = true; }
            renderSummary(stats);
            renderDepth();
            renderFeed(newCount);
        } catch (e) {
            setLive('down', 'Disconnected, retrying');
        }
    }

    async function loadInfo() {
        try {
            const info = await getJSON('/api/info');
            $('m-platform').textContent = info.platform;
            $('m-kernel').textContent = info.is_linux ? 'available' : 'need Linux or WSL2';
            if (!!info.is_linux !== state.isLinux) { state.isLinux = !!info.is_linux; renderDepth(); }
        } catch (e) { /* keep the server-rendered values */ }
    }

    // ── Clear (confirmation popover) ──

    const confirmBox = $('confirm');
    function openConfirm() {
        if ($('btn-clear').getAttribute('aria-disabled') === 'true') return;
        const n = state.logs.length;
        $('confirm-body').textContent = `All ${n.toLocaleString()} ${n === 1 ? 'entry' : 'entries'} in logs/guardrails.jsonl will be removed. This cannot be undone.`;
        confirmBox.hidden = false;
        $('btn-clear').setAttribute('aria-expanded', 'true');
        $('confirm-no').focus();
    }
    function closeConfirm(refocus) {
        confirmBox.hidden = true;
        $('btn-clear').setAttribute('aria-expanded', 'false');
        if (refocus) $('btn-clear').focus();
    }
    async function doClear() {
        closeConfirm(true);
        try {
            const r = await fetch('/api/clear', { method: 'POST' });
            if (!r.ok) throw new Error(String(r.status));
            state.signature = null;
            state.prevCount = null;
            state.expanded.clear();
            announce('Decision log cleared.');
            await refresh();
        } catch (e) {
            announce('Could not clear the log.');
            setLive('down', 'Clear failed');
        }
    }
    $('btn-clear').addEventListener('click', () => (confirmBox.hidden ? openConfirm() : closeConfirm(false)));
    $('confirm-no').addEventListener('click', () => closeConfirm(true));
    $('confirm-yes').addEventListener('click', doClear);
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && !confirmBox.hidden) closeConfirm(true); });
    document.addEventListener('pointerdown', e => {
        if (!confirmBox.hidden && !confirmBox.contains(e.target) && !e.target.closest('#btn-clear')) closeConfirm(false);
    });

    // ── Feed interaction ──

    function toggleRow(tr) {
        const open = tr.getAttribute('aria-expanded') !== 'true';
        tr.setAttribute('aria-expanded', open ? 'true' : 'false');
        if (open) state.expanded.add(tr.dataset.key); else state.expanded.delete(tr.dataset.key);
    }
    $('tbody').addEventListener('click', e => {
        const tr = e.target.closest('tr');
        if (tr && !String(window.getSelection())) toggleRow(tr);
    });
    $('tbody').addEventListener('keydown', e => {
        if ((e.key === 'Enter' || e.key === ' ') && e.target.matches('tr')) { e.preventDefault(); toggleRow(e.target); }
    });
    document.querySelectorAll('.chip').forEach(b => b.addEventListener('click', () => {
        state.filter = b.dataset.filter;
        document.querySelectorAll('.chip').forEach(x => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
        renderFeed(0);
    }));
    let qTimer = null;
    $('q').addEventListener('input', e => {
        clearTimeout(qTimer);
        qTimer = setTimeout(() => { state.query = e.target.value.trim().toLowerCase(); renderFeed(0); }, 120);
    });

    renderGhosts();
    renderDepth();
    loadInfo();
    refresh();
    setInterval(refresh, POLL_MS);
})();
