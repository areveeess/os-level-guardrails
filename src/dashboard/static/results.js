/*
 * Results view: renders results/phase5-final/numbers.json (via /api/results) as figures, two
 * native SVG charts, the headline / partial / legacy tables, blocks by layer and provenance.
 * Read-only. Every string from the file is inserted with textContent.
 */
'use strict';

(function () {
    const root = document.getElementById('results-root');
    const API = root.dataset.api;
    const FILES = root.dataset.files;
    const SVG = 'http://www.w3.org/2000/svg';
    const LAYERS = [
        ['tool_broker', 'Tool broker'],
        ['policy_engine', 'Policy engine'],
        ['sandbox', 'Namespaces'],
        ['seccomp', 'seccomp-BPF'],
    ];

    // ── Small helpers ──

    function el(tag, cls, text) {
        const n = document.createElement(tag);
        if (cls) n.className = cls;
        if (text !== undefined && text !== null) n.textContent = text;
        return n;
    }
    function sv(tag, attrs, text) {
        const n = document.createElementNS(SVG, tag);
        for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
        if (text !== undefined) n.textContent = text;
        return n;
    }
    function icon(name) {
        const i = el('i', 'ti ' + name);
        i.setAttribute('aria-hidden', 'true');
        return i;
    }
    function num(x) { return Number(x).toLocaleString(); }
    function p1(x) { return x == null ? '' : Number(x).toFixed(1).replace(/\.0$/, '') + '%'; }
    function pFull(x) { return x == null ? '' : Number(x).toFixed(1) + '%'; }

    // Rows in numbers.json carry {pct, k, n, lo, hi}; legacy rows carry the same values flattened.
    function metric(m, key) {
        if (m.metrics && m.metrics[key]) return m.metrics[key];
        if (m[key + '_n'] !== undefined) {
            return { pct: m[key + '_pct'], k: m[key + '_k'], n: m[key + '_n'], lo: m[key + '_lo'], hi: m[key + '_hi'] };
        }
        return null;
    }
    function isLLM(m) { return m.group !== 'control'; }

    function panel(title, sub) {
        const p = el('section', 'panel');
        const head = el('div', 'panel-head');
        const h = el('h2', '', title);
        head.append(h);
        if (sub) head.append(el('p', 'sub', sub));
        p.append(head);
        return p;
    }

    // ── Tooltip ──

    function makeTip(host) {
        const tip = el('div', 'tip');
        tip.hidden = true;
        tip.setAttribute('role', 'status');
        host.append(tip);
        return {
            show(evOrNode, title, rows) {
                tip.replaceChildren(el('div', 'tip-h', title));
                for (const r of rows) {
                    const row = el('div', 'tip-r');
                    const key = el('span', 'tip-key');
                    key.style.borderColor = r.color;
                    row.append(key, el('b', '', r.value), el('span', '', r.label));
                    tip.append(row);
                }
                tip.hidden = false;
                const hb = host.getBoundingClientRect();
                let x, y;
                if (evOrNode.clientX !== undefined) { x = evOrNode.clientX - hb.left; y = evOrNode.clientY - hb.top; }
                else { const b = evOrNode.getBoundingClientRect(); x = b.left + b.width / 2 - hb.left; y = b.top - hb.top; }
                const w = tip.offsetWidth, h = tip.offsetHeight;
                let left = x + 14;
                if (left + w > hb.width - 4) left = Math.max(4, x - w - 14);
                let top = y - h - 10;
                if (top < 0) top = y + 16;
                tip.style.left = left + 'px';
                tip.style.top = top + 'px';
            },
            hide() { tip.hidden = true; },
        };
    }
    function ciText(m) { return m ? `${num(m.k)}/${num(m.n)} [${m.lo}, ${m.hi}]` : 'no data'; }

    // ── Chart 1: attack success with intervals ──

    function drawIntervals(wrap, rows, controlPct, tip) {
        const W = Math.max(280, wrap.clientWidth - 2);
        const stacked = W < 600;
        const labelW = stacked ? 0 : Math.min(230, W * 0.3);
        const right = 58;
        const top = 26, bottom = 40;
        const rowH = stacked ? 52 : 36;
        const headH = 30;
        let h = top;
        const layout = rows.map(r => { const y = h; h += r.heading ? headH : rowH; return { r, y }; });
        const H = h + bottom;
        const x0 = labelW + (stacked ? 4 : 12), x1 = W - right;
        const x = v => x0 + (Math.max(0, Math.min(100, v)) / 100) * (x1 - x0);

        const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img',
            'aria-label': 'Attack success per model without and with guardrails, with 95% Wilson intervals' });
        for (const t of [0, 25, 50, 75, 100]) {
            svg.append(sv('line', { class: 'grid', x1: x(t), x2: x(t), y1: top - 6, y2: H - bottom + 4 }));
            svg.append(sv('text', { x: x(t), y: H - bottom + 20, 'text-anchor': 'middle' }, t + '%'));
        }
        svg.append(sv('text', { x: x0, y: H - 4 }, 'Attack success, share of attack episodes'));
        if (controlPct != null) {
            svg.append(sv('line', { class: 'ref', x1: x(controlPct), x2: x(controlPct), y1: top - 10, y2: H - bottom + 4 }));
            svg.append(sv('text', { x: x(controlPct) - 6, y: top - 12, 'text-anchor': 'end' }, 'Scripted control, no guardrails'));
        }

        for (const { r, y } of layout) {
            if (r.heading) {
                svg.append(sv('text', { class: 't-group', x: stacked ? 0 : 0, y: y + 20 }, r.heading));
                continue;
            }
            const g = sv('g', { tabindex: 0, role: 'listitem', 'aria-label':
                `${r.label}: without guardrails ${pFull(r.off.pct)}, with guardrails ${pFull(r.on.pct)}` });
            const hot = sv('rect', { class: 'row-hot', x: 0, y: y + 1, width: W, height: rowH - 2, rx: 8, opacity: 0 });
            g.append(hot);
            const cy = stacked ? y + 34 : y + rowH / 2;
            if (stacked) g.append(sv('text', { class: 't-label', x: 4, y: y + 15 }, r.label + (r.partial ? '  (partial)' : '')));
            else {
                g.append(sv('text', { class: 't-label', x: 4, y: cy + 4 }, r.label));
                if (r.partial) g.append(sv('text', { class: 'partial-tag', x: 4, y: cy + 17 }, 'partial, not final'));
            }
            const yo = cy - 5, yn = cy + 5;
            g.append(sv('line', { class: 'ci-off', x1: x(r.off.lo), x2: x(r.off.hi), y1: yo, y2: yo }));
            g.append(sv('line', { class: 'ci-on', x1: x(r.on.lo), x2: x(r.on.hi), y1: yn, y2: yn }));
            g.append(sv('circle', { class: 'dot-off', cx: x(r.off.pct), cy: yo, r: 5 }));
            g.append(sv('circle', { class: 'dot-on', cx: x(r.on.pct), cy: yn, r: 5 }));
            // One direct label per row: the no-guardrails value, beside its interval.
            const lx = x(r.off.hi) + 8;
            const fits = lx + 40 < W;
            g.append(sv('text', { class: 't-val', x: fits ? lx : x(r.off.lo) - 8, y: yo + 4, 'text-anchor': fits ? 'start' : 'end' }, p1(r.off.pct)));
            g.append(sv('rect', { class: 'hit', x: 0, y, width: W, height: rowH }));
            const rowsTip = [
                { color: 'var(--s-off)', value: pFull(r.off.pct), label: 'no guardrails, ' + ciText(r.off) },
                { color: 'var(--s-on)', value: pFull(r.on.pct), label: 'guardrails, ' + ciText(r.on) },
            ];
            const on = e => { hot.setAttribute('opacity', 1); tip.show(e, r.label, rowsTip); };
            const off = () => { hot.setAttribute('opacity', 0); tip.hide(); };
            g.addEventListener('pointermove', on);
            g.addEventListener('pointerleave', off);
            g.addEventListener('focus', () => on(g));
            g.addEventListener('blur', off);
            svg.append(g);
        }
        svg.setAttribute('role', 'list');
        return svg;
    }

    // ── Chart 2: benign task done vs attack success, no guardrails -> guardrails ──

    function drawArrows(wrap, pts, tip) {
        const W = Math.max(280, wrap.clientWidth - 2);
        const H = Math.round(Math.max(300, Math.min(460, W * 0.58)));
        const L = 52, R = 18, T = 14, B = 46;
        const x = v => L + (v / 100) * (W - L - R);
        const y = v => T + (1 - v / 100) * (H - T - B);
        const svg = sv('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'list',
            'aria-label': 'Benign task completion against attack success for each model, without and with guardrails' });
        const defs = sv('defs');
        const marker = sv('marker', { id: 'arrowhead', viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' });
        marker.append(sv('path', { d: 'M0 0 L10 5 L0 10 z', class: 'arrow-head' }));
        defs.append(marker);
        svg.append(defs);
        for (const t of [0, 25, 50, 75, 100]) {
            svg.append(sv('line', { class: 'grid', x1: x(t), x2: x(t), y1: T, y2: H - B }));
            svg.append(sv('line', { class: 'grid', x1: L, x2: W - R, y1: y(t), y2: y(t) }));
            svg.append(sv('text', { x: x(t), y: H - B + 18, 'text-anchor': 'middle' }, t + '%'));
            svg.append(sv('text', { x: L - 8, y: y(t) + 4, 'text-anchor': 'end' }, t + '%'));
        }
        svg.append(sv('text', { x: W - R, y: H - 6, 'text-anchor': 'end' }, 'Benign task done (utility)'));
        const yTitle = sv('text', { x: 14, y: T + (H - T - B) / 2, 'text-anchor': 'middle',
            transform: `rotate(-90 14 ${T + (H - T - B) / 2})` }, 'Attack success');
        svg.append(yTitle);

        for (const p of pts) {
            const ax = x(p.x0), ay = y(p.y0), bx = x(p.x1), by = y(p.y1);
            const g = sv('g', { tabindex: 0, role: 'listitem', 'aria-label':
                `${p.label}: without guardrails ${pFull(p.y0)} attack success and ${pFull(p.x0)} benign task done; with guardrails ${pFull(p.y1)} and ${pFull(p.x1)}` });
            const dist = Math.hypot(bx - ax, by - ay);
            if (dist > 16) {
                const ux = (bx - ax) / dist, uy = (by - ay) / dist;
                g.append(sv('line', { class: 'arrow', x1: ax + ux * 8, y1: ay + uy * 8, x2: bx - ux * 9, y2: by - uy * 9, 'marker-end': 'url(#arrowhead)' }));
            }
            g.append(sv('circle', { class: 'ring-off', cx: ax, cy: ay, r: 6 }));
            g.append(sv('circle', { class: 'dot-on', cx: bx, cy: by, r: 5 }));
            if (p.labelled) {
                const right = ax < W - 170;
                g.append(sv('text', { class: 't-label', x: right ? ax + 11 : ax - 11, y: ay + 4, 'text-anchor': right ? 'start' : 'end' }, p.label));
            }
            const hitA = sv('circle', { class: 'hit', cx: ax, cy: ay, r: 14 });
            const hitB = sv('circle', { class: 'hit', cx: bx, cy: by, r: 14 });
            g.append(hitA, hitB);
            const rowsTip = [
                { color: 'var(--s-off)', value: pFull(p.y0), label: 'attack success, no guardrails' },
                { color: 'var(--s-on)', value: pFull(p.y1), label: 'attack success, guardrails' },
                { color: 'var(--s-off)', value: pFull(p.x0), label: 'benign task done, no guardrails' },
                { color: 'var(--s-on)', value: pFull(p.x1), label: 'benign task done, guardrails' },
            ];
            const on = e => tip.show(e, p.label, rowsTip);
            g.addEventListener('pointermove', on);
            g.addEventListener('pointerleave', () => tip.hide());
            g.addEventListener('focus', () => on(hitA));
            g.addEventListener('blur', () => tip.hide());
            svg.append(g);
        }
        return svg;
    }

    function chartPanel(title, sub, legendItems, draw) {
        const p = panel(title, sub);
        const legend = el('div', 'legend');
        for (const it of legendItems) {
            const s = el('span');
            const key = el('span', it.cls);
            if (it.color) { key.style.background = it.cls === 'key-dot' ? it.color : ''; key.style.color = it.color; }
            s.append(key, document.createTextNode(it.label));
            legend.append(s);
        }
        p.append(legend);
        const wrap = el('div', 'chart-wrap');
        p.append(wrap);
        const tip = makeTip(wrap);
        let lastW = 0;
        const render = () => {
            const w = wrap.clientWidth;
            if (!w || w === lastW) return;
            lastW = w;
            wrap.querySelectorAll('svg').forEach(n => n.remove());
            wrap.prepend(draw(wrap, tip));
        };
        new ResizeObserver(render).observe(wrap);
        return p;
    }

    // ── Tables ──

    function th(text) { const c = el('th', '', text); c.scope = 'col'; return c; }
    function modelCell(m) {
        const td = el('td');
        td.append(el('span', 'm', m.model));
        if (m.family) td.append(el('span', 'fam', m.family));
        return td;
    }
    function valueCell(x, tone) {
        const td = el('td');
        if (!x || x.n === 0 || x.n == null) { td.append(el('span', 'na', 'no data')); return td; }
        const v = el('span', 'v', pFull(x.pct));
        if (tone) v.dataset.tone = tone;
        td.append(v, el('span', 'kn', `${num(x.k)}/${num(x.n)}`), el('span', 'kn', `[${x.lo}, ${x.hi}]`));
        return td;
    }
    function table(caption, heads, rows) {
        const wrap = el('div', 'scroll');
        const t = el('table', 'dt');
        t.append(el('caption', 'sr-only', caption));
        const tr = el('tr');
        heads.forEach(h => tr.append(th(h)));
        const thead = el('thead');
        thead.append(tr);
        const tbody = el('tbody');
        rows.forEach(r => tbody.append(r));
        t.append(thead, tbody);
        wrap.append(t);
        return wrap;
    }
    const COLS = ['Model', 'Episodes done', 'Attempted, no guardrails', 'Success, no guardrails', 'Success, guardrails',
                  'Blocked given attempt', 'Benign task, no guardrails', 'Benign task, guardrails', 'False positive', 'Parse errors per step'];
    function modelRow(m) {
        const tr = el('tr');
        tr.append(modelCell(m));
        const ep = el('td');
        ep.append(el('span', 'v', `${num(m.episodes_done)}/${num(m.episodes_expected)}`));
        if (m.label && m.label !== 'complete') ep.append(el('span', 'kn', m.label.replace(/^partial, \d+\/\d+ episodes ?/, '') || 'partial'));
        tr.append(ep);
        const su = metric(m, 'success_unprotected');
        const sp = metric(m, 'success_protected');
        tr.append(valueCell(metric(m, 'attempted')));
        tr.append(valueCell(su, su && su.pct >= 10 ? 'bad' : null));
        tr.append(valueCell(sp, sp ? (sp.k === 0 ? 'good' : 'bad') : null));
        tr.append(valueCell(metric(m, 'blocked_given_attempt')));
        tr.append(valueCell(metric(m, 'benign_task_unprotected')));
        tr.append(valueCell(metric(m, 'benign_task_protected')));
        tr.append(valueCell(metric(m, 'false_positive')));
        tr.append(valueCell(metric(m, 'parse_error_rate')));
        return tr;
    }

    // ── Sections ──

    function stateBanner(d) {
        const final = d.data_state === 'final';
        const box = el('div', 'state');
        box.dataset.state = final ? 'final' : 'snapshot';
        box.append(icon(final ? 'ti-lock' : 'ti-clock'));
        const text = el('div');
        const l1 = el('p');
        const l2 = el('p');
        if (final) {
            l1.append(el('b', '', 'Final data. '), document.createTextNode(`Frozen at ${d.freeze_display || d.freeze_utc} (UTC).`));
            l2.textContent = 'Partial rows are still labelled with the episodes they reached and are not ranked against complete runs.';
        } else {
            l1.append(el('b', '', 'Snapshot, not final. '),
                      document.createTextNode(`Taken ${d.snapshot_display || d.snapshot_utc} (UTC). The freeze is ${d.freeze_display || d.freeze_utc}.`));
            const freeze = new Date(d.freeze_utc);
            let when = '';
            if (!isNaN(freeze)) {
                const mins = Math.round((freeze - Date.now()) / 60000);
                when = mins > 0
                    ? `The freeze is ${mins >= 60 ? Math.floor(mins / 60) + ' h ' : ''}${mins % 60} min away. `
                    : 'The freeze time has passed; these numbers stay until the final build replaces them. ';
            }
            l2.textContent = when + 'Hosted rows that are still running are partial and their rates will change.';
        }
        text.append(l1, l2);
        box.append(text);
        const badge = document.getElementById('state-badge');
        badge.textContent = final ? 'Final data' : 'Snapshot ' + (d.snapshot_display || '');
        badge.dataset.tone = final ? 'accent' : 'warn';
        return box;
    }

    function figures(d) {
        const llm = d.models.filter(isLLM);
        let k = 0, n = 0;
        for (const m of llm) { const s = metric(m, 'success_protected'); if (s) { k += s.k; n += s.n; } }
        const complete = llm.filter(m => m.complete);
        let worst = null;
        for (const m of complete) {
            const s = metric(m, 'success_unprotected');
            if (s && (!worst || s.pct > worst.s.pct)) worst = { m, s };
        }
        const pooled = d.pooled && d.pooled.blocked_given_attempt;
        const ctl = d.control && d.control.protected;

        const wrap = el('section', 'figures');
        wrap.setAttribute('aria-label', 'Key figures');
        const fig = (hero, k1, v, desc) => {
            const f = el('div', 'panel panel-pad fig' + (hero ? ' fig-hero' : ''));
            f.append(el('span', 'fig-k', k1), el('span', 'fig-v', v), el('span', 'fig-d', desc));
            wrap.append(f);
        };
        fig(true, 'Attacks that succeeded with guardrails on', `${num(k)} of ${num(n)}`,
            `attack episodes across all ${llm.length} LLM models, partial runs included`);
        if (pooled) fig(false, 'Attempted attacks blocked', pFull(pooled.pct),
            `${num(pooled.k)}/${num(pooled.n)} pooled, 95% interval [${pooled.lo}, ${pooled.hi}]`);
        if (worst) fig(false, 'Worst model without guardrails', pFull(worst.s.pct),
            `${worst.m.model} succeeded in ${num(worst.s.k)}/${num(worst.s.n)} attack episodes`);
        if (ctl) fig(false, 'Scripted control blocked', `${num(ctl.blocked)}/${num(ctl.attempted)}`,
            `an agent that always obeys the injection; ${num(ctl.succeeded)} succeeded with guardrails`);
        return wrap;
    }

    function intervalChart(d) {
        const toRow = m => ({ label: m.model, off: metric(m, 'success_unprotected'), on: metric(m, 'success_protected'), partial: !m.complete });
        const byOff = (a, b) => b.off.pct - a.off.pct;
        const complete = d.models.filter(m => isLLM(m) && m.complete).map(toRow).filter(r => r.off && r.on).sort(byOff);
        const partial = d.models.filter(m => isLLM(m) && !m.complete).map(toRow).filter(r => r.off && r.on).sort(byOff);
        const rows = [{ heading: 'Complete runs' }, ...complete];
        if (partial.length) rows.push({ heading: 'Partial runs, not final and not ranked' }, ...partial);
        const ctl = d.models.find(m => m.group === 'control');
        const controlPct = ctl ? (metric(ctl, 'success_unprotected') || {}).pct : null;
        return chartPanel('Attack success, without and with guardrails',
            'Each row is one model. Dots are the rate over attack episodes; lines are 95% Wilson intervals, which are optimistic because repeated cases are not independent.',
            [{ cls: 'key-dot', color: 'var(--s-off)', label: 'No guardrails' },
             { cls: 'key-dot', color: 'var(--s-on)', label: 'Guardrails' },
             { cls: 'key-line', label: 'Scripted control without guardrails' }],
            (wrap, tip) => drawIntervals(wrap, rows, controlPct, tip));
    }

    function arrowChart(d) {
        const pts = d.models.filter(m => m.complete).map(m => {
            const su = metric(m, 'success_unprotected'), sp = metric(m, 'success_protected');
            const bu = metric(m, 'benign_task_unprotected'), bp = metric(m, 'benign_task_protected');
            if (!su || !sp || !bu || !bp) return null;
            return { label: m.model, x0: bu.pct, y0: su.pct, x1: bp.pct, y1: sp.pct };
        }).filter(Boolean);
        // Name the points that stand apart; the clustered ones are in the tooltip and the table.
        pts.forEach(p => { p.labelled = p.y0 >= 15 || p.x0 < 50; });
        return chartPanel('Capability against vulnerability',
            'Complete runs only. Each arrow moves a model from no guardrails (hollow) to guardrails (filled). Down is safer; right is more useful.',
            [{ cls: 'key-ring', color: 'var(--s-off)', label: 'No guardrails' },
             { cls: 'key-dot', color: 'var(--s-on)', label: 'Guardrails' }],
            (wrap, tip) => drawArrows(wrap, pts, tip));
    }

    function headline(d) {
        const p = panel('Headline: complete runs',
            'Only models whose run finished every expected episode. Each cell is the rate, k/n episodes, and the 95% Wilson interval.');
        const rows = d.models.filter(m => isLLM(m) && m.complete).map(modelRow);
        p.append(table('Headline results for complete runs', COLS, rows));
        return p;
    }

    function partial(d) {
        const ms = d.models.filter(m => isLLM(m) && !m.complete);
        if (!ms.length) return null;
        const p = panel('Partial runs, not compared',
            'Hosted runs stopped by free-tier quota. Their rates are not final and must not be ranked against the complete runs.');
        p.append(table('Partial runs', COLS, ms.map(modelRow)));
        return p;
    }

    function legacy(d) {
        if (!d.legacy || !d.legacy.length) return null;
        const p = panel('Legacy rows, older engine, not compared',
            'Recorded before the review fixes to the policy engine. Kept for the record only.');
        const cols = ['Model', 'Status', 'Episodes done', 'Success, no guardrails', 'Success, guardrails', 'Blocked given attempt',
                      'Benign task, no guardrails', 'Benign task, guardrails', 'False positive'];
        const rows = d.legacy.map(m => {
            const tr = el('tr');
            tr.append(modelCell(m));
            tr.append(el('td', '', m.status));
            const ep = el('td');
            ep.append(el('span', 'v', `${num(m.episodes_done)}/${num(m.episodes_expected)}`));
            tr.append(ep);
            ['success_unprotected', 'success_protected', 'blocked_given_attempt', 'benign_task_unprotected',
             'benign_task_protected', 'false_positive'].forEach(k => tr.append(valueCell(metric(m, k))));
            return tr;
        });
        p.append(table('Legacy rows', cols, rows));
        return p;
    }

    function layerSection(d) {
        const p = panel('Where attacks were stopped',
            'Share of episodes in which each layer blocked at least one action, protected arm. One episode can be stopped at more than one layer, so the bars do not add up.');
        const pl = d.pooled && d.pooled.layers;
        if (pl) {
            const bars = el('div', 'bars');
            for (const [kind, label] of [['attack', 'Attack episodes'], ['benign', 'Benign episodes (blocks here are false positives)']]) {
                const g = pl[kind];
                if (!g) continue;
                bars.append(el('div', 'bars-group', `${label}, pooled over all LLM models, n = ${num(g.n)}`));
                for (const [key, name] of LAYERS) {
                    const x = g[key];
                    if (!x) continue;
                    const row = el('div', 'bar-row');
                    row.append(el('span', '', name));
                    const track = el('div', 'bar-track');
                    const fill = el('div', 'bar-fill');
                    fill.style.width = Math.max(0, Math.min(100, x.pct)) + '%';
                    if (!x.k) fill.style.display = 'none';
                    const val = el('span', 'bar-val', `${pFull(x.pct)}  (${num(x.k)}/${num(x.n)})`);
                    val.style.left = Math.min(100, x.pct) + '%';
                    track.append(fill, val);
                    row.append(track);
                    row.setAttribute('aria-label', `${name}: ${pFull(x.pct)}, ${x.k} of ${x.n} ${kind} episodes, ${x.actions} blocked actions`);
                    bars.append(row);
                }
            }
            p.append(bars);
        }
        const rows = d.models.filter(m => m.layers && m.layers.attack).map(m => {
            const tr = el('tr');
            tr.append(modelCell(m));
            for (const [key] of LAYERS) {
                const x = m.layers.attack[key];
                const td = valueCell(x);
                if (x && x.k) {
                    const bar = el('span', 'cellbar');
                    bar.style.width = Math.max(2, x.pct) + '%';
                    td.append(bar);
                }
                tr.append(td);
            }
            return tr;
        });
        p.append(table('Attack episodes blocked at each layer, per model', ['Model'].concat(LAYERS.map(l => l[1])), rows));
        return p;
    }

    function controlSection(d) {
        const c = d.control;
        if (!c) return null;
        const p = panel('Model-independent control',
            'A scripted agent that always obeys the injection: the worst case, with no model judgement to rely on.');
        const body = el('div', 'bars');
        const line = (k, v) => { const r = el('div', 'bar-row'); r.append(el('span', '', k), el('span', 'num', v)); body.append(r); };
        if (c.unprotected) line('No guardrails', `${num(c.unprotected.succeeded)} of ${num(c.unprotected.episodes)} attack episodes succeeded`);
        if (c.protected) line('Guardrails', `${num(c.protected.blocked)} of ${num(c.protected.attempted)} attempts blocked, ${num(c.protected.succeeded)} succeeded`);
        if (c.cases) line('Distinct cases', `${num(c.cases.blocked)} of ${num(c.cases.attack)} blocked`);
        if (c.layers) {
            const parts = LAYERS.filter(([k]) => c.layers[k]).map(([k, n]) => `${n} ${num(c.layers[k].actions)}`);
            line('Blocked actions by layer', parts.join(', '));
        }
        p.append(body);
        return p;
    }

    function gallery(d) {
        if (!d.charts || !d.charts.length) return null;
        const p = panel('Report figures', 'The PNG charts written by the build, as they appear in the report and slides.');
        const g = el('div', 'gallery');
        g.style.padding = '16px 24px 24px';
        for (const c of d.charts) {
            const f = el('figure');
            const a = el('a');
            a.href = FILES + c.file;
            a.target = '_blank';
            a.rel = 'noopener';
            const img = el('img');
            img.src = FILES + c.file;
            img.alt = c.caption || c.file;
            img.loading = 'lazy';
            a.append(img);
            f.append(a, el('figcaption', '', c.caption || c.file));
            g.append(f);
        }
        p.append(g);
        return p;
    }

    function provenance(d) {
        const p = el('section', 'panel');
        const det = (title, note, content) => {
            const x = el('details', 'more');
            const s = el('summary');
            s.append(icon('ti-chevron-right'), document.createTextNode(title));
            if (note) s.append(el('span', '', note));
            x.append(s, content);
            return x;
        };
        if (d.threats && d.threats.length) {
            const ol = el('ol', 'threats');
            d.threats.forEach(t => ol.append(el('li', '', t)));
            p.append(det('Threats to validity', `${d.threats.length} items`, ol));
        }
        if (d.confounds && d.confounds.length) {
            const keys = ['parameters', 'precision', 'context', 'decoding', 'placement', 'provenance'];
            const rows = d.confounds.map(c => {
                const tr = el('tr');
                const td = el('td');
                td.append(el('span', 'm', c.model));
                tr.append(td);
                keys.forEach(k => tr.append(el('td', '', c[k] || '')));
                return tr;
            });
            p.append(det('Confounds between the models', 'size, precision, context, placement',
                table('Confounds', ['Model', 'Parameters', 'Precision', 'Context', 'Decoding', 'Placement', 'Provenance'], rows)));
        }
        if (d.sources && d.sources.length) {
            const rows = d.sources.map(s => {
                const tr = el('tr');
                const td = el('td');
                td.append(el('span', 'm', s.name));
                tr.append(td, el('td', '', s.group), el('td', 'num', num(s.raw_lines)));
                const sha = el('td');
                const code = el('span', 'sha', (s.raw_sha256 || '').slice(0, 16));
                code.title = s.raw_sha256 || '';
                sha.append(code);
                tr.append(sha);
                return tr;
            });
            p.append(det('Sources', `${d.sources.length} runs, raw.jsonl hashes`,
                table('Sources', ['Run', 'Group', 'Episodes in raw.jsonl', 'sha256 (first 16)'], rows)));
        }
        return p.childNodes.length ? p : null;
    }

    // ── Load ──

    function fail(title, text) {
        const box = el('div', 'panel blank');
        box.append(el('span', 'blank-i'));
        box.firstChild.append(icon('ti-chart-dots'));
        box.append(el('h3', '', title), el('p', '', text));
        root.replaceChildren(box);
        const badge = document.getElementById('state-badge');
        badge.textContent = 'No data';
        badge.dataset.tone = 'warn';
    }

    async function load() {
        let d;
        try {
            const r = await fetch(API, { cache: 'no-store' });
            if (r.status === 404) {
                fail('No frozen results yet', 'results/phase5-final/numbers.json was not found. It is written by the Phase 5 build; this page only reads it.');
                return;
            }
            if (!r.ok) throw new Error(String(r.status));
            d = await r.json();
        } catch (e) {
            fail('Could not read the results', 'The dashboard server did not return results/phase5-final/numbers.json. Check that it is running and reload.');
            return;
        } finally {
            root.setAttribute('aria-busy', 'false');
        }
        d.models = Array.isArray(d.models) ? d.models : [];
        const parts = [
            stateBanner(d), figures(d), intervalChart(d), arrowChart(d), headline(d), layerSection(d),
            controlSection(d), partial(d), legacy(d), gallery(d), provenance(d),
        ].filter(Boolean);
        const stack = el('div', 'stack');
        parts.slice(2).forEach(x => stack.append(x));
        root.replaceChildren(parts[0], parts[1], stack);
    }

    load();
})();
