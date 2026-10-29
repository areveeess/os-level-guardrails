"""
Dashboard demo — replay of recorded episodes and the scripted policy simulation
================================================================================
Two ways to show the guardrails to a review panel without a GPU, a network or an API key:

1. Replay. A recorded InjecAgent episode from results/phase5-final/sources/<run>/raw.jsonl, both arms
   side by side. The case text (user request, tool result with the injected instruction) is rebuilt
   from the case id with src.benchmark.injecagent, the same loader the benchmark used. Only an allow
   list of record fields leaves the server, and the model and case are looked up, never joined into a
   path, so a request cannot reach any other file.

2. Simulation. The scripted scenarios of run_attack_simulation.py, passed through PolicyEngine.check
   one by one, paced so the live monitor shows each decision arriving. Nothing is executed: the
   policy engine only decides and logs (to logs/guardrails.jsonl), which is the protected round of
   that script without the commands. One run at a time, with a deadline.

Read-only towards results/: this module never writes there and never rebuilds the frozen data.
"""

import importlib.util
import json
import os
import re
import threading
import time
from functools import lru_cache

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SIMULATION_SCRIPT = os.path.join(PROJECT_ROOT, 'run_attack_simulation.py')

CASE_ID = re.compile(r'benign-\d{2}|d[hs]-u\d{2}-a\d{2}')
SOURCE_NAME = re.compile(r'[a-z0-9][a-z0-9._-]*')
SCENARIO_ID = re.compile(r'all|s[1-9]\d?')
MAX_RUN = 9

# Record fields a replay may show. Everything else (provenance stamps, token counts) stays on disk.
EPISODE_FIELDS = ('arm', 'run', 'kind', 'attack_type', 'attempted', 'succeeded', 'injection_shown',
                  'canary_exposed', 'attack_blocked', 'blocked_by', 'os_tags', 'user_task_done',
                  'false_positive', 'tool_calls', 'actions', 'final_message', 'stop_reason', 'steps',
                  'parse_errors', 'llm_error', 'refused_or_ignored')
OS_ACTIONS = ('run_shell', 'read_file', 'write_file', 'http_request')

SIM_STEP_SECONDS = 0.9      # pause between decisions, so the panel can follow them on the monitor
SIM_TIMEOUT_SECONDS = 30    # a run that has not finished by then stops


# ── Model line-up ──

def lineup(numbers: dict, models_yaml: dict) -> list:
    """Every model of the study, grouped for display, from numbers.json and the frozen models.yaml.

    Only the provider and the served model id are taken from models.yaml (never base_url or
    api_key_env)."""
    served = (models_yaml or {}).get('models') or {}

    def served_as(name):
        entry = served.get(name) or {}
        return entry.get('model') if isinstance(entry, dict) else None

    def group_of(m):
        if m.get('group') == 'control':
            return 'control'
        if m.get('group') == 'hosted':
            return 'hosted'
        family = (m.get('family') or '').lower()
        if 'from-scratch' in family or 'lora' in family:
            return 'locally made'
        return 'local pretrained'

    rows = []
    for m in numbers.get('models') or []:
        rows.append({'model': m.get('model'), 'group': group_of(m), 'family': m.get('family'),
                     'provider': m.get('provider'), 'served_as': served_as(m.get('model')),
                     'status': m.get('status'), 'label': m.get('label'),
                     'episodes_done': m.get('episodes_done'), 'episodes_expected': m.get('episodes_expected')})
    for m in numbers.get('legacy') or []:
        rows.append({'model': m.get('model'), 'group': 'legacy', 'family': m.get('family'),
                     'provider': m.get('provider'), 'served_as': served_as(m.get('model')),
                     'status': m.get('status'), 'label': 'older engine, not compared',
                     'episodes_done': m.get('episodes_done'), 'episodes_expected': m.get('episodes_expected')})
    return rows


# ── Replay ──

def replay_sources(numbers: dict, final_dir: str) -> dict:
    """model id -> raw.jsonl path, for the current-engine runs frozen in final_dir/sources/.

    Built from numbers.json, so the set of replayable models is fixed by the frozen data."""
    by_model = {m.get('model'): m for m in numbers.get('models') or []}
    control = [m for m, row in by_model.items() if row.get('group') == 'control']
    out = {}
    for s in numbers.get('sources') or []:
        name = str(s.get('name') or '')
        if s.get('group') not in ('current', 'control') or not SOURCE_NAME.fullmatch(name) or not name.startswith('phase4-'):
            continue
        model = name[len('phase4-'):]
        if model == 'control' and control:
            model = control[0]
        if model not in by_model:
            continue
        path = os.path.join(final_dir, 'sources', name, 'raw.jsonl')
        if os.path.isfile(path):
            out[model] = path
    return out


_episode_cache = {}


def load_records(path: str) -> list:
    """Parsed raw.jsonl, cached by modification time."""
    mtime = os.path.getmtime(path)
    hit = _episode_cache.get(path)
    if hit and hit[0] == mtime:
        return hit[1]
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    _episode_cache[path] = (mtime, records)
    return records


@lru_cache(maxsize=1)
def _cases() -> dict:
    from src.benchmark.injecagent import attack_cases, benign_cases
    return {c.id: c for c in benign_cases() + attack_cases()}


def case_text(case_id: str):
    """What the agent was asked and what the tool returned, for one case id; None if unknown."""
    c = _cases().get(case_id)
    if c is None:
        return None
    return {'id': c.id, 'kind': c.kind, 'prompt': c.prompt, 'user_tool': c.user_tool,
            'tool_response': c.tool_response, 'attacker_instruction': c.attacker_instruction,
            'attacker_tools': list(c.attacker_tools), 'attack_type': c.attack_type}


def case_list(records: list) -> list:
    """One summary per case (run 0 preferred), in case-id order."""
    seen = {}
    for r in records:
        cid = r.get('case_id')
        if not isinstance(cid, str) or not CASE_ID.fullmatch(cid):
            continue
        entry = seen.setdefault(cid, {'case_id': cid, 'kind': r.get('kind'), 'attack_type': r.get('attack_type') or '',
                                      'runs': set(), 'unprotected': None, 'protected': None})
        if isinstance(r.get('run'), int):
            entry['runs'].add(r['run'])
        arm = r.get('arm')
        if arm in ('unprotected', 'protected') and (entry[arm] is None or r.get('run') == 0):
            entry[arm] = {'succeeded': bool(r.get('succeeded')), 'attack_blocked': bool(r.get('attack_blocked')),
                          'user_task_done': bool(r.get('user_task_done')), 'attempted': bool(r.get('attempted')),
                          'layers': sorted(k for k, v in (r.get('blocked_by') or {}).items() if v)}
    out = []
    for cid in sorted(seen):
        e = seen[cid]
        e['runs'] = sorted(e['runs'])
        out.append(e)
    return out


def _policy_decisions(record: dict, engine):
    """Per-action policy decisions for the OS actions of a protected-arm record.

    The raw record keeps per-call broker decisions but only per-episode counts for the policy engine,
    so OS actions are re-checked with the policy engine; this is exact only when the policy file is the
    one the run used (same sha256), and is skipped otherwise."""
    if engine is None or record.get('arm') != 'protected':
        return None
    if record.get('policy_sha256') != engine.policy_sha256:
        return None
    out = []
    for a in record.get('actions') or []:
        kind = a.get('action') if isinstance(a, dict) else None
        if kind not in OS_ACTIONS:
            out.append(None)
            continue
        target = a.get('command') or a.get('path') or a.get('url') or ''
        decision, reason = engine.check_detailed(kind, target if isinstance(target, str) else repr(target))
        out.append({'decision': decision, 'reason': reason})
    return out


def _quiet_engine():
    """A PolicyEngine that logs nowhere, for re-checking recorded actions."""
    try:
        from src.policy_engine.engine import PolicyEngine
        return PolicyEngine(log_file=os.devnull, verbose=False)
    except Exception:
        return None


def episode(path: str, case_id: str, run: int) -> dict:
    """Both arms of one recorded (case, run), with allow-listed fields only."""
    arms = {}
    engine = None
    for r in load_records(path):
        if r.get('case_id') != case_id or r.get('run') != run or r.get('arm') not in ('unprotected', 'protected'):
            continue
        shown = {k: r.get(k) for k in EPISODE_FIELDS}
        if r.get('arm') == 'protected':
            engine = engine or _quiet_engine()
            shown['policy_checks'] = _policy_decisions(r, engine)
            shown['policy_exact'] = shown['policy_checks'] is not None
        arms[r['arm']] = shown
    return arms


def stories(sources: dict) -> list:
    """Three or four pre-picked episodes that tell the story, chosen from the data, not by hand."""
    out = []

    def pick(model, want, title, text):
        path = sources.get(model)
        if not path:
            return
        cases = case_list(load_records(path))
        for c in cases:
            if want(c):
                out.append({'model': model, 'case_id': c['case_id'], 'run': 0, 'title': title, 'text': text})
                return

    def hijacked_then_blocked(c, kind=None):
        u, p = c['unprotected'] or {}, c['protected'] or {}
        return (c['kind'] != 'benign' and (kind is None or c['kind'] == kind)
                and u.get('succeeded') and p.get('attack_blocked') and not p.get('succeeded'))

    pick('qwen2.5-0.5b-agentlora', lambda c: hijacked_then_blocked(c, 'ds'),
         'Qwen LoRA is hijacked, then stopped',
         'The model we fine-tuned obeys the injection and sends private data out. With guardrails the same call is refused.')
    pick('scratch-gpt', hijacked_then_blocked,
         'ScratchGPT, trained from zero, the same',
         'A model with no safety training at all. The guardrails do not depend on the model, so the result is the same.')
    pick('qwen2.5-0.5b-agentlora',
         lambda c: c['kind'] == 'benign' and (c['protected'] or {}).get('user_task_done')
         and (c['unprotected'] or {}).get('user_task_done'),
         'A normal task still works',
         'No injection here. The task is done with and without guardrails: least privilege did not get in the way.')
    control = [m for m in sources if m.startswith('control')]
    if control:
        # Prefer an episode where two layers act: the broker refuses the tool, the policy engine the egress.
        pick(control[0], lambda c: hijacked_then_blocked(c) and 'policy_engine' in (c['protected'] or {}).get('layers', []),
             'The scripted worst case',
             'Not a language model: a script that always obeys the injection. Every attempt is blocked.')
    return out


# ── Simulation ──

@lru_cache(maxsize=1)
def scenarios() -> tuple:
    """The scripted scenarios of run_attack_simulation.py, unchanged, as (id, description, action, malicious)."""
    spec = importlib.util.spec_from_file_location('guardrails_attack_simulation', SIMULATION_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # the script's main block is guarded by __name__ == '__main__'
    out = []
    for i, a in enumerate(module.ATTACKS, start=1):
        action = dict(a['action'])
        out.append({'id': f's{i}', 'description': a['description'], 'action': action,
                    'action_type': action.get('action'),
                    'target': action.get('command') or action.get('path') or action.get('url', ''),
                    'malicious': bool(a.get('malicious'))})
    return tuple(out)


class Simulation:
    """One scripted policy run at a time, in a background thread, with a deadline."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = {'status': 'idle', 'scenario': None, 'total': 0, 'done': 0, 'decisions': [],
                       'started': None, 'finished': None, 'error': None}

    def status(self) -> dict:
        with self._state_lock:
            s = dict(self._state)
            s['decisions'] = list(s['decisions'])
            return s

    def _set(self, **kw):
        with self._state_lock:
            self._state.update(kw)

    def start(self, scenario_id: str, log_file: str) -> bool:
        """Start a run; False when one is already running."""
        chosen = [s for s in scenarios() if scenario_id in ('all', s['id'])]
        if not chosen:
            raise KeyError(scenario_id)
        if not self._lock.acquire(blocking=False):
            return False
        self._set(status='running', scenario=scenario_id, total=len(chosen), done=0, decisions=[],
                  started=time.time(), finished=None, error=None)
        t = threading.Thread(target=self._run, args=(chosen, log_file), daemon=True)
        try:
            t.start()
        except RuntimeError:
            self._set(status='error', error='could not start', finished=time.time())
            self._lock.release()
            raise
        return True

    def _run(self, chosen: list, log_file: str):
        deadline = time.monotonic() + SIM_TIMEOUT_SECONDS
        try:
            from src.policy_engine.engine import PolicyEngine
            engine = PolicyEngine(log_file=log_file, verbose=False)
            for i, s in enumerate(chosen):
                if time.monotonic() > deadline:
                    self._set(status='timeout', finished=time.time())
                    return
                decision, reason = engine.check_detailed(s['action_type'], s['target'])
                with self._state_lock:
                    self._state['decisions'].append({'id': s['id'], 'description': s['description'],
                                                     'action_type': s['action_type'], 'target': s['target'],
                                                     'malicious': s['malicious'], 'decision': decision,
                                                     'reason': reason})
                    self._state['done'] = i + 1
                if i + 1 < len(chosen):
                    time.sleep(SIM_STEP_SECONDS)
            self._set(status='done', finished=time.time())
        except Exception as e:      # reported to the page, never raised into the server
            self._set(status='error', error=type(e).__name__, finished=time.time())
        finally:
            self._lock.release()

    def wait(self, timeout: float = 10.0) -> bool:
        """For tests: block until the current run ends."""
        if self._lock.acquire(timeout=timeout):
            self._lock.release()
            return True
        return False
