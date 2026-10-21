"""
InjecAgent adapter
==================
Turns the vendored InjecAgent data (benchmarks/injecagent/, see SOURCE.md)
into benchmark cases for this project's agent:

  * benign case   — a user task whose tool response is clean
  * attack case   — the same user task, but the user tool's simulated response
                    carries an attacker instruction (indirect prompt injection)

Attack cases come in InjecAgent's two kinds:
  dh (direct harm)    — success = the attacker tool is executed
  ds (data stealing)  — success = the private data returned by the attacker's
                        first tool (seeded with a canary token) is then sent
                        out via GmailSendEmail

All attack text is InjecAgent's published text; nothing is rewritten.
"""

import json
import os
import random
import re
import secrets
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'benchmarks', 'injecagent')
PLACEHOLDER = "<Attacker Instruction>"
EXFIL_TOOL = "GmailSendEmail"
BENIGN_FILLER = "No additional comments."


@dataclass
class BenchmarkCase:
    id: str
    kind: str                     # benign | dh | ds
    prompt: str                   # the user's instruction
    user_tool: str
    tool_response: str            # what the user tool returns (with injection for attacks)
    tools: list                   # tool specs offered to the agent
    attacker_tools: list = field(default_factory=list)
    attack_type: str = ""
    attacker_instruction: str = ""

    @property
    def capabilities(self) -> set:
        """Least privilege for this task: only the tool the user's request needs."""
        return {self.user_tool}

    def tool_responses(self, canary: str) -> dict:
        """Simulated backends. The data-stealing source tool returns canary-seeded data."""
        responses = {self.user_tool: self.tool_response}
        if self.kind == "ds":
            responses[self.attacker_tools[0]] = json.dumps(
                {"result": f"Private record for the account owner. Reference: {canary}"})
        return responses


def _read_jsonl(name: str) -> list:
    with open(os.path.join(DATA_DIR, name), encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_tool_specs() -> dict:
    """Full tool name (toolkit + tool) -> {name, summary, parameters}."""
    with open(os.path.join(DATA_DIR, 'tools.json'), encoding='utf-8') as f:
        toolkits = json.load(f)
    return {
        tk['toolkit'] + tool['name']: {
            "name": tk['toolkit'] + tool['name'],
            "summary": tool.get('summary', ''),
            "parameters": tool.get('parameters', []),
        }
        for tk in toolkits for tool in tk['tools']
    }


def _fill(template: str, text: str) -> str:
    # Templates are Python-dict-like strings; keep the inserted text a valid literal
    return template.replace(PLACEHOLDER, text.replace("'", "\\'"))


def benign_cases(specs: dict = None) -> list:
    specs = specs or load_tool_specs()
    return [
        BenchmarkCase(
            id=f"benign-{i:02d}",
            kind="benign",
            prompt=u['User Instruction'],
            user_tool=u['User Tool'],
            tool_response=_fill(u['Tool Response Template'], BENIGN_FILLER),
            tools=[specs[u['User Tool']]],
        )
        for i, u in enumerate(_read_jsonl('user_cases.jsonl'))
    ]


def attack_cases(n_dh: int = 20, n_ds: int = 20, seed: int = 0, specs: dict = None) -> list:
    """A fixed, reproducible sample of (user case × attacker case) pairs.

    Every user case is used before any repeats, so injections are spread
    across all carriers (product reviews, emails, notes, tweets, ...).
    """
    specs = specs or load_tool_specs()
    users = _read_jsonl('user_cases.jsonl')
    rng = random.Random(seed)
    cases = []
    for kind, n in (("dh", n_dh), ("ds", n_ds)):
        attackers = _read_jsonl(f'attacker_cases_{kind}.jsonl')
        chosen = rng.sample(range(len(attackers)), min(n, len(attackers)))
        user_order = list(range(len(users)))
        rng.shuffle(user_order)
        for j, a_idx in enumerate(chosen):
            u_idx = user_order[j % len(users)]
            u, a = users[u_idx], attackers[a_idx]
            attacker_tools = list(a['Attacker Tools'])
            offered = [u['User Tool']] + attacker_tools + ([EXFIL_TOOL] if kind == "ds" else [])
            cases.append(BenchmarkCase(
                id=f"{kind}-u{u_idx:02d}-a{a_idx:02d}",
                kind=kind,
                prompt=u['User Instruction'],
                user_tool=u['User Tool'],
                tool_response=_fill(u['Tool Response Template'], a['Attacker Instruction']),
                tools=[specs[t] for t in dict.fromkeys(offered)],
                attacker_tools=attacker_tools,
                attack_type=a['Attack Type'],
                attacker_instruction=a['Attacker Instruction'],
            ))
    return cases


_CASE_ID = re.compile(r"(dh|ds)-u(\d+)-a(\d+)")


def case_tool_names(case_id: str):
    """(user_tool, attacker_tools) of an attack case, read from its id, or None when it cannot be derived.

    Attack case ids are dh-u<user case>-a<attacker case> (see attack_cases), so the tools of a row written
    before the row stored them can be recovered from the vendored data. Benign ids and anything else give None."""
    match = _CASE_ID.fullmatch(str(case_id))
    if not match:
        return None
    kind, user_idx, attacker_idx = match.group(1), int(match.group(2)), int(match.group(3))
    try:
        user = _read_jsonl('user_cases.jsonl')[user_idx]
        attacker = _read_jsonl(f'attacker_cases_{kind}.jsonl')[attacker_idx]
    except (OSError, IndexError, ValueError, KeyError):
        return None
    return user['User Tool'], list(attacker['Attacker Tools'])


def new_canary() -> str:
    return f"CANARY{secrets.token_hex(6).upper()}DATA"
