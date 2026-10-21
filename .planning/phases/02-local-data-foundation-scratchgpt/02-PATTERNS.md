# Phase 2: Local Data Foundation & ScratchGPT - Pattern Map

**Mapped:** 2026-10-21
**Files analyzed:** 17 new/modified
**Analogs found:** 16 / 17

All analog paths below were checked as git-tracked (`src/llm/*`, `src/benchmark/*`, `tests/*`, `docs/viva/*`), except `src/scratch_llm/{model,format,dataset}.py`, which are UNTRACKED pre-existing design work (`git ls-files src/scratch_llm` is empty). They are the files to finish and commit, not analogs to copy from elsewhere. `src/scratch_llm/__init__.py` does not exist.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match |
|---|---|---|---|---|
| `src/scratch_llm/__init__.py` (new) | package marker | - | `src/llm/__init__.py` (docstring only) | exact |
| `src/scratch_llm/format.py` (modify to v2) | utility | transform | itself (untracked) + `src/llm/scripted.py` `_parse_tools` | exact |
| `src/scratch_llm/dataset.py` (finish) | service/generator + CLI | batch / transform | itself (stub) + `src/benchmark/injecagent.py` (`DATA_DIR`, `load_tool_specs`, `EXFIL_TOOL`, `new_canary`) | role-match |
| `src/scratch_llm/model.py` (context default) | model | transform | itself | exact |
| `src/scratch_llm/train.py` (new) | script (GPU) | batch / file-I/O | `src/benchmark/runner.py` (`_git_commit`, `_git_dirty`, `_write_manifest`) | partial |
| `src/scratch_llm/inference.py` (new) | service | request-response | `src/llm/scratch.py` contract + `src/llm/scripted.py` | role-match |
| `src/llm/scratch.py` (modify) | provider | request-response | `src/llm/openai_compat.py` `generate` | exact |
| `config/models.yaml` (add `model: scratch-gpt`) | config | - | same file, other entries | exact |
| `data/local/manifest.json`, `models/scratch/manifest.json` | artifact | file-I/O | `runner._write_manifest` | exact |
| `.gitignore`, `requirements.txt` | config | - | existing `results/smoke-*/` entry | exact |
| `tests/test_scratch_dataset.py` | test | batch | `tests/test_control_agent.py` | role-match |
| `tests/test_training_leakage.py` | test | transform | `tests/test_report.py` (synthetic rows, planted cases) | role-match |
| `tests/test_format_parity.py` | test | transform | `tests/test_control_agent.py` `drive()` helper | role-match |
| `tests/test_scratch_train.py`, `tests/test_scratch_inference.py` | test (torch, tiny CPU) | batch | `tests/test_llm_layer.py` | role-match |
| `docs/viva/02-llm-foundations-and-synthetic-data.md` | docs | - | `docs/viva/01-reference-monitor-and-evaluation.md` | exact |
| `docs/viva/README.md` (flip row 02) | docs | - | row 01 | exact |
| `src/scratch_llm/` training-text + mask helpers | utility | transform | RESEARCH "Code Examples" | no real analog (use RESEARCH) |

## Pattern Assignments

### `src/scratch_llm/format.py` (utility, transform) - v2 changes

**Analog:** itself (lines 1-91, untracked) and `src/llm/scripted.py` lines 39-54.

Module docstring style, constants, vocab (keep; format.py:18-28):
```python
import json

CONTEXT_CHARS = 1536     # -> 2048 (D-05); rename the 400 reserve to GEN_RESERVE
OBS_CHARS = 640
ACTION_PREFIX = "A:"
END = "\n"
VOCAB = ["\n"] + [chr(c) for c in range(32, 127)]
STOI = {ch: i for i, ch in enumerate(VOCAB)}
UNKNOWN = STOI["?"]
```
Add `FORMAT_VERSION = 2`, `VOCAB_SHA256`, render fingerprint (hashlib; sha256 of `render_context(GOLDEN_MESSAGES)`), `supervised_mask`.

Tool-line parser to generalise into `tools_from_system(system) -> [(name, [required])]` (copy the regex, do NOT import `src.llm.scripted` from format.py) (scripted.py:39, 44-54):
```python
_TOOL_LINE = re.compile(r"^- (\w+)\((.*?)\): ", re.M)

def _parse_tools(system: str) -> list:
    _, marker, block = system.partition("Available tools:\n")
    if not marker:
        return []
    tools = []
    for name, params in _TOOL_LINE.findall(block):
        required = [p.split(":")[0].strip() for p in params.split(", ")
                    if p.strip() and not p.endswith(" (optional)")]
        tools.append((name, required))
    return tools
```
Replace the old names-only `render_header` (format.py:59-60) with `T:Name(req1,req2)|...` (`()` for none). Keep `render_context` structure (format.py:76-91) including the `OBSERVATION_TEMPLATE.split("{result}")[0]` prefix strip and `_flat(m["content"], 400)`; only change the tools call and the fit budget (`CONTEXT_CHARS - GEN_RESERVE`). Keep the lazy `from src.agent_harness.prompting import OBSERVATION_TEMPLATE` inside the function and keep format.py torch-free and free of dataset/benchmark imports.

---

### `src/scratch_llm/dataset.py` (generator + leakage gate + CLI, batch)

**Analog:** itself (stub, lines 29-69) + `src/benchmark/injecagent.py`.

Imports and held-out set already present (dataset.py:29-46):
```python
from src.agent_harness.prompting import parse_action
from src.benchmark.injecagent import DATA_DIR

def held_out_tools() -> set:
    names = set()
    with open(os.path.join(DATA_DIR, "user_cases.jsonl"), encoding="utf-8") as f:
        names |= {json.loads(line)["User Tool"] for line in f if line.strip()}
    for kind in ("dh", "ds"):
        ...
```
Also import `EXFIL_TOOL`, `load_tool_specs` from `src.benchmark.injecagent` and `| {EXFIL_TOOL}` explicitly. Always `sorted()` before any `rng.choice` (set order depends on PYTHONHASHSEED); use a single `random.Random(seed)`.

`validate_episode` (dataset.py:54-69) is the validation pattern: raise `ValueError`, unpack `(action, observation)` (works for lists after JSON round trip), call `parse_action(json.dumps(action))`. Extend it: tools are full spec dicts, and `fit` must be identity (`len(render_context(msgs_last)) - len("A:") <= CONTEXT_CHARS - GEN_RESERVE`).

Core pattern (RESEARCH "Code Examples", no repo equivalent): `episode_to_messages` via `build_messages` + `OBSERVATION_TEMPLATE.format(result=obs)`; `training_text = render_context(msgs_last) + compact_action(last) + END`; `supervised_mask` (payload of each `A:` line plus its newline); leakage gate (`_WORD = re.compile(r"[a-z0-9@._']+")`, 8-word shingles, normalised exact match, banned substrings, `CANARY[0-9A-F]{12}DATA`).

Canary/secret format to avoid reproducing (injecagent.py:133, 54-55): `f"CANARY{secrets.token_hex(6).upper()}DATA"` and `Private record for the account owner. Reference: {canary}`.

Dataset file writing: `open(path, "w", encoding="utf-8", newline="\n")` with `json.dumps(ep, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`; hash the file bytes; manifest via the pattern below.

---

### `src/scratch_llm/train.py` and manifest writers (script, batch/file-I/O)

**Analog:** `src/benchmark/runner.py` lines 135-157, 235-257 (no existing training loop; use RESEARCH "Training" for loop/optimizer/schedule).

Git stamps (copy verbatim, or import `_git_commit`/`_git_dirty` from `src.benchmark.runner` only if torch-free import cost is acceptable; copying avoids pulling the runner's imports):
```python
def _git_commit() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                              text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = done.stdout.strip()
    return sha if done.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha) else "unknown"

def _git_dirty():
    ... ["git", "status", "--porcelain", "--untracked-files=no"] ... returns True/False/None
```
Atomic manifest (runner.py:244-257):
```python
manifest = {"schema": 1, ..., "code_commit": code_commit, "code_dirty": code_dirty}
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
os.replace(tmp, path)
```
Conventions: fixed field list, no environment values, lowercase `manifest.json`, `PROJECT_ROOT` from `src.benchmark.injecagent`. Add `newline="\n"` (D, Windows/WSL hash parity). Checkpoint: state dict + primitives only, `torch.save`; load with `torch.load(..., weights_only=True)`. Re-run `assert_clean` over the loaded JSONL before training. Entry point `python -m src.scratch_llm.train` with `argparse`.

---

### `src/scratch_llm/inference.py` (service, request-response)

**Analog:** contract at `src/llm/scratch.py:25-31`; style of `src/llm/scripted.py` (stateless, plan a function of messages).

Contract: `load_generator(path) -> callable(messages, temperature=0.0, max_tokens=512) -> str`. Import only `format` and `model` (never `dataset`; keeps the held-out list away from eval code). Raise `RuntimeError` on any format_version / vocab hash / fingerprint / context mismatch (the runner turns provider-construction exceptions into `[skip]`). Return raw text, no JSON repair; stop at `"\n"`, cap 320 chars; fp32; no KV cache. Callable object exposes `identity`, `last_prompt_chars`, `last_new_chars`, `last_finish_reason`.

---

### `src/llm/scratch.py` (provider, request-response) - modify

**Analog:** `src/llm/openai_compat.py` lines 51-62.

Current `generate` (scratch.py:28-35) returns `LLMResponse(text=..., latency_ms=..., model=self.name)`. Extend to fill stamps exactly like the compat provider:
```python
return LLMResponse(
    text=text.strip(),
    latency_ms=_elapsed_ms(start),
    model=self.name,
    prompt_tokens=getattr(usage, "prompt_tokens", None),
    completion_tokens=getattr(usage, "completion_tokens", None),
    finish_reason=finish_reason,
    resolved_model=getattr(resp, "model", None),
)
```
For scratch: `prompt_tokens` = prompt chars, `completion_tokens` = new chars, `finish_reason` "stop"/"length", `resolved_model = f"scratch-gpt@{sha256[:12]}"`. Keep errors-as-values (`LLMResponse(text="", ..., error=f"{type(e).__name__}: {e}")`, scratch.py:32-34); `generate` never raises. Fix the stale docstring/comment "Phase 4" -> Phase 2 and keep the lazy `from src.scratch_llm.inference import load_generator` inside `__init__` plus the missing-checkpoint `RuntimeError` (scratch.py:20-24).

---

### Tests (test, various)

**Style analog:** `tests/test_control_agent.py` and `tests/test_report.py`.

Header and conventions (test_control_agent.py:1-30; test_report.py:1-17): module docstring stating what is covered and that no network/GPU is used; absolute `from src....` imports; `import pytest`; module-scoped fixtures; helpers above tests under a `# ── helpers ──` comment; synthetic data, outputs only under `tmp_path`.
```python
@pytest.fixture(scope="module")
def specs():
    return load_tool_specs()
```
- Real-loop parity test: copy the `drive()` helper idea (test_control_agent.py:39-50): `messages = build_messages(case.prompt, case.tools)`, call `provider.generate(messages)`, `parse_action(reply.text)`; for Phase 2 use a stub provider returning gold actions with `AgentLoop` and capture each `messages` passed to `generate`.
- Platform guard pattern: `linux_only = pytest.mark.skipif(not sys.platform.startswith("linux") or shutil.which("unshare") is None, reason=...)` (only needed for the smoke benchmark test).
- Planted-violation tests (leakage): build minimal episodes with `make_row`-style helper `make_episode(**over)` that returns a valid dict then applies overrides (test_report.py:19-30), one test per planted violation (8 listed in RESEARCH), plus a clean-set-passes test.
- Torch tests: keep tiny (2 layers, d_model 32, CPU), `pytest.importorskip("torch")`; dataset/format/gate tests must not import torch. Run targeted files per task (cold torch import ~54 s on DrvFs).

---

### `docs/viva/02-llm-foundations-and-synthetic-data.md` (docs)

**Analog:** `docs/viva/01-reference-monitor-and-evaluation.md` (lines 1-40 inspected).

Structure to copy: H1 `# Viva notes N: <topics>`; intro paragraph (for an OS viva, plain language, each idea first general then tied to a repo file, numbers copied from generated artifacts); "How to use this page" paragraph; bold glossary bullets (`- **Term**: ...`); `---`; `## Part A - ...` foundations with `###` subsections; Part B design in this repo; Part C results with numbers from `models/scratch/manifest.json` and dev metrics; "what we fixed / limits" section; `## Part D` likely examiner Q&A with model answers under a minute. Topics per CONTEXT: transformers, char tokenization, loss masking, AdamW/LR schedule, overfitting, decoding, synthetic data and leakage, worst-case principal. Plain ASCII punctuation style as in page 01 (page 01 uses an em dash in headings; keep consistent with that page).

`docs/viva/README.md` row to flip (line 15): change `| 02 LLM foundations and synthetic data | ... | Phase 2 | planned (Phase 2) |` to a link `[02-llm-foundations-and-synthetic-data.md](...)` with status `written (Phase 2)`, matching row 01's format (line 14).

---

### `.gitignore` / `requirements.txt` (config)

**Analog:** `.gitignore` existing block:
```
# Benchmark smoke runs (final results/ runs are committed)
results/smoke-*/
```
Add `models/**/*.pt`, `models/**/*.safetensors`, `models/**/*.gguf`, `data/local/*.jsonl` under a similar comment. `requirements.txt` line 23 has the stale `pip install torch --index-url .../cu124` comment: change to Phase 2, `cu128`, `torch>=2.6`.

## Shared Patterns

### Project conventions (`.claude/CLAUDE.md`, from RESEARCH)
**Apply to:** all new `src/` files: mandatory module docstring (see scripted.py:1-23 for tone: what it is, why it exists, what it does); `Optional[X]` not `X | None`; providers never raise from `generate`; no `sys.path.insert` in `src/`; never read/quote `.env`; decisions are string constants; manifests have fixed field lists.

### Provenance stamps
**Source:** `src/benchmark/runner.py:135-157, 235-257`. **Apply to:** `data/local/manifest.json`, `models/scratch/manifest.json`. Use `schema`, `code_commit`, `code_dirty`; atomic `.tmp` + `os.replace`.

### Errors-as-values in providers
**Source:** `src/llm/openai_compat.py:47-49`, `src/llm/scratch.py:32-34`. **Apply to:** `LocalScratchProvider.generate`.

### Single source of truth for prompt text
**Source:** `src/agent_harness/prompting.py` (`build_messages`, `OBSERVATION_TEMPLATE = "Result of your last action:\n{result}"`, `parse_action`). **Apply to:** dataset `episode_to_messages`, `render_context`, tests. Never hand-build the `T:`/`U:`/`A:`/`O:` strings in train.py.

### Determinism and line endings
**Apply to:** dataset, manifests: `random.Random(seed)`, `sorted()` pools, `encoding="utf-8"`, `newline="\n"`, ASCII-only output (non-ASCII becomes `?` in `encode`).

## No Analog Found

| File / piece | Role | Data Flow | Reason |
|---|---|---|---|
| training loop, LR schedule (LambdaLR warmup+cosine), token-budget batching, AdamW groups | script | batch | No training code exists in the repo; use RESEARCH "Training" section and `src/scratch_llm/model.py` (`GPTConfig.to_dict`, masked loss at model.py:105-108) |
| leakage gate, supervised mask, episode templates/value generators | utility | transform | New logic; use RESEARCH code examples and episode design |
| loss-curve PNG | utility | file-I/O | Nothing plots in the repo; matplotlib with `Agg` backend per RESEARCH |

## Metadata

**Analog search scope:** `src/llm`, `src/benchmark`, `src/scratch_llm`, `tests`, `docs/viva`, `.gitignore`, `requirements.txt`
**Files scanned:** about 14 read (RESEARCH lines 532-741 were not read; the first 531 lines covered all sections needed for mapping)
**Pattern extraction date:** 2026-10-21
