# Feature Research

**Domain:** Multi-model evaluation write-up for an agent prompt-injection defense (reference monitor / guardrails), including weak and local models
**Researched:** 2026-10-20
**Confidence:** MEDIUM. AgentDojo metrics, plots, CI method and defense table are verified against the paper's HTML (HIGH). ASB metric names (ASR, RR, PNA, NRP) are verified from search snippets (MEDIUM). InjecAgent (ASR-valid / ASR-all, DH/DS split, base vs enhanced prompt) and CaMeL (77% vs 84% tasks solved with provable security on AgentDojo) are partly from the abstract and prior knowledge, because full-text fetches timed out (MEDIUM). The Wilson-interval arithmetic is computed here and is exact.

## Feature Landscape

"Features" here means the report content a reader (review panel, viva examiner) expects: metrics, tables, charts, analyses, and the surrounding theory notes.

### Table Stakes (Reviewers Expect These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Unprotected vs protected arm, same cases/seeds/prompt | Every paper (InjecAgent, AgentDojo, ASB, CaMeL) shows "no defense" next to "defense". Without the baseline, "0% success" is meaningless | LOW | Already present. Keep `prompting.py` as the single source of truth and state it in the report |
| Attack Success Rate (ASR), unprotected and protected | The headline security metric in all benchmarks. AgentDojo calls it targeted ASR; ASB and InjecAgent call it ASR | LOW | Already present as Success %. Define precisely: dh = attacker tool executed, ds = canary data left the sandbox |
| Attempted rate (model chose the attacker action), separate from success | Needed to tell "model resisted" from "defense blocked". ASB counts a case as an ASR hit once the attacker tool is invoked. Your split is more informative | LOW | Already present. Make attempted/success/blocked a funnel: attempted -> would-have-succeeded -> blocked |
| Benign utility (task completion, no attack) with and without the defense | AgentDojo's benign utility and ASB's PNA. The cost-of-defense number. CaMeL's headline is a utility gap (77% provably secure vs 84% undefended) | LOW | Already present as "benign task %". Add an unprotected-arm benign column so the delta is visible; today only the guardrails column exists |
| Utility under attack (user task done AND no attacker side effect) | The third AgentDojo metric. Shows the agent still works when injected | MEDIUM | Not yet reported. Derivable from existing episode logs if user-task success is recorded per attacked episode |
| False positive rate on benign tasks | Needed to answer "does the guardrail block legitimate work". AgentDojo shows a PI detector cutting GPT-4o benign utility from 69% to 41% through false positives | LOW | Already present. Report the numerator and denominator, not just a percentage |
| Per-model rows with identical columns, one table | The standard cross-model table. Gemini is one row among many | LOW | Already present. Keep consistent row ordering (by size/locality), mark closed vs open and local vs hosted |
| Attack breakdown by kind (direct harm vs data stealing) | InjecAgent's core split. ds needs a private-data source and an exfil sink, so it behaves differently from dh | LOW | Already present. Hosted models lack ds rows; fill them or show them as explicit "not run" |
| Uncertainty: n per cell and a 95% CI | AgentDojo reports 95% CIs (statsmodels `proportion_confint`). With 5 to 120 episodes per cell, bare percentages mislead | LOW | Use Wilson intervals for proportions. Exact numbers for your table: 0/47 gives an upper bound of about 7.6%, 0/55 about 6.5%, 0/50 about 7.1%, 0/120 about 3.1%. "0%" on small n is "not detected", not "proved zero" |
| Parse/format error rate and invalid-output handling | InjecAgent separates ASR-valid (among well-formed outputs) from ASR-all. Weak models emit malformed actions, which silently lowers ASR | LOW | Already present (per step). State whether a parse error counts as safe, retried, or excluded, and report both ASR-valid and ASR-all |
| Episode completeness disclosure | Several rows (gemma 0 attack episodes, gpt-oss-20b only 5) are incomplete. Reviewers penalise silent denominators | LOW | Add "episodes planned / completed / API-failed" per model, and grey out or exclude rows with too few attack episodes |
| Where-blocked breakdown by layer | The system's own claim is defense in depth. A table of tool broker / policy / sandbox / seccomp is expected from a systems paper | LOW | Already present. Flag that all non-broker layers are near 0 in the current data, which is a finding that needs explaining, not hiding |
| Latency / overhead of the defense | OS-security reviewers always ask "what does it cost". CaMeL reports token overhead | LOW | Latency per step exists. Add protected vs unprotected overhead, since the current column mixes model latency with defense latency |
| Threat model and metric definitions in the report | Every benchmark paper opens with them. The viva will start here | LOW | One half-page: attacker capability, trusted vs untrusted inputs, what counts as success, what is out of scope |
| Reproducibility statement (seed, model versions, dates, run command) | Hosted models drift. Gemma and Groq model names change | LOW | Record model ID, provider, date, temperature, seeds in the result bundle |

### Differentiators (Competitive Advantage)

Aligned with the Core Value: the guardrails hold regardless of which LLM drives the agent, including a weak, unaligned local model.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Model-agnostic claim made explicit and tested: protected-arm success is ~0 across every model, whatever the unprotected rate | This is the project's thesis. CaMeL makes the same claim ("secure even when underlying models are susceptible"). A forest-style plot of protected ASR with CIs per model, next to unprotected ASR, makes it visible in one glance | MEDIUM | Needs the funnel (attempted -> executed-if-unguarded -> blocked) so the claim rests on attempted cases, not on models that never attempted |
| Conditional block rate: blocked / attempted, per model, with CI | Models that never attempt an attack (qwen3.8 at 0% attempted) say nothing about the defense. Conditioning on attempts is the honest efficacy number | LOW | Pool attempts across models and layers for statistical power. Report n_attempts explicitly |
| Capability-vs-vulnerability scatter: x = benign utility (or model size), y = unprotected ASR, one point per model | AgentDojo Figure 6(a) found more capable models easier to attack (inverse scaling). Showing where ScratchGPT, Qwen-0.5B LoRA, 3B local models and 120B hosted models fall tells the "spectrum" story the project wants | MEDIUM | Needs the scratch and LoRA rows to exist. Annotate points; do not draw a trend line through 6 to 10 points. Say "consistent with", not "proves" |
| Utility-vs-ASR Pareto plot for the system (unprotected vs protected, per model) | AgentDojo Figure 9 bolds Pareto-optimal defenses. Shows protection costs little utility. One arrow per model from unprotected to protected | MEDIUM | Needs unprotected benign utility (see table stakes) |
| Weak-model worst-case principal: ScratchGPT and LoRA Qwen as adversarially-prone drivers | The reviewer's ask. A model with no alignment that obeys injections should maximise attempted rate, which stress-tests the defense where hosted models rarely attempt. Likely the strongest evidence in the report | HIGH | Depends on the training work. Report attempted rate for these models as the key number. If a weak model attempts rarely because it is incompetent, say so, and consider a forced-injection arm (below) |
| Forced-attempt / "oracle attacker" arm: replay the injected attacker action regardless of what the model chose | Removes dependence on model gullibility, so the defense is tested on 100% of attacker actions. Analogous to AgentDojo's adaptive-attack framing and to testing the monitor independent of the LLM | MEDIUM | Cheap, deterministic, no API quota. It turns "0% of ~5 attempts" into "0% of N attacks" and is arguably the cleanest proof of model independence |
| Where-blocked by layer, with ablation (broker only, +policy, +sandbox, +seccomp) | Shows each layer's marginal contribution, which is the systems-security contribution. Current logs show nearly everything stopped at the broker, so ablation exposes whether other layers matter | MEDIUM | Add a small synthetic suite hitting the policy, sandbox and seccomp layers directly (shell chaining `ls; curl`, chmod, mount, ptrace) so those layers have non-zero counts |
| Shell-chaining bypass reproduced, fixed, and reported as a regression test | A found-and-fixed bypass is a credible security result and a strong viva point. Honest limitations section builds trust | LOW | Already in Active requirements. Report before/after on a small adversarial suite |
| Retries-after-block (adaptive behaviour) | Measures whether a blocked model tries again or another route (partial adaptive attacker). Already reported, which most papers lack | LOW | Keep. Extend to "alternate route found?" for weak models |
| Failure taxonomy for the unprotected arm: why attacks failed (refused, malformed, wrong tool, step limit) | Explains the low attempted rates and separates alignment from incompetence. Helpful for weak models | MEDIUM | Derive from existing logs. A stacked bar per model |
| Per-case and clustered uncertainty | The same case run multiple times is not independent. Cluster bootstrap or per-case aggregation avoids overstated CIs | MEDIUM | Do if time allows. At minimum say episodes within a case are correlated |
| Paired comparison across arms (McNemar or just discordant-pair counts) | Same case in both arms is a paired design, so report "N cases succeeded unprotected and were blocked, 0 cases the reverse" | LOW | Strong, simple, defendable. Prefer counts over p-values |
| Cost/footprint row for local models (VRAM, tokens/s, params) | Makes the local-model story concrete (6 GB GPU). Supports the claim that the defense is cheap relative to the model | LOW | One extra column set in the model table |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| A single "security score" combining ASR and utility (ASB's NRP-style composite) | Compact ranking | Hides the trade-off, invites questions about the weighting, hard to defend in a viva | Report ASR and utility side by side plus the Pareto plot |
| Ranking models as "best/worst LLM" | Natural reading of a comparison table | The project compares defense behaviour across drivers, not model quality. Incomplete episodes make rankings unsound | Group by family/size/locality, and say the table is not a leaderboard |
| Headline numbers from rows with few or no attack episodes (gemma 0, gpt-oss-20b 5) | Looks complete | Percentages from n=5 or an all-timeout run are noise or fake | Show n, show CIs, grey out or exclude, and say why |
| Claiming "0% success proves the defense works" | Satisfying result | 0/47 is only an upper bound of about 7.6% (95%). Models with 0% attempted give no information about the guardrails | Use conditional block rate, forced-attempt arm, and upper bounds |
| Changing benchmark or adding AgentDojo/ASB runs | Cross-benchmark credibility | Out of scope: week deadline, would break comparability with the committed run | Cite them as related work and map your metrics to theirs in a table |
| Training or tuning on InjecAgent tools/cases for the local models | Raises attempted rate for weak models | Contaminates evaluation (already ruled out). Would make the weak model an artificially tuned attacker victim | Keep held-out tools. Fine-tune only on the action format |
| Learned or LLM-based detector layers (PI detector) | Seen in AgentDojo as a baseline | Introduces false positives (AgentDojo: benign utility 69% to 41%), non-determinism and a model dependency that undermines the model-agnostic claim | Keep the deterministic reference monitor. Discuss PI detectors only as related work |
| Many fancy interactive dashboards/charts for the report | Presentation polish | Time sink in a one-week window. Dashboard already exists and has a debug-on-0.0.0.0 concern | Five static charts (below) exported to PNG/PDF for the deck |
| Mixing runs from different prompt/seed/model-version configurations in one cell | Fills gaps from resumed runs | Violates the fairness constraint. Silent drift between runs | Tag each episode with config hash. Merge only identical hashes |
| Pretraining-scale ambitions for ScratchGPT (claiming it is a capable agent) | Appears impressive | Out of scope and misleading. Its role is a weak worst-case principal | Describe it honestly: tiny, unaligned, format-conditioned |

## Feature Dependencies

```
Episode completeness (resume hosted runs + ds episodes)
    └──requires──> stable run config + resumable runner (exists)
                       └──enables──> per-model table with CIs

Per-model table with Wilson CIs
    └──requires──> n per cell + defined metric semantics (attempted / success / valid)

Conditional block rate (blocked / attempted)
    └──requires──> attempted flag logged per episode (exists)
Forced-attempt arm ──enhances──> Conditional block rate (more attempts, same defense)

Capability-vs-vulnerability scatter
    └──requires──> unprotected benign utility column
    └──requires──> scratch-gpt + LoRA Qwen + extra Ollama models benchmarked

Utility-vs-ASR Pareto plot
    └──requires──> unprotected benign utility + utility under attack

Layer ablation / non-zero layer counts
    └──requires──> shell-chaining fix + synthetic layer-targeted suite
    └──enhances──> where-blocked table

Viva theory notes ──enhances──> every report section (each phase adds a note)

Rankings / composite score ──conflicts──> thesis framing (defense is model-agnostic)
Training on InjecAgent tools ──conflicts──> held-out evaluation hygiene
```

### Dependency Notes

- **Per-model table requires episode completeness:** the existing run has only two models at 342 episodes and no hosted ds data. The comparison is only credible after resumes finish or after incomplete rows are marked.
- **Scatter and Pareto plots require the unprotected benign column:** currently only guardrails-arm benign task % is shown. Add it before building those plots.
- **Forced-attempt arm enhances conditional block rate:** it does not depend on any model, so it can be built in parallel with model training and gives a result even if the weak model underperforms.
- **Layer ablation requires the shell-chaining fix:** otherwise the policy layer's contribution cannot be shown cleanly, and the report would present a known bypass.
- **Local-model rows gate the capability story:** if ScratchGPT is not ready, the scatter still works with qwen2.5:3b, phi3.5:3.8b, gemma3:1b, llama3.2:3b and the hosted models.

## MVP Definition

### Launch With (v1)

Minimum to defend the thesis within the one-week deadline.

- [ ] Unified model table with unprotected vs protected ASR, attempted %, benign (both arms), FP %, parse error %, latency, plus n and 95% Wilson CIs — core credibility
- [ ] Episode completeness column and greyed-out incomplete rows (gemma, gpt-oss-20b) — honesty about denominators
- [ ] Conditional block rate (blocked / attempted) pooled and per model — turns near-zero ASR into evidence
- [ ] At least one weak local row beyond llama3.2:3b (scratch-gpt or LoRA Qwen, plus qwen2.5:3b) — directly answers the reviewer
- [ ] dh/ds breakdown with explicit "not run" cells — matches InjecAgent
- [ ] Where-blocked table plus a short explanation of why only the broker fires in the current data — avoids an unexplained gap
- [ ] Threat model, metric definitions, mapping of your metrics to AgentDojo/InjecAgent/ASB names — viva readiness
- [ ] Viva notes updated per phase in `docs/viva/` — project requirement

### Add After Validation (v1.x)

- [ ] Forced-attempt arm with N attacker actions per tool — add once the main table is stable
- [ ] Capability-vs-vulnerability scatter and Pareto plot — add once at least 6 models have complete rows
- [ ] Shell-chaining fix with before/after adversarial suite and layer ablation — add when the policy-engine phase lands
- [ ] Paired-case discordant counts — add when the final data is frozen
- [ ] Unprotected-arm failure taxonomy — add if time remains

### Future Consideration (v2+)

- [ ] Utility under attack on a larger task set, with cluster-bootstrap CIs — needs more cases
- [ ] AgentDojo-style adaptive attacks against the monitor — a separate research effort
- [ ] Cross-benchmark replication (AgentDojo, ASB) — out of scope for now

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Wilson CIs and n per cell | HIGH | LOW | P1 |
| Unprotected benign column | HIGH | LOW | P1 |
| Completeness / excluded rows | HIGH | LOW | P1 |
| Conditional block rate | HIGH | LOW | P1 |
| Local weak-model rows | HIGH | HIGH | P1 |
| Threat model + metric mapping | HIGH | LOW | P1 |
| Viva notes per phase | HIGH | MEDIUM | P1 |
| Forced-attempt arm | HIGH | MEDIUM | P2 |
| Capability-vs-vulnerability scatter | MEDIUM | MEDIUM | P2 |
| Utility-vs-ASR Pareto plot | MEDIUM | MEDIUM | P2 |
| Layer ablation + synthetic layer suite | MEDIUM | MEDIUM | P2 |
| Shell-chaining regression test and fix | HIGH | LOW | P1 |
| Utility under attack | MEDIUM | MEDIUM | P2 |
| Paired discordant counts | MEDIUM | LOW | P2 |
| Failure taxonomy | MEDIUM | MEDIUM | P3 |
| Cluster bootstrap | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Competitor Feature Analysis

| Feature | InjecAgent | AgentDojo | ASB | CaMeL | Our Approach |
|---------|------------|-----------|-----|-------|--------------|
| Attack success metric | ASR-valid and ASR-all (valid = well-formed output) | Targeted ASR over (user task, injection task) pairs | ASR (attacker tool invoked) | ASR on AgentDojo | Attempted and Success separately, add ASR-valid vs ASR-all |
| Clean utility | Not a focus | Benign utility | PNA (performance under no attack) | Tasks solved (77% provably secure vs 84% undefended) | Benign task %, both arms |
| Utility under attack | No | Yes | No (NRP composite in v1) | Implicit | Add, derived from logs |
| False positives of a defense | No | Shown through the PI detector's utility drop | Partly | No | Explicit FP % on benign tasks |
| Attack split | Direct harm vs data stealing; base vs enhanced prompt | By suite (workspace, travel, banking, slack) | By attack type (DPI, OPI, memory, PoT) | By suite | dh vs ds |
| Uncertainty | Typically none | 95% CIs (`proportion_confint`) | Typically none | Limited | Wilson CIs and n on every cell |
| Capability vs vulnerability | Observed (stronger models more vulnerable under the enhanced prompt) | Plot of benign utility vs ASR; inverse scaling | Backbone comparison | Not the point | Scatter including weak local models |
| Defense layer attribution | No | Compares defenses, not layers | Compares defenses | Single design | Where-blocked by layer, ablation |
| Model-agnostic demonstration | Models only | Many models, one defense each | Many models | Mostly a handful of frontier models | Many models including a from-scratch one, with a forced-attempt arm |

## Viva-prep theory notes (what `docs/viva/` should cover)

This is content rather than code, but the milestone asks for it. Each note should be 1 to 2 pages with a "likely examiner question" list.

**OS security foundations (BCSE303P core)**
- Reference monitor concept: complete mediation, tamper-proofness, verifiability (Anderson report). Map the broker, policy engine, sandbox and seccomp to these three properties.
- Access control models: access matrix, capabilities vs ACLs, DAC vs MAC, least privilege, confused deputy (an injected agent acts as a confused deputy).
- Process isolation: Linux namespaces (user, PID, net, mount), `unshare`, chroot vs namespaces, cgroups; what each does and does not isolate.
- seccomp-bpf: filter modes, BPF program, default-deny vs allow-list, why chmod/mount/ptrace/setuid are blocked, `no_new_privs`, limits (TOCTOU on pointer arguments).
- TOCTOU, command injection and shell metacharacter chaining (the `ls; curl` bypass), allow-list parsing vs shell semantics.
- Defense in depth, fail-closed vs fail-open, and what the tool broker adds that the OS layers cannot see (semantic intent).
- Contrast with eBPF/LSM approaches (ActPlane, AgentSight) as future work.

**LLM foundations**
- Transformer decoder, tokenisation, attention, next-token prediction, training loss, sampling temperature; what ScratchGPT is and is not.
- Why a tiny from-scratch model is "unaligned" and weak; SFT vs pretraining; LoRA (low-rank adaptation, rank, alpha, which weights, why it fits 6 GB).
- Agent loop: tool calling, JSON action protocol, observation feedback, context window, system prompt, why parse errors occur.
- Prompt injection: direct vs indirect, instruction/data confusion, why prompts alone cannot fix it (OWASP LLM01, Greshake, Perez & Ribeiro "Ignore Previous Prompt").
- Related defenses: CaMeL (capability-based, privileged vs quarantined LLM), spotlighting/delimiting, PI detectors, tool filtering, and how this project differs (enforcement below the LLM).

**Evaluation methodology**
- Metric definitions and their mapping (ASR, benign utility, utility under attack, FP rate, PNA); why attempted differs from success.
- Binomial proportions, Wilson interval vs Wald, why 0/n is not zero, rule of three (about 3/n).
- Paired design (same case, two arms), correlation among repeated runs of the same case, seeds and temperature.
- Threats to validity: small n, incomplete episodes, API timeouts, model version drift, benchmark contamination (held-out tools), a synthetic canary for data stealing, LLM-as-judge avoidance.
- Fairness: identical system prompt, seeds, arms; the compact rendering for the scratch model as an acknowledged deviation.
- Limits and honesty: InjecAgent covers a narrow attack style, non-adaptive attackers, simulated tools.

## Suggested chart set (five static figures)

1. Grouped bars: protected vs unprotected ASR per model with Wilson error bars (the model-agnostic figure).
2. Funnel / stacked bar: attempted -> would-succeed -> blocked, per model.
3. Scatter: benign utility (or params) vs unprotected ASR (capability vs vulnerability).
4. Arrow plot: benign utility vs ASR, unprotected to protected per model (cost of defense).
5. Stacked bar: where blocked by layer, with retries after block.

## Sources

- AgentDojo (Debenedetti et al.), arXiv 2406.13352, HTML version: benign utility, utility under attack, targeted ASR definitions; Figure 6 and 9 plots; 95% CIs via statsmodels `proportion_confint`; GPT-4o defense table (HIGH)
- CaMeL, "Defeating Prompt Injections by Design", arXiv 2503.18813, abstract: 77% of tasks provably secure vs 84% undefended on AgentDojo (HIGH for the number, MEDIUM for the rest)
- Agent Security Bench, arXiv 2410.02644 (search snippets): ASR, PNA, refusal rate, NRP; 8 metrics in v1 and 7 in v2 (MEDIUM)
- InjecAgent, arXiv 2403.02691: ASR-valid / ASR-all, DH vs DS, base vs enhanced prompt (MEDIUM; full-text fetch failed, details from prior knowledge, verify against the local PDF in `papers/` before citing)
- Project files: `.planning/PROJECT.md`, `results/final-20261011/summary.md`
- Wilson upper bounds computed as z^2/(n+z^2) for zero events at z=1.96 (exact)

---
*Feature research for: multi-model agent prompt-injection defense evaluation*
*Researched: 2026-10-20*
