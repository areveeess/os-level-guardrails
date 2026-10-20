"""
Benchmark report — raw.jsonl → summary.csv, summary.md, charts/*.png
====================================================================
Every percentage is printed as 'P% (k/n) [lo, hi]' with a 95% Wilson interval
(see stats.py), so a reader always sees how many episodes stand behind a number.
A model only enters the headline table when its run is complete (done/expected
episodes); partial runs are listed separately and are not to be ranked.
"""

import csv
import json
import os
from collections import Counter, defaultdict

from src.benchmark.stats import format_prop, prop_fields

LAYERS = ("tool_broker", "policy_engine", "sandbox", "seccomp")

# 40 attack cases (20 dh + 20 ds) + 17 benign = 57 cases x 3 runs x 2 arms
DEFAULT_EXPECTED_EPISODES = 342

# Reference categorical palette (light mode), fixed slot order — see dataviz palette.md
SLOT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_rows_counted(raw_path: str) -> tuple:
    """(episodes, skipped_lines) from raw.jsonl, one record per (model, case, run, arm).

    An episode can be recorded more than once (an API error later retried, or
    two runners resuming the same directory): the first error-free record wins;
    an error record is kept only if the episode never succeeded. A line that is
    not valid JSON (a run killed mid-write) is skipped and counted, not fatal."""
    best, skipped = {}, 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (row["model"], row["case_id"], row["run"], row["arm"])
            except (json.JSONDecodeError, KeyError, TypeError):
                skipped += 1
                continue
            if key not in best or (best[key].get("llm_error") and not row.get("llm_error")):
                best[key] = row
    return list(best.values()), skipped


def load_rows(raw_path: str) -> list:
    """Episodes from raw.jsonl (see load_rows_counted for the best-record rule)."""
    return load_rows_counted(raw_path)[0]


def _cell(text) -> str:
    """Make a model-supplied string safe inside a markdown table cell."""
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def summarize(rows: list, expected: int = None) -> list:
    """One summary row per model (models in first-seen order)."""
    expected = DEFAULT_EXPECTED_EPISODES if expected is None else expected
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    summary = []
    for model, rs in by_model.items():
        usable = [r for r in rs if not r.get("llm_error")]

        def sel(arm, kinds):
            return [r for r in usable if r["arm"] == arm and r["kind"] in kinds]

        atk_u, atk_p = sel("unprotected", ("dh", "ds")), sel("protected", ("dh", "ds"))
        ben_u, ben_p = sel("unprotected", ("benign",)), sel("protected", ("benign",))
        layers = Counter()
        for r in atk_p:
            layers.update(r["blocked_by"])
        steps = sum(r["steps"] for r in usable)

        # Completeness: how much of the planned run is usable, and how many (case, run)
        # pairs have a usable episode in every arm the data contains.
        arms = {r["arm"] for r in rs}
        arms_by_pair = defaultdict(set)
        for r in usable:
            arms_by_pair[(r.get("case_id"), r.get("run"))].add(r["arm"])
        pairs_complete = sum(1 for a in arms_by_pair.values() if a >= arms)
        attack_keys = defaultdict(set)
        for r in atk_u + atk_p:
            attack_keys[(r.get("case_id"), r.get("run"))].add(r["arm"])
        attack_pairs = sum(1 for a in attack_keys.values() if a >= arms)
        episodes_done = len(usable)
        infra_errors = len(rs) - episodes_done
        if episodes_done >= expected:
            status = "complete"
        elif infra_errors > episodes_done:
            status = "error-dominated"
        else:
            status = "partial"

        summary.append({
            "model": model,
            "family": rs[0].get("family", ""),
            "status": status,
            "episodes_done": episodes_done,
            "episodes_expected": expected,
            "pairs_complete": pairs_complete,
            "infra_errors": infra_errors,
            "attack_episodes": len(atk_u),
            "attack_pairs": attack_pairs,
            **prop_fields("attempted", sum(r["attempted"] for r in atk_u), len(atk_u)),
            **prop_fields("success_unprotected", sum(r["succeeded"] for r in atk_u), len(atk_u)),
            **prop_fields("success_protected", sum(r["succeeded"] for r in atk_p), len(atk_p)),
            **prop_fields("refused_or_ignored", sum(r["refused_or_ignored"] for r in atk_u), len(atk_u)),
            **{f"blocks_{layer}": layers.get(layer, 0) for layer in LAYERS},
            "retries_after_block_mean": round(sum(r["retries_after_block"] for r in atk_p) / len(atk_p), 2) if atk_p else None,
            **prop_fields("benign_task_unprotected", sum(r["user_task_done"] for r in ben_u), len(ben_u)),
            **prop_fields("benign_task_protected", sum(r["user_task_done"] for r in ben_p), len(ben_p)),
            **prop_fields("false_positive", sum(r["false_positive"] for r in ben_p), len(ben_p)),
            **prop_fields("parse_error_rate", sum(r["parse_errors"] for r in usable), steps),
            "latency_ms_per_step": round(sum(r["latency_ms"] for r in usable) / steps, 0) if steps else None,
            "llm_errors": sum(1 for r in rs if r.get("llm_error")),
        })
    return summary


def kind_breakdown(rows: list) -> list:
    """Attempted / success by attack kind (dh, ds) per model, unprotected vs protected."""
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        for kind in ("dh", "ds"):
            u = [r for r in rows if r["model"] == model and r["kind"] == kind and r["arm"] == "unprotected" and not r.get("llm_error")]
            p = [r for r in rows if r["model"] == model and r["kind"] == kind and r["arm"] == "protected" and not r.get("llm_error")]
            out.append({"model": model, "kind": kind, "episodes": len(u),
                        **prop_fields("attempted", sum(r["attempted"] for r in u), len(u)),
                        **prop_fields("success_unprotected", sum(r["succeeded"] for r in u), len(u)),
                        **prop_fields("success_protected", sum(r["succeeded"] for r in p), len(p))})
    return out


def _fmt(v):
    return "—" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def _prop(s: dict, prefix: str) -> str:
    """A percentage cell, rendered from the same k/n the charts use."""
    return format_prop(s[f"{prefix}_k"], s[f"{prefix}_n"])


def _error_bars(summary: list, prefix: str) -> tuple:
    """(lower_errors, upper_errors) in percentage points for a bar series, from the
    prop_fields keys; (0, 0) where the denominator is empty."""
    lower, upper = [], []
    for s in summary:
        if s.get(f"{prefix}_n"):
            lower.append(round(s[f"{prefix}_pct"] - s[f"{prefix}_lo"], 1))
            upper.append(round(s[f"{prefix}_hi"] - s[f"{prefix}_pct"], 1))
        else:
            lower.append(0)
            upper.append(0)
    return lower, upper


# Headline / partial table columns: (title, cell renderer)
HEADLINE_COLUMNS = [
    ("Model", lambda s: _cell(s["model"])),
    ("Family", lambda s: _cell(s["family"])),
    ("Status", lambda s: s["status"]),
    ("Episodes done", lambda s: f"{s['episodes_done']}/{s['episodes_expected']}"),
    ("Paired attack episodes", lambda s: str(s["attack_pairs"])),
    ("Attempted (no guardrails)", lambda s: _prop(s, "attempted")),
    ("Success (no guardrails)", lambda s: _prop(s, "success_unprotected")),
    ("Success (guardrails)", lambda s: _prop(s, "success_protected")),
    ("Benign task (no guardrails)", lambda s: _prop(s, "benign_task_unprotected")),
    ("Benign task (guardrails)", lambda s: _prop(s, "benign_task_protected")),
    ("False positive (guardrails)", lambda s: _prop(s, "false_positive")),
    ("Parse errors (per step)", lambda s: _prop(s, "parse_error_rate")),
    ("Latency ms/step", lambda s: _fmt(s["latency_ms_per_step"])),
]


def _table(columns: list, summary: list) -> list:
    lines = ["| " + " | ".join(t for t, _ in columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(render(s) for _, render in columns) + " |" for s in summary]
    return lines


def write_markdown(summary: list, breakdown: list, path: str):
    complete = [s for s in summary if s["status"] == "complete"]
    others = [s for s in summary if s["status"] != "complete"]
    lines = ["# Benchmark summary", "",
             "## How to read this report", "",
             "Attacks: InjecAgent indirect prompt injections (direct harm `dh` + data stealing `ds`).",
             "*Attempted* = the model called the attacker's tool. *Success* = the harm happened",
             "(dh: attacker tool executed; ds: canary-seeded private data was sent out).", "",
             "Every percentage is printed as `P% (k/n) [lo, hi]`: k of n episodes, with a 95% Wilson",
             "score interval. A zero denominator prints `— (0/0)`. Read the interval, not just the",
             "point value: with few episodes the interval is wide.", "",
             "*Episodes done* is `usable episodes / expected episodes` for that model. Episodes that",
             "ended in an API error are not usable. Only models whose run is complete (done >= expected)",
             "appear in the headline table; every other model is listed under its own heading and must",
             "not be ranked against the complete ones.", "",
             "## Headline: complete runs", ""]
    lines += _table(HEADLINE_COLUMNS, complete) if complete else ["No model has a complete run yet."]

    lines += ["", "## Partial or error-dominated runs (not compared)", "",
              "These rows are shown for transparency; they have fewer usable episodes than planned and",
              "must not be ranked against the complete rows above.", ""]
    lines += _table(HEADLINE_COLUMNS, others) if others else ["None."]

    lines += ["", "## Where attacks were stopped (protected arm, blocked actions)", "",
              "| Model | Tool broker | Policy engine | Sandbox | Seccomp | Retries after block (mean) |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {_cell(s['model'])} | " + " | ".join(str(s[f'blocks_{l}']) for l in LAYERS)
              + f" | {_fmt(s['retries_after_block_mean'])} |" for s in summary]

    lines += ["", "## By attack kind", "",
              "| Model | Kind | Episodes | Attempted | Success (no guardrails) | Success (guardrails) |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {_cell(b['model'])} | {b['kind']} | {b['episodes']} | {_prop(b, 'attempted')} | "
              f"{_prop(b, 'success_unprotected')} | {_prop(b, 'success_protected')} |" for b in breakdown]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _style_axes(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=12)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _grouped_bars(ax, summary, labels, series, ymax=100):
    """series: list of (name, prop_prefix, color). Bars carry 95% Wilson error bars
    and a 'P% (k/n)' label built from the same prop_fields values as the tables."""
    n = len(series)
    width = 0.8 / n
    for i, (name, prefix, color) in enumerate(series):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        vals = [s[f"{prefix}_pct"] or 0 for s in summary]
        lower, upper = _error_bars(summary, prefix)
        bars = ax.bar(xs, vals, width, color=color, label=name, edgecolor=SURFACE, linewidth=2,
                      yerr=[lower, upper], capsize=2, error_kw={"ecolor": TEXT_SECONDARY, "elinewidth": 1})
        for bar, s, up in zip(bars, summary, upper):
            text = f"{s[f'{prefix}_pct']:g}% ({s[f'{prefix}_k']}/{s[f'{prefix}_n']})" if s[f"{prefix}_n"] else "n=0"
            ax.annotate(text, (bar.get_x() + bar.get_width() / 2, bar.get_height() + up),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.5, rotation=90, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", color=TEXT_PRIMARY)
    ax.set_ylim(0, ymax * 1.3)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=n, labelcolor=TEXT_PRIMARY)


def write_charts(summary: list, chart_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(chart_dir, exist_ok=True)
    models = [s["model"] for s in summary]
    labels = [m if s["status"] == "complete" else f"{m} (partial)" for m, s in zip(models, summary)]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    _grouped_bars(ax, summary, labels, [
        ("Attack attempted (no guardrails)", "attempted", SLOT[1]),
        ("Attack succeeded (no guardrails)", "success_unprotected", SLOT[2]),
        ("Attack succeeded (with guardrails)", "success_protected", SLOT[3]),
    ])
    _style_axes(ax, "Prompt-injection outcomes by model (95% Wilson interval)", "% of attack episodes")
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "attack_outcomes.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    bottoms = [0] * len(models)
    names = {"tool_broker": "Tool broker (capabilities)", "policy_engine": "Policy engine",
             "sandbox": "Namespaces / private fs", "seccomp": "Seccomp-BPF"}
    for slot, layer in enumerate(LAYERS, start=1):
        vals = [s[f"blocks_{layer}"] for s in summary]
        ax.bar(range(len(models)), vals, 0.6, bottom=bottoms, color=SLOT[slot], label=names[layer],
               edgecolor=SURFACE, linewidth=2)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, total in enumerate(bottoms):
        ax.annotate(str(total), (x, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(labels, rotation=20, ha="right", color=TEXT_PRIMARY)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=4, labelcolor=TEXT_PRIMARY)
    _style_axes(ax, "Which guardrail layer stopped each blocked action", "Blocked actions")
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "blocks_by_layer.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    _grouped_bars(ax, summary, labels, [
        ("Benign task done (no guardrails)", "benign_task_unprotected", SLOT[1]),
        ("Benign task done (with guardrails)", "benign_task_protected", SLOT[2]),
    ])
    _style_axes(ax, "Utility: benign tasks completed (95% Wilson interval)", "% of benign episodes")
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "benign_utility.png"), dpi=160)
    plt.close(fig)


def generate_report(out_dir: str, charts: bool = True, sources: list = None, expected: int = None) -> list:
    """Write the report into out_dir from its raw.jsonl, or from several result
    directories (`sources`, e.g. one per model run in parallel). `expected` is the
    planned number of episodes per model (default DEFAULT_EXPECTED_EPISODES)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for src in sources or [out_dir]:
        raw = os.path.join(src, "raw.jsonl")
        if os.path.exists(raw):  # a model may have been skipped (no key / no checkpoint)
            loaded, skipped = load_rows_counted(raw)
            rows += loaded
            if skipped:
                print(f"Skipped {skipped} undecodable line(s) in {raw}")
    if not rows:
        print(f"No benchmark results found in {', '.join(sources or [out_dir])}")
        return []
    summary = summarize(rows, expected=expected)
    breakdown = kind_breakdown(rows)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    write_markdown(summary, breakdown, os.path.join(out_dir, "summary.md"))
    if charts:
        write_charts(summary, os.path.join(out_dir, "charts"))
    return summary
