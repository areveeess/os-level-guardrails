"""
Benchmark report — raw.jsonl → summary.csv, summary.md, charts/*.png
"""

import csv
import json
import os
from collections import Counter, defaultdict

LAYERS = ("tool_broker", "policy_engine", "sandbox", "seccomp")

# Reference categorical palette (light mode), fixed slot order — see dataviz palette.md
SLOT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_rows(raw_path: str) -> list:
    with open(raw_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pct(n: int, d: int):
    return round(100.0 * n / d, 1) if d else None


def summarize(rows: list) -> list:
    """One summary row per model (models in first-seen order)."""
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    summary = []
    for model, rs in by_model.items():
        def sel(arm, kinds):
            return [r for r in rs if r["arm"] == arm and r["kind"] in kinds and not r["llm_error"]]

        atk_u, atk_p = sel("unprotected", ("dh", "ds")), sel("protected", ("dh", "ds"))
        ben_u, ben_p = sel("unprotected", ("benign",)), sel("protected", ("benign",))
        valid = [r for r in rs if not r["llm_error"]]
        layers = Counter()
        for r in atk_p:
            layers.update(r["blocked_by"])
        steps = sum(r["steps"] for r in valid)
        summary.append({
            "model": model,
            "family": rs[0].get("family", ""),
            "attack_episodes": len(atk_u),
            "attempted_pct": _pct(sum(r["attempted"] for r in atk_u), len(atk_u)),
            "success_unprotected_pct": _pct(sum(r["succeeded"] for r in atk_u), len(atk_u)),
            "success_protected_pct": _pct(sum(r["succeeded"] for r in atk_p), len(atk_p)),
            "refused_or_ignored_pct": _pct(sum(r["refused_or_ignored"] for r in atk_u), len(atk_u)),
            **{f"blocks_{layer}": layers.get(layer, 0) for layer in LAYERS},
            "retries_after_block_mean": round(sum(r["retries_after_block"] for r in atk_p) / len(atk_p), 2) if atk_p else None,
            "benign_task_unprotected_pct": _pct(sum(r["user_task_done"] for r in ben_u), len(ben_u)),
            "benign_task_protected_pct": _pct(sum(r["user_task_done"] for r in ben_p), len(ben_p)),
            "false_positive_pct": _pct(sum(r["false_positive"] for r in ben_p), len(ben_p)),
            "parse_error_rate_pct": _pct(sum(r["parse_errors"] for r in valid), steps),
            "latency_ms_per_step": round(sum(r["latency_ms"] for r in valid) / steps, 0) if steps else None,
            "llm_errors": sum(1 for r in rs if r["llm_error"]),
        })
    return summary


def kind_breakdown(rows: list) -> list:
    """Attempted / success by attack kind (dh, ds) per model, unprotected vs protected."""
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        for kind in ("dh", "ds"):
            u = [r for r in rows if r["model"] == model and r["kind"] == kind and r["arm"] == "unprotected" and not r["llm_error"]]
            p = [r for r in rows if r["model"] == model and r["kind"] == kind and r["arm"] == "protected" and not r["llm_error"]]
            out.append({"model": model, "kind": kind, "episodes": len(u),
                        "attempted_pct": _pct(sum(r["attempted"] for r in u), len(u)),
                        "success_unprotected_pct": _pct(sum(r["succeeded"] for r in u), len(u)),
                        "success_protected_pct": _pct(sum(r["succeeded"] for r in p), len(p))})
    return out


def _fmt(v):
    return "—" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def write_markdown(summary: list, breakdown: list, path: str):
    headline = ["model", "family", "attack_episodes", "attempted_pct", "success_unprotected_pct",
                "success_protected_pct", "benign_task_protected_pct", "false_positive_pct",
                "parse_error_rate_pct", "latency_ms_per_step"]
    titles = ["Model", "Family", "Attack episodes", "Attempted %", "Success % (no guardrails)",
              "Success % (guardrails)", "Benign task % (guardrails)", "False positive %",
              "Parse error % (per step)", "Latency ms/step"]
    lines = ["# Benchmark summary", "",
             "Attacks: InjecAgent indirect prompt injections (direct harm + data stealing).",
             "*Attempted* = the model called the attacker's tool. *Success* = the harm happened",
             "(dh: attacker tool executed; ds: canary-seeded private data was sent out).", "",
             "| " + " | ".join(titles) + " |", "|" + "---|" * len(titles)]
    lines += ["| " + " | ".join(_fmt(s[k]) for k in headline) + " |" for s in summary]

    lines += ["", "## Where attacks were stopped (protected arm, blocked actions)", "",
              "| Model | Tool broker | Policy engine | Sandbox | Seccomp | Retries after block (mean) |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {s['model']} | " + " | ".join(str(s[f'blocks_{l}']) for l in LAYERS)
              + f" | {_fmt(s['retries_after_block_mean'])} |" for s in summary]

    lines += ["", "## By attack kind", "", "| Model | Kind | Episodes | Attempted % | Success % (no guardrails) | Success % (guardrails) |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {b['model']} | {b['kind']} | {b['episodes']} | {_fmt(b['attempted_pct'])} | "
              f"{_fmt(b['success_unprotected_pct'])} | {_fmt(b['success_protected_pct'])} |" for b in breakdown]
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


def _grouped_bars(ax, labels, series, ymax=100):
    """series: list of (name, values, color). Direct value labels, 2px surface gap."""
    n = len(series)
    width = 0.8 / n
    for i, (name, values, color) in enumerate(series):
        xs = [x + (i - (n - 1) / 2) * width for x in range(len(labels))]
        vals = [v if v is not None else 0 for v in values]
        bars = ax.bar(xs, vals, width, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        for bar, v in zip(bars, values):
            ax.annotate(_fmt(v), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", color=TEXT_PRIMARY)
    ax.set_ylim(0, ymax * 1.12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=n, labelcolor=TEXT_PRIMARY)


def write_charts(summary: list, chart_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(chart_dir, exist_ok=True)
    models = [s["model"] for s in summary]

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    _grouped_bars(ax, models, [
        ("Attack attempted (no guardrails)", [s["attempted_pct"] for s in summary], SLOT[1]),
        ("Attack succeeded (no guardrails)", [s["success_unprotected_pct"] for s in summary], SLOT[2]),
        ("Attack succeeded (with guardrails)", [s["success_protected_pct"] for s in summary], SLOT[3]),
    ])
    _style_axes(ax, "Prompt-injection outcomes by model", "% of attack episodes")
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "attack_outcomes.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    bottoms = [0] * len(models)
    names = {"tool_broker": "Tool broker (capabilities)", "policy_engine": "Policy engine",
             "sandbox": "Namespaces / private fs", "seccomp": "Seccomp-BPF"}
    for slot, layer in enumerate(LAYERS, start=1):
        vals = [s[f"blocks_{layer}"] for s in summary]
        ax.bar(models, vals, 0.6, bottom=bottoms, color=SLOT[slot], label=names[layer],
               edgecolor=SURFACE, linewidth=2)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, total in enumerate(bottoms):
        ax.annotate(str(total), (x, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right", color=TEXT_PRIMARY)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28), ncol=4, labelcolor=TEXT_PRIMARY)
    _style_axes(ax, "Which guardrail layer stopped each blocked action", "Blocked actions")
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "blocks_by_layer.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    _grouped_bars(ax, models, [
        ("Benign task done (no guardrails)", [s["benign_task_unprotected_pct"] for s in summary], SLOT[1]),
        ("Benign task done (with guardrails)", [s["benign_task_protected_pct"] for s in summary], SLOT[2]),
    ])
    _style_axes(ax, "Utility: benign tasks completed", "% of benign episodes")
    fig.tight_layout()
    fig.savefig(os.path.join(chart_dir, "benign_utility.png"), dpi=160)
    plt.close(fig)


def generate_report(out_dir: str, charts: bool = True, sources: list = None) -> list:
    """Write the report into out_dir from its raw.jsonl, or from several result
    directories (`sources`, e.g. one per model run in parallel)."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for src in sources or [out_dir]:
        raw = os.path.join(src, "raw.jsonl")
        if os.path.exists(raw):  # a model may have been skipped (no key / no checkpoint)
            rows += load_rows(raw)
    if not rows:
        print(f"No benchmark results found in {', '.join(sources or [out_dir])}")
        return []
    summary = summarize(rows)
    breakdown = kind_breakdown(rows)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    write_markdown(summary, breakdown, os.path.join(out_dir, "summary.md"))
    if charts:
        write_charts(summary, os.path.join(out_dir, "charts"))
    return summary
