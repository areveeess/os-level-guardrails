"""
Deliverables charts — the three D-05 figures, drawn from numbers.json only
=========================================================================
  asr_ci.png                       attack success per model with 95% Wilson intervals, without and
                                   with guardrails, the scripted control as a dashed reference line
  capability_vs_vulnerability.png  benign utility vs attack success, both without guardrails
  utility_vs_asr_arrows.png        one arrow per model from unprotected to protected

Static matplotlib PNGs (Agg backend, APIs of matplotlib 3.8 and later). Colour is never the only
cue: Okabe-Ito colours (validated for colour-vision deficiency) go with distinct marker shapes, every
point carries its k/n as text, a partial model is drawn hollow and labelled with done/expected, and
each footer states the data state and snapshot time. Legacy rows are never plotted (D-02): only the
current-engine models of numbers.json and the control's reference line are read.
"""

import os

from src.benchmark.report import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

CHART_FILES = ("asr_ci.png", "capability_vs_vulnerability.png", "utility_vs_asr_arrows.png")
CAPTIONS = {
    "asr_ci.png": "Attack success per model with 95% Wilson intervals, without and with guardrails; "
                  "dashed line: the scripted control without guardrails",
    "capability_vs_vulnerability.png": "Capability vs vulnerability: benign task done vs attack success, "
                                       "both without guardrails",
    "utility_vs_asr_arrows.png": "Each model from no guardrails (hollow) to guardrails (filled): "
                                 "benign task done vs attack success",
}

# Okabe-Ito
BLUE, ORANGE, GREEN, VERMILLION, PURPLE, SKY, BLACK = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                                                       "#56B4E9", "#000000")
ARM_STYLE = {"unprotected": (VERMILLION, "o", "Without guardrails"), "protected": (BLUE, "s", "With guardrails")}
GROUP_STYLE = {"local": (BLUE, "Local model"), "hosted": (ORANGE, "Hosted model (Groq)")}
MODEL_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LEADER = "#9a9994"
SIZE, DPI = (10, 5.6), 160


def chart_links() -> list:
    """(path relative to the frozen directory, caption) per chart, in CHART_FILES order."""
    return [(f"charts/{name}", CAPTIONS[name]) for name in CHART_FILES]


def _footer(numbers: dict) -> str:
    state = "snapshot" if numbers["data_state"] == "snapshot" else "final data"
    return (f"Data: {state} of {numbers['snapshot_display']} (UTC); D-09 freeze {numbers['freeze_display']}. "
            "Legacy October rows are not plotted.\n"
            "95% Wilson intervals per episode; repeats at temperature 0 are near-duplicates, so the intervals "
            "are optimistic")


def _name(m: dict) -> str:
    return m["model"] if m["complete"] else f"{m['model']} (partial {m['episodes_done']}/{m['episodes_expected']})"


def _kn(metric: dict) -> str:
    return f"{metric['k']}/{metric['n']}"


def _err(metric: dict) -> list:
    return [[metric["pct"] - metric["lo"]], [metric["hi"] - metric["pct"]]]


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)


def _not_plotted(m: dict, prefixes: tuple) -> str:
    """The note for a model with an empty axis (a partial run with no episodes of a kind yet)."""
    counts = ", ".join(f"{p.replace('_', ' ')} {_kn(m['metrics'][p])}" for p in prefixes)
    return f"Not plotted, an axis has no episodes yet: {_name(m)} ({counts})"


def _save(fig, plt, footer: str, path: str, notes: list = ()):
    for j, note in enumerate(notes):
        fig.text(0.01, 0.075 + 0.03 * j, note, fontsize=7, color=TEXT_SECONDARY, ha="left", va="bottom")
    fig.text(0.01, 0.015, footer, fontsize=7, color=TEXT_SECONDARY, ha="left", va="bottom")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def _asr_ci(plt, Line2D, numbers: dict, models: list, path: str) -> list:
    drawn = []
    fig, ax = plt.subplots(figsize=SIZE, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.36)
    for i, m in enumerate(models):
        for arm, offset in (("unprotected", -0.17), ("protected", 0.17)):
            color, marker, _ = ARM_STYLE[arm]
            metric = m["metrics"][f"success_{arm}"]
            if not metric["n"]:
                continue
            x = i + offset
            ax.errorbar([x], [metric["pct"]], yerr=_err(metric), fmt="none", ecolor=color, elinewidth=1.2, capsize=3)
            ax.plot([x], [metric["pct"]], linestyle="none", marker=marker, markersize=8, color=color,
                    markerfacecolor=color if m["complete"] else SURFACE, markeredgewidth=1.6)
            label = _kn(metric)
            ax.annotate(label, (x, metric["hi"]), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, rotation=90, color=TEXT_SECONDARY)
            drawn.append(label)
    handles = [Line2D([], [], linestyle="none", marker=ARM_STYLE[a][1], color=ARM_STYLE[a][0], markersize=8,
                      label=ARM_STYLE[a][2]) for a in ("unprotected", "protected")]
    handles.append(Line2D([], [], linestyle="none", marker="o", color=TEXT_SECONDARY, markerfacecolor=SURFACE,
                          markersize=8, label="Hollow marker: partial run (not final)"))
    control = next((m for m in numbers["models"] if m["group"] == "control"), None)
    if control and control["metrics"]["success_unprotected"]["n"]:
        metric = control["metrics"]["success_unprotected"]
        ref = f"scripted control, no guardrails: {metric['pct']:.1f}% ({_kn(metric)})"
        ax.axhline(metric["pct"], color=BLACK, linestyle="--", linewidth=1)
        ax.annotate(ref, (len(models) - 0.5, metric["pct"]), xytext=(0, -3), textcoords="offset points",
                    ha="right", va="top", fontsize=7.5, color=TEXT_PRIMARY)
        handles.append(Line2D([], [], color=BLACK, linestyle="--", linewidth=1, label=ref))
        drawn.append(ref)
    ticks = [_name(m).replace(" (partial", "\n(partial") for m in models]
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(ticks, rotation=25, ha="right", fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_ylim(-4, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.30), ncol=2, fontsize=7.5,
              labelcolor=TEXT_PRIMARY)
    _style(ax, "Attack success per model, without and with guardrails (95% Wilson intervals)", "",
           "% of attack episodes")
    ax.grid(axis="x", visible=False)
    drawn += [_name(m) for m in models] + [h.get_label() for h in handles if h.get_label() not in drawn]
    footer = _footer(numbers)
    drawn.append(footer)
    _save(fig, plt, footer, path)
    return drawn


def _column_labels(ax, points: list):
    """Labels in a column right of the axes, one slot per point in y order, with a leader line and the point's
    marker beside each label, so labels never overlap however close the points are."""
    points = sorted(points, key=lambda p: -p["y"])
    count = len(points)
    for j, p in enumerate(points):
        slot = 0.97 - j * (0.94 / max(count - 1, 1))
        ax.annotate("", xy=(p["x"], p["y"]), xycoords="data", xytext=(1.02, slot), textcoords="axes fraction",
                    arrowprops={"arrowstyle": "-", "color": LEADER, "linewidth": 0.5, "alpha": 0.7, "shrinkA": 0,
                                   "shrinkB": 5},
                    annotation_clip=False)
        ax.plot([1.035], [slot], transform=ax.transAxes, clip_on=False, linestyle="none", marker=p["marker"],
                markersize=6, color=p["color"], markerfacecolor=p["face"])
        ax.text(1.055, slot, p["label"], transform=ax.transAxes, fontsize=7, va="center", ha="left",
                color=TEXT_PRIMARY)


def _group_handles(Line2D, extra: list) -> list:
    handles = [Line2D([], [], linestyle="none", marker="o", color=c, markersize=7, label=label)
               for c, label in GROUP_STYLE.values()]
    return handles + extra


def _capability(plt, Line2D, numbers: dict, models: list, path: str) -> list:
    drawn, points, notes = [], [], []
    fig, ax = plt.subplots(figsize=SIZE, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.56, top=0.90, bottom=0.24)
    for i, m in enumerate(models):
        benign, attack = m["metrics"]["benign_task_unprotected"], m["metrics"]["success_unprotected"]
        if not benign["n"] or not attack["n"]:
            notes.append(_not_plotted(m, ("success_unprotected", "benign_task_unprotected")))
            continue
        color, marker = GROUP_STYLE[m["group"]][0], MODEL_MARKERS[i % len(MODEL_MARKERS)]
        face = color if m["complete"] else SURFACE
        ax.errorbar([benign["pct"]], [attack["pct"]], xerr=_err(benign), yerr=_err(attack), fmt="none",
                    ecolor=color, elinewidth=0.9, capsize=2, alpha=0.7)
        ax.plot([benign["pct"]], [attack["pct"]], linestyle="none", marker=marker, markersize=9, color=color,
                markerfacecolor=face, markeredgewidth=1.6)
        label = f"{_name(m)}\nbenign {_kn(benign)}, attack {_kn(attack)}"
        points.append({"x": benign["pct"], "y": attack["pct"], "label": label, "marker": marker, "color": color,
                       "face": face})
        drawn.append(label)
    _column_labels(ax, points)
    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 104)
    handles = _group_handles(Line2D, [Line2D([], [], linestyle="none", marker="o", color=TEXT_SECONDARY,
                                             markerfacecolor=SURFACE, markersize=7,
                                             label="Hollow marker: partial run (not final)")])
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3, fontsize=7.5,
              labelcolor=TEXT_PRIMARY)
    _style(ax, "Capability vs vulnerability (no guardrails)", "Benign task done (%)", "Attack success (%)")
    drawn += [h.get_label() for h in handles] + notes
    footer = _footer(numbers)
    drawn.append(footer)
    _save(fig, plt, footer, path, notes)
    return drawn


def _arrows(plt, Line2D, numbers: dict, models: list, path: str) -> list:
    drawn, points, notes = [], [], []
    fig, ax = plt.subplots(figsize=SIZE, facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.56, top=0.90, bottom=0.24)
    for i, m in enumerate(models):
        bu, bp = m["metrics"]["benign_task_unprotected"], m["metrics"]["benign_task_protected"]
        au, ap = m["metrics"]["success_unprotected"], m["metrics"]["success_protected"]
        if not all(x["n"] for x in (bu, bp, au, ap)):
            notes.append(_not_plotted(m, ("success_unprotected", "success_protected", "benign_task_unprotected",
                                          "benign_task_protected")))
            continue
        color, marker = GROUP_STYLE[m["group"]][0], MODEL_MARKERS[i % len(MODEL_MARKERS)]
        start, end = (bu["pct"], au["pct"]), (bp["pct"], ap["pct"])
        ax.annotate("", xy=end, xytext=start, xycoords="data",
                    arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.4, "shrinkA": 5, "shrinkB": 5,
                                "mutation_scale": 12})
        ax.plot([start[0]], [start[1]], linestyle="none", marker=marker, markersize=8, color=color,
                markerfacecolor=SURFACE, markeredgewidth=1.6)
        ax.plot([end[0]], [end[1]], linestyle="none", marker=marker, markersize=8, color=color)
        label = (f"{_name(m)}\nattack {_kn(au)} -> {_kn(ap)}, benign {_kn(bu)} -> {_kn(bp)}")
        points.append({"x": end[0], "y": end[1], "label": label, "marker": marker, "color": color, "face": color})
        drawn.append(label)
    _column_labels(ax, points)
    ax.set_xlim(-4, 104)
    ax.set_ylim(-4, 104)
    handles = _group_handles(Line2D, [
        Line2D([], [], linestyle="none", marker="o", color=TEXT_SECONDARY, markerfacecolor=SURFACE, markersize=7,
               label="Hollow: no guardrails"),
        Line2D([], [], linestyle="none", marker="o", color=TEXT_SECONDARY, markersize=7, label="Filled: guardrails")])
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=4, fontsize=7.5,
              labelcolor=TEXT_PRIMARY)
    _style(ax, "From no guardrails to guardrails, per model", "Benign task done (%)", "Attack success (%)")
    drawn += [h.get_label() for h in handles] + notes
    footer = _footer(numbers)
    drawn.append(footer)
    _save(fig, plt, footer, path, notes)
    return drawn


def write_phase5_charts(numbers: dict, chart_dir: str) -> dict:
    """Write the three charts into chart_dir; returns {file name: every text label drawn}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(chart_dir, exist_ok=True)
    models = [m for m in numbers["models"] if m["group"] in ("local", "hosted")]
    drawers = {"asr_ci.png": _asr_ci, "capability_vs_vulnerability.png": _capability,
               "utility_vs_asr_arrows.png": _arrows}
    return {name: drawers[name](plt, Line2D, numbers, models, os.path.join(chart_dir, name)) for name in CHART_FILES}
