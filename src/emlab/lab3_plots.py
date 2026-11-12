"""Notebook figures for laboratory 3."""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

from graphmh import all_masks, running_total_variation, visit_frequencies

from .lab3_config import OUTPUT_PATH
from .lab3_experiment import (
    OBSERVATIONAL,
    exact_edge_frequencies,
    stability_profile,
)

# Categorical colours in a fixed order: an edge kind (or a chain start) keeps
# its colour in every figure. Text always uses the ink colours below.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
FAINT = "#b9b8b2"
NODE_FILL = "#f3f2ee"
# Every figure has this width. With the font sizes below its text stays
# legible where the figures are read: in a notebook rendered about 900 px
# wide (GitHub) and on a 16:9 slide, where the figure shrinks to 80%.
WIDTH = 7.4
NODE_RADIUS = 0.17
LOOP_RADIUS = 0.2


def configure_plots():
    """Apply one quiet style to every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )


def _finish(figure, root, filename):
    output = Path(root) / OUTPUT_PATH
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / f"{filename}.png", bbox_inches="tight")
    plt.show()
    plt.close(figure)


def _bottom_legend(figure, axis):
    """One legend under all panels, where it cannot cover the data."""
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=len(handles), fontsize=7.5
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))


def _kind_colors(problem):
    kinds = list(dict.fromkeys(edge.kind for edge in problem.edges))
    return {kind: SERIES[index] for index, kind in enumerate(kinds)}


def _edge_color(problem, edge):
    return _kind_colors(problem)[edge.kind]


def _trim(start, end, radius):
    """Shorten a segment so that it ends on the node circles."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    direction = (end - start) / np.linalg.norm(end - start)
    return start + radius * direction, end - radius * direction


def _draw_edge(axis, problem, edge, color, style="-", width=2.0):
    start = np.array(problem.positions[edge.source])
    end = np.array(problem.positions[edge.target])
    if edge.source == edge.target:
        centre = start + np.array([0.0, NODE_RADIUS + 0.6 * LOOP_RADIUS])
        axis.add_patch(
            Arc(
                centre,
                2 * LOOP_RADIUS,
                2 * LOOP_RADIUS,
                theta1=-50,
                theta2=230,
                color=color,
                linewidth=width,
                linestyle=style,
            )
        )
        anchor = centre + np.array([0.0, LOOP_RADIUS + 0.12])
    else:
        tail, head = _trim(start, end, NODE_RADIUS)
        axis.add_patch(
            FancyArrowPatch(
                tail,
                head,
                connectionstyle=f"arc3,rad={edge.curve}",
                arrowstyle="-|>" if edge.directed else "-",
                mutation_scale=9,
                color=color,
                linewidth=width,
                linestyle=style,
                shrinkA=0,
                shrinkB=0,
            )
        )
        # arc3 puts its control point rad * length off the chord, so the
        # curve passes half of that away from the chord midpoint.
        chord = head - tail
        normal = np.array([chord[1], -chord[0]])
        anchor = (
            tail
            + problem.label_at * chord
            + 2
            * problem.label_at
            * (1 - problem.label_at)
            * edge.curve
            * normal
        )
    axis.text(
        *anchor,
        edge.label,
        ha="center",
        va="center",
        fontsize=7.5,
        color=INK,
        zorder=5,
        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none"},
    )


def _draw_nodes(axis, problem, labels):
    centre_x = np.mean([x for x, _ in problem.positions.values()])
    for name, (x, y) in problem.positions.items():
        axis.add_patch(
            Circle(
                (x, y),
                NODE_RADIUS,
                facecolor=NODE_FILL,
                edgecolor=MUTED,
                linewidth=1.0,
                zorder=3,
            )
        )
        if not labels:
            continue
        if problem.label_side == "below":
            axis.text(
                x,
                y - NODE_RADIUS - 0.08,
                fill(name, 10),
                ha="center",
                va="top",
                fontsize=8,
                color=INK,
            )
        else:
            left = x < centre_x
            axis.text(
                x + (-1 if left else 1) * (NODE_RADIUS + 0.08),
                y,
                name,
                ha="right" if left else "left",
                va="center",
                fontsize=8,
                color=INK,
            )


def draw_graph(axis, problem, mask, absent="hide", node_labels=True):
    """Draw a graph; ``absent='dashed'`` also shows missing candidates."""
    for source, target in problem.known_edges:
        tail, head = _trim(
            problem.positions[source], problem.positions[target], NODE_RADIUS
        )
        axis.add_patch(
            FancyArrowPatch(
                tail,
                head,
                arrowstyle="-|>",
                mutation_scale=9,
                color=FAINT,
                linewidth=1.5,
                shrinkA=0,
                shrinkB=0,
            )
        )
    for edge, present in zip(problem.edges, mask):
        if present:
            _draw_edge(axis, problem, edge, _edge_color(problem, edge))
        elif absent == "dashed":
            _draw_edge(
                axis,
                problem,
                edge,
                _edge_color(problem, edge),
                style=(0, (2, 2)),
                width=1.3,
            )
    _draw_nodes(axis, problem, node_labels)
    xs, ys = zip(*problem.positions.values())
    side, below, above = problem.margin
    axis.set_xlim(min(xs) - side, max(xs) + side)
    axis.set_ylim(min(ys) - below, max(ys) + above)
    axis.set_aspect("equal")
    axis.axis("off")


def _kind_legend(axis, problem, **options):
    handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=kind)
        for kind, color in _kind_colors(problem).items()
    ]
    axis.legend(handles=handles, fontsize=7.5, **options)


def plot_candidates(root, problems):
    """Candidate graphs: solid edges generated the data, dashed did not."""
    configure_plots()
    figure, axes = plt.subplots(
        1,
        len(problems),
        figsize=(WIDTH, 3.7),
        gridspec_kw={"width_ratios": (1.35, 1, 1)},
    )
    for axis, problem in zip(axes, problems.values()):
        draw_graph(axis, problem, problem.truth, absent="dashed")
        axis.set_anchor("N")  # panels differ in height; titles share a line
        axis.set_title(f"{problem.title}\n{len(problem.edges)} кандидатов")
        _kind_legend(
            axis,
            problem,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=1,
            handlelength=1.6,
        )
    figure.tight_layout()
    _finish(figure, root, "candidates")


def _move(problem, before, after):
    """Describe the difference between two consecutive graphs."""
    removed = [
        e.label for e, a, b in zip(problem.edges, before, after) if a and not b
    ]
    added = [
        e.label for e, a, b in zip(problem.edges, before, after) if b and not a
    ]
    if removed and added:
        return f"{removed[0]} → {added[0]}"
    return f"− {removed[0]}" if removed else f"+ {added[0]}"


def plot_sequence(root, result, stage_label, start, count):
    """First accepted graphs of one chain: the generated sequence."""
    configure_plots()
    problem = result.problem
    stage = result.stages[stage_label]
    chain = stage.chains[start]
    accepted = np.flatnonzero(chain.accepted)[: count - 1]
    steps = np.r_[0, accepted + 1]
    columns = (len(steps) + 1) // 2
    figure, axes = plt.subplots(2, columns, figsize=(WIDTH, 3.0))
    for axis in axes.ravel():
        axis.axis("off")
    minimum = min(item.losses.min() for item in stage.chains.values())
    for panel, (axis, step) in enumerate(zip(axes.ravel(), steps)):
        draw_graph(axis, problem, tuple(chain.states[step]), node_labels=False)
        move = "старт"
        if panel > 0:
            move = _move(problem, chain.states[step - 1], chain.states[step])
        excess = chain.losses[step] - minimum
        axis.set_title(
            f"шаг {step}: {move}\nJ(Gₜ) − J(G*) = {excess:.1f}",
            fontsize=7.5,
            fontweight="normal",
            color=INK,
        )
    figure.suptitle(
        f"{problem.title}, {stage_label}, старт «{start}»",
        fontsize=10,
        fontweight="bold",
    )
    figure.legend(
        handles=[
            plt.Line2D([], [], color=color, linewidth=2, label=kind)
            for kind, color in _kind_colors(problem).items()
        ],
        loc="lower center",
        ncol=len(_kind_colors(problem)),
        fontsize=7.5,
    )
    figure.subplots_adjust(
        left=0.01, right=0.99, top=0.78, bottom=0.10, hspace=0.45, wspace=0.04
    )
    _finish(figure, root, f"{problem.name}_sequence")


def plot_convergence(root, result):
    """Loss traces and distance to the exact target for every start."""
    configure_plots()
    config = result.config
    figure, axes = plt.subplots(2, 2, figsize=(WIDTH, 4.8))
    kept = config.steps - config.burn_in
    checkpoints = np.unique(np.geomspace(20, kept, 40).astype(int))
    trace = min(300, config.steps)
    for row, stage in enumerate(result.stages.values()):
        minimum = min(chain.losses.min() for chain in stage.chains.values())
        for color, (start, chain) in zip(SERIES, stage.chains.items()):
            axes[row, 0].plot(
                np.arange(trace + 1),
                chain.losses[: trace + 1] - minimum + 1.0,
                color=color,
                label=start,
                drawstyle="steps-post",
            )
            axes[row, 1].plot(
                checkpoints,
                running_total_variation(
                    chain.states[config.burn_in + 1 :],
                    stage.exact,
                    checkpoints,
                ),
                color=color,
                label=start,
            )
        axes[row, 0].set(
            yscale="log",
            xlabel="шаг t",
            ylabel="J(Gₜ) − J(G*) + 1",
            title=f"{stage.label}: начало цепи",
        )
        axes[row, 1].plot(
            checkpoints, 2.0 * checkpoints**-0.5, color=FAINT, linewidth=1.2
        )
        # The corner above the reference line is the only empty place.
        axes[row, 1].text(
            0.98,
            0.95,
            "серая линия: наклон t^(−1/2)",
            transform=axes[row, 1].transAxes,
            ha="right",
            va="top",
            fontsize=7.5,
            color=MUTED,
        )
        axes[row, 1].set(
            xscale="log",
            yscale="log",
            xlabel=f"шагов после прогрева ({config.burn_in})",
            ylabel="TV(частоты, π)",
            title=f"{stage.label}: расстояние до π",
        )
    axes[0, 0].legend(fontsize=7.5)
    figure.tight_layout()
    _finish(figure, root, f"{result.problem.name}_convergence")


def plot_posterior(root, result, top=6):
    """Chain frequencies against the exact target for the likeliest graphs."""
    configure_plots()
    problem, config = result.problem, result.config
    masks = all_masks(len(problem.edges))
    reference = result.stages[OBSERVATIONAL].exact
    order = np.argsort(reference)[::-1][:top]
    names = [problem.describe(masks[index]) for index in order]
    figure, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    rows = np.arange(top)[::-1]
    for axis, stage in zip(axes, result.stages.values()):
        pooled = np.concatenate(
            [c.states[config.burn_in + 1 :] for c in stage.chains.values()]
        )
        frequency = visit_frequencies(pooled)[order]
        axis.barh(
            rows,
            frequency,
            height=0.55,
            color=SERIES[0],
            label="частота в цепях",
        )
        axis.scatter(
            stage.exact[order],
            rows,
            s=42,
            color=INK,
            zorder=3,
            edgecolor="white",
            linewidth=1.2,
            label="точное π(G)",
        )
        for y, exact, share in zip(rows, stage.exact[order], frequency):
            axis.text(
                max(exact, share) + 0.025,
                y,
                f"{exact:.3f}",
                va="center",
                fontsize=7.5,
                color=MUTED,
            )
        axis.set(
            xlim=(0, 1.12),
            xlabel="вероятность графа",
            title=stage.label,
        )
        axis.grid(axis="y", visible=False)
    axes[0].set_yticks(rows, names)
    axes[1].legend(fontsize=7.5, loc="lower right")
    figure.tight_layout()
    _finish(figure, root, f"{problem.name}_posterior")


def plot_edge_frequencies(root, results):
    """Share of sampled graphs that contain each edge, before and after."""
    configure_plots()
    figure, axes = plt.subplots(
        1, len(results), figsize=(WIDTH, 2.7), sharey=True
    )
    width = 0.36
    for axis, result in zip(axes, results.values()):
        labels = [edge.label for edge in result.problem.edges]
        positions = np.arange(len(labels))
        for offset, color, stage in zip(
            (-width / 2, width / 2), SERIES, result.stages.values()
        ):
            axis.bar(
                positions + offset,
                stage.edge_frequency,
                width - 0.04,
                color=color,
                label=stage.label,
            )
            if stage.exact is not None:
                axis.scatter(
                    positions + offset,
                    exact_edge_frequencies(stage.exact),
                    s=14,
                    color=INK,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                    label="точно, перебором",
                )
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set(title=fill(result.problem.title, 20), ylim=(0, 1.08))
        axis.grid(axis="x", visible=False)
    axes[0].set_ylabel("частота ребра в цепях")
    # The first panel has the exact values of both stages; its second set
    # of dots repeats the label of the first one.
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        ncol=len(unique),
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    _finish(figure, root, "edge_frequencies")


def _end_labels(axis, x, values, labels):
    """Label line ends; lines that end together share one label."""
    order = np.argsort(values)
    groups = [[order[0]]]
    for index in order[1:]:
        if values[index] < 1.3 * values[groups[-1][-1]]:
            groups[-1].append(index)
        else:
            groups.append([index])
    for group in groups:
        axis.annotate(
            ", ".join(labels[index] for index in group),
            (x, np.exp(np.mean(np.log(values[group])))),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )


def plot_stability(root, cases, levels):
    """Stability ratio of each edge against the intervention level."""
    configure_plots()
    figure, axes = plt.subplots(1, len(cases), figsize=(WIDTH, 3.2))
    for axis, (result, mask) in zip(np.atleast_1d(axes), cases):
        problem = result.problem
        profile = stability_profile(result.scorer, mask, levels)
        present = np.flatnonzero(mask)
        for index in present:
            edge = problem.edges[index]
            axis.plot(
                levels,
                profile[:, index],
                marker="o",
                markersize=5,
                color=_edge_color(problem, edge),
                linestyle="-" if edge.truth else (0, (4, 2)),
            )
        _end_labels(
            axis,
            levels[-1],
            profile[-1, present],
            [problem.edges[index].label for index in present],
        )
        axis.axhline(1.0, color=INK, linewidth=1.0)
        axis.annotate(
            "порог ρₑ = 1",
            (levels[-1], 1.0),
            xytext=(7, 3),
            textcoords="offset points",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
        axis.set(
            xscale="log",
            yscale="log",
            xlabel="уровень интервенции α",
            ylabel="относительный сдвиг ρₑ",
            title=f"{problem.title}\nграф {problem.describe(mask)}",
        )
        axis.set_xticks(levels, [f"{level:g}" for level in levels])
        axis.minorticks_off()
        axis.set_xlim(levels[0] / 1.2, levels[-1] * 1.9)
    figure.tight_layout()
    _finish(figure, root, "stability_profile")


def plot_approximation(root, results):
    """Observations, the selected graph and its standardized residuals."""
    configure_plots()
    figure, axes = plt.subplots(
        2,
        len(results),
        figsize=(WIDTH, 4.0),
        gridspec_kw={"height_ratios": (2.4, 1)},
        sharex="col",
    )
    for column, result in enumerate(results.values()):
        problem, scorer = result.problem, result.scorer
        model = problem.dataset.model
        x = model.x * problem.x_factor
        grid = np.linspace(model.x.min(), model.x.max(), 1200)
        top, bottom = axes[0, column], axes[1, column]
        top.plot(
            grid * problem.x_factor,
            scorer.predict(result.selected, grid) * problem.y_factor,
            color=SERIES[0],
            linewidth=1.3,
            label="выбранный граф",
        )
        top.scatter(
            x,
            model.y * problem.y_factor,
            s=8,
            color=INK,
            linewidth=0,
            zorder=3,
            label="наблюдения",
        )
        residual = (
            scorer.predict(result.selected, model.x) - model.y
        ) / model.scale
        bottom.scatter(x, residual, s=6, color=SERIES[0], linewidth=0)
        bottom.axhline(0.0, color=MUTED, linewidth=0.8)
        rmse = scorer.validation_rmse(result.selected) * problem.y_factor
        tolerance = problem.dataset.tolerance * problem.y_factor
        top.set_ylabel(problem.y_label)
        top.set_title(
            f"{problem.title}\n{problem.describe(result.selected)}", pad=20
        )
        top.text(
            0.5,
            1.03,
            f"RMSE {rmse:.2g}{problem.y_unit} · "
            f"допуск τ {tolerance:.2g}{problem.y_unit}",
            transform=top.transAxes,
            ha="center",
            fontsize=7.5,
            color=MUTED,
        )
        bottom.set(xlabel=problem.x_label, ylim=(-4.5, 4.5))
        bottom.set_ylabel("остаток / s")
    _bottom_legend(figure, axes[0, 0])
    _finish(figure, root, "approximation")
