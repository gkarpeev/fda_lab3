"""Compact notebook tables for laboratory 3."""

import platform
from dataclasses import asdict

import numpy as np
import scipy
from IPython.display import Markdown, display

from graphmh import mask_index

from .lab3_experiment import (
    chain_distance,
    edge_report,
    lab2_rule_agrees,
    proposal_comparison,
    weight_sensitivity,
)

YES, NO = "да", "нет"


def _table(headers, rows):
    lines = ["|" + "|".join(headers) + "|"]
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    lines.extend("|" + "|".join(map(str, row)) + "|" for row in rows)
    display(Markdown("\n".join(lines)))


def show_environment(config):
    """Print interpreter and library versions and the experiment settings."""
    print(
        f"Python {platform.python_version()}, NumPy {np.__version__}, "
        f"SciPy {scipy.__version__}"
    )
    display(asdict(config))


def show_edges(problems):
    """List the candidate edges and their parameters."""
    rows = []
    for problem in problems.values():
        lower, upper = problem.dataset.model.bounds
        for index, edge in enumerate(problem.edges):
            rows.append(
                (
                    problem.title,
                    edge.label,
                    f"{edge.source} → {edge.target}",
                    edge.kind,
                    f"`{edge.name}`",
                    f"{edge.ablation:g}",
                    f"[{lower[index]:g}; {upper[index]:g}]",
                    YES if edge.truth else NO,
                )
            )
    _table(
        [
            "Задача",
            "Ребро",
            "Вершины",
            "Смысл",
            "Параметр",
            "Значение без ребра",
            "Границы",
            "В порождающей модели",
        ],
        rows,
    )


def show_lab2_rule(result, mask):
    """Laboratory 2 deletion rule applied once to a given graph."""
    problem, scorer = result.problem, result.scorer
    stability = scorer.stability(mask)
    rows = [
        (
            problem.edges[index].label,
            f"{record['estimate']:.4g}",
            f"{record['delta_rmse']:.3g}",
            f"{record['tolerance']:.3g}",
            YES if stability.identified[index] else NO,
            f"{stability.instability[index]:.2g}",
            record["decision"],
        )
        for index, record in scorer.lab2_decisions(mask).items()
    ]
    print(
        f"{problem.title}, граф {problem.describe(mask)}: "
        f"доля бутстреп-оценок на границах {stability.boundary_fraction:.0%}"
    )
    _table(
        [
            "Ребро",
            "Оценка",
            "ΔRMSE",
            "Допуск τ",
            "Параметр определён",
            "Неустойчивость uₑ",
            "Правило ЛР2",
        ],
        rows,
    )


def show_summary(results):
    """Selected graph and sampler statistics for every problem and stage."""
    rows = []
    for result in results.values():
        problem, config = result.problem, result.config
        for stage in result.stages.values():
            exact, distance = "—", "—"
            if stage.exact is not None:
                exact = f"{stage.exact[int(mask_index(stage.selected))]:.3f}"
                distance = f"{max(chain_distance(stage, config).values()):.3f}"
            acceptance = np.mean(
                [chain.acceptance_rate for chain in stage.chains.values()]
            )
            rows.append(
                (
                    problem.title,
                    stage.label,
                    problem.describe(stage.selected),
                    YES if stage.selected == problem.truth else NO,
                    f"{stage.selected_share:.3f}",
                    exact,
                    f"{acceptance:.3f}",
                    distance,
                    YES if lab2_rule_agrees(result) else NO,
                )
            )
    _table(
        [
            "Задача",
            "Этап",
            "Выбранный граф G*",
            "Совпал с порождающим",
            "Частота G* в цепях",
            "Точное π(G*)",
            "Доля принятых шагов",
            "max TV по цепям",
            "Правило ЛР2 согласно с итогом",
        ],
        rows,
    )


def show_edge_report(results):
    """Per-edge criteria around the selected graph of every problem."""
    rows = []
    for result in results.values():
        for row in edge_report(result):
            exact = row["exact_frequency"]
            rows.append(
                (
                    result.problem.title,
                    row["edge"],
                    YES if row["selected"] else NO,
                    f"{row['ratio']:.3g}",
                    f"{row['instability']:.2g}",
                    f"{row['delta_loss_before']:.3g}",
                    f"{row['delta_loss_after']:.3g}",
                    f"{row['frequency']:.4f}",
                    "—" if exact is None else f"{exact:.4f}",
                    f"{row['lab2_delta_rmse'] / row['lab2_tolerance']:.2g}",
                    row["lab2_decision"],
                )
            )
    _table(
        [
            "Задача",
            "Ребро",
            "В G*",
            "ρₑ",
            "uₑ",
            "ΔJₑ без интервенции",
            "ΔJₑ с интервенцией",
            "Частота с интервенцией",
            "Точная частота",
            "ΔRMSE / τ",
            "Правило ЛР2",
        ],
        rows,
    )


def show_proposals(result):
    """Exact comparison of the uniform and the intervention-based proposal."""
    _table(
        [
            "Предложение",
            "Ошибка детального баланса",
            "Шагов до G* из полного графа",
            "Шагов до G* из пустого графа",
            "Спектральный зазор",
            "Доля принятых шагов",
        ],
        [
            (
                row["proposal"],
                f"{row['balance_error']:.1e}",
                f"{row['hit_from_full']:.0f}",
                f"{row['hit_from_empty']:.0f}",
                f"{row['spectral_gap']:.2e}",
                f"{row['acceptance']:.4f}",
            )
            for row in proposal_comparison(result)
        ],
    )


def show_weight_sensitivity(result):
    """Exact probability of the selected graph against the stability weight."""
    grid = result.config.weight_grid
    values = weight_sensitivity(result.scorer, result.selected, grid)
    _table(
        ["Вес неустойчивости λ", *[f"{weight:g}" for weight in grid]],
        [
            (
                f"π({result.problem.describe(result.selected)})",
                *[f"{v:.4f}" for v in values],
            )
        ],
    )
