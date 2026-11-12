"""Fail if a statement of the notebook or slide text contradicts the results.

The prose of ``scripts/build_lab3.py`` is written once, the numbers come from
a run. Every qualitative statement of that prose is repeated here as a check
on ``results/lab3``, so that a rerun with other settings cannot leave stale
text next to new figures.

    python scripts/check_lab3_claims.py
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from emlab.lab3_config import DEFAULT_CONFIG
from graphmh import running_total_variation

ROOT = Path(__file__).resolve().parents[1]
BEFORE, AFTER = "без интервенции", "с интервенцией"
KEEP = "СОХРАНИТЬ"
STARTS = ("полный граф", "пустой граф", "случайный граф")


def _require(condition, statement):
    if not condition:
        raise AssertionError(f"Stale text: {statement}")


def _settings(summary):
    """The text quotes the settings of the code, the figures those of a run."""
    current = json.loads(json.dumps(asdict(DEFAULT_CONFIG)))
    _require(
        summary["config"] == current,
        "the results were computed with the current configuration",
    )


def _lab2_scaled_distance(root):
    """Section 2: W1/SD of laboratory 2 does not separate the parameters."""
    records = json.loads(
        (root / "results/lab2/distances.json").read_text(encoding="utf-8")
    )
    means = {}
    for record in records:
        if (
            record["dataset"] == "silica_offset"
            and record["loss"] == "linear"
            and record["alpha"] == 2.0
        ):
            means.setdefault(record["parameter"], []).append(
                record["w1_scaled"]
            )
    values = [np.mean(group) for group in means.values()]
    _require(
        len(values) == 4 and 1.65 <= min(values) and max(values) <= 2.55,
        "W1/SD is 1.7-2.5 for all four silica_offset parameters",
    )


def _full_graph_rule(problem):
    """Section 4.1: the laboratory 2 rule on the full circuit graph."""
    full = problem["lab2_full_graph"]
    records = dict(zip(problem["edges"], full["records"]))
    _require(full["boundary_fraction"] > 0.05, "bounds block every deletion")
    _require(
        all(record["decision"] == KEEP for record in records.values()),
        "the full-graph rule keeps all edges",
    )
    for label, record in records.items():
        removable = record["delta_rmse"] <= record["tolerance"]
        _require(
            removable == (label != "R₂"),
            "by RMSE alone M, Rm, Sm, E2 are removable and R2 is not",
        )
    identified = dict(zip(problem["edges"], full["identified"]))
    _require(
        not identified["M"], "in the full graph M is not determined (4.4)"
    )


def _stability_profile(summary):
    """Section 4.2: where the extra edge crosses the threshold."""
    crossing = {"rlc": (1.0, 2.0), "silica": (0.5, 1.0)}
    shown = {case["problem"] for case in summary["stability_profile"]}
    _require(shown == set(crossing), "the figure shows the circuit and silica")
    for case in summary["stability_profile"]:
        levels = case["levels"]
        ratio = np.array(case["ratio"])
        _require(
            ratio[:, :-1].max() < 0.1,
            "generating edges stay an order of magnitude below the threshold",
        )
        low, high = (
            levels.index(level) for level in crossing[case["problem"]]
        )
        _require(
            ratio[low, -1] < 1 < ratio[high, -1],
            f"the extra edge of {case['problem']} crosses the threshold",
        )


def sequence_path(root):
    """Section 4.3: the path of the drawn chain, as steps of its key moves.

    Returns the step at which M leaves the graph, the step at which it
    comes back, the number of proposals rejected in between and the first
    step spent in the generating graph. The slides quote these numbers.
    """
    with np.load(Path(root) / "results/lab3/chains.npz") as saved:
        states = saved[f"rlc|{AFTER}|полный граф|states"]
    truth = np.array([True, True, False, False, False])
    first = int(np.flatnonzero((states == truth).all(axis=1))[0])
    before = states[: first + 1]
    without = np.flatnonzero(~before[:, 0])
    _require(len(without) > 0, "the chain first enters a graph without M")
    lost, back = int(without[0]), int(without[-1]) + 1
    _require(
        np.array_equal(without, np.arange(lost, back)),
        "M is absent during one uninterrupted stretch",
    )
    _require(
        before[lost].sum() == before[lost - 1].sum() and before[lost, 3],
        "M leaves by a replacement while the capacitive edge Sm stays",
    )
    _require(
        before[back - 1].tolist() == [False, True, True, True, True],
        "the graph without M holds Sm corrected by Rm and E2",
    )
    _require(
        before[back, 0] and before[back].sum() == before[back - 1].sum(),
        "M returns through a replacement move",
    )
    stay = (before[lost + 1 : back] == before[lost]).all(axis=1)
    _require(stay.all(), "the chain does not move until M returns")
    return {
        "lost": lost,
        "back": back,
        "rejected": back - lost - 1,
        "truth": first,
        "states": states,
    }


def _sequence(root, shown):
    """Section 4.3: once in the generating graph the chain stays near it."""
    path = sequence_path(root)
    states, first = path["states"], path["truth"]
    changes = np.flatnonzero((states[1:] != states[:-1]).any(axis=1)) + 1
    later = states[changes[changes > first][: shown - 1]]
    _require(
        later[:, :2].all() and later[:, 2:].sum(axis=1).max() <= 1,
        "afterwards the drawn graphs hold {M, R2} plus at most one edge",
    )


def _convergence(root, config):
    """Section 4.4: what the convergence figure is said to show."""
    kept = config["steps"] - config["burn_in"]
    lengths = np.unique(np.geomspace(20, kept, 40).astype(int))
    with np.load(root / "results/lab3/chains.npz") as saved:
        distance = {}
        for stage in (BEFORE, AFTER):
            minima = [
                saved[f"rlc|{stage}|{start}|losses"].min() for start in STARTS
            ]
            _require(
                np.ptp(minima) < 1e-9, "all three starts reach one minimum"
            )
            distance[stage] = np.array(
                [
                    running_total_variation(
                        saved[f"rlc|{stage}|{start}|states"][
                            config["burn_in"] + 1 :
                        ],
                        saved[f"rlc|{stage}|exact"],
                        lengths,
                    )
                    for start in STARTS
                ]
            )
    slope = np.polyfit(
        np.tile(np.log(lengths), len(STARTS)),
        np.log(distance[BEFORE]).ravel(),
        1,
    )[0]
    _require(-0.75 <= slope <= -0.35, "stage 1 distance decays like t^-1/2")


def _proposals(problem):
    """Section 4.4: exact comparison of the two proposals."""
    uniform, informed = problem["proposals"]
    for row in (uniform, informed):
        _require(row["balance_error"] < 1e-12, "detailed balance holds")
    _require(
        1.3 <= informed["acceptance"] / uniform["acceptance"] <= 1.7,
        "the informed proposal is accepted about 1.5 times more often",
    )
    _require(
        uniform["spectral_gap"]
        < informed["spectral_gap"]
        < 1.5 * uniform["spectral_gap"],
        "its spectral gap is slightly larger",
    )
    _require(
        informed["hit_from_full"] > uniform["hit_from_full"],
        "from the full graph the informed proposal reaches G* later",
    )
    _require(
        problem["stages"][AFTER]["exact_truth_probability"] > 0.95,
        "almost all of the stage 2 mass lies in one graph",
    )


def _selection(problems):
    """Sections 4.5 and 4.6: the selected graphs."""
    below_threshold = set()
    for name, problem in problems.items():
        before, after = problem["stages"][BEFORE], problem["stages"][AFTER]
        _require(
            before["selected_is_truth"] and after["selected_is_truth"],
            f"{name}: both stages select the generating graph",
        )
        _require(
            after["selected_share"] > before["selected_share"],
            f"{name}: the share of G* grows",
        )
        _require(problem["lab2_rule_agrees"], f"{name}: the ЛР2 rule agrees")
        for row, old, new in zip(
            problem["edge_report"],
            before["edge_frequency"],
            after["edge_frequency"],
        ):
            if row["truth"]:
                _require(
                    row["instability"] < 0.1,
                    f"{name}: generating edges are stable in G*",
                )
                continue
            _require(
                row["delta_loss_before"] > 0,
                f"{name}: every extra edge already loses at stage 1",
            )
            _require(
                row["instability"] > 0.85,
                f"{name}: the penalty of every extra edge is almost full",
            )
            if row["ratio"] < 1:
                below_threshold.add((name, row["edge"]))
            _require(
                7 <= old / new <= 20,
                f"{name}: extra edges become about ten times rarer",
            )
        for stage in (before, after):
            if "exact_edge_frequency" not in stage:
                continue
            for chain, exact in zip(
                stage["edge_frequency"], stage["exact_edge_frequency"]
            ):
                _require(
                    abs(chain - exact)
                    <= max(0.35 * min(exact, 1 - exact), 1e-3),
                    f"{name}: chain frequencies agree with the exact ones",
                )
        _require(
            problem["validation_rmse"] < problem["tolerance"] / 10,
            f"{name}: validation error is far below the tolerance",
        )
    _require(
        below_threshold == {("silica", "c₂"), ("water", "c₂")},
        "only c2 of both dispersion problems stays just below rho = 1 (4.2)",
    )
    exact = {
        name: [
            "exact_edge_frequency" in problem["stages"][stage]
            for stage in (BEFORE, AFTER)
        ]
        for name, problem in problems.items()
    }
    _require(
        exact
        == {
            "rlc": [True, True],
            "silica": [True, False],
            "water": [True, False],
        },
        "exact frequencies: stage 1 everywhere, stage 2 for the circuits",
    )


def check_claims(root=ROOT):
    """Verify every statement; return the summary used for the checks."""
    root = Path(root)
    summary = json.loads(
        (root / "results/lab3/summary.json").read_text(encoding="utf-8")
    )
    problems = summary["problems"]
    _settings(summary)
    _lab2_scaled_distance(root)
    _full_graph_rule(problems["rlc"])
    _stability_profile(summary)
    _sequence(root, summary["config"]["sequence_steps"])
    _convergence(root, summary["config"])
    _proposals(problems["rlc"])
    _selection(problems)
    return summary


if __name__ == "__main__":
    check_claims()
    print("Statements of the notebook text agree with the results")
