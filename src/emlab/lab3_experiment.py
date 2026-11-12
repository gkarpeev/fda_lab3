"""Run the two sampling stages, store the chains and select a graph."""

import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy

from graphmh import (
    Proposal,
    all_masks,
    best_visited,
    edge_frequencies,
    expected_hitting_time,
    flip,
    gibbs_distribution,
    log_acceptance_ratio,
    mask_index,
    metropolis_hastings,
    spectral_gap,
    stationary_acceptance,
    total_variation,
    transition_matrix,
    uniform_weights,
    visit_frequencies,
)

from .lab3_config import CHAIN_SEED_OFFSET, OUTPUT_PATH
from .lab3_scoring import GraphScorer

OBSERVATIONAL = "без интервенции"
INTERVENTIONAL = "с интервенцией"
STARTS = ("полный граф", "пустой граф", "случайный граф")
UNIFORM = "равномерное"
INFORMED = "по интервенции"
KEEP = "СОХРАНИТЬ"  # verdict text of laboratory 2 `decide_elements`
STABILITY_PROBLEMS = ("rlc", "silica")  # shown in the stability figure


@dataclass
class StageResult:
    """Chains of one target and the graph selected from the stored sample."""

    label: str
    weight: float
    proposal: Proposal
    chains: dict
    selected: tuple
    selected_loss: float
    edge_frequency: np.ndarray
    selected_share: float
    exact: np.ndarray | None


@dataclass
class ProblemResult:
    """Everything the notebook shows for one problem."""

    problem: object
    scorer: GraphScorer
    config: object
    stages: dict

    @property
    def selected(self):
        """Final answer: the graph chosen with the intervention term."""
        return self.stages[INTERVENTIONAL].selected


def make_proposal(scorer, informed):
    """Add/delete/replace moves; deletions may follow the intervention.

    Replacements stay uniform: the intervention measures how firmly an edge
    is determined, and an edge that mimics the true one is determined firmly,
    so stability gives no reason to keep it out of a replacement.
    """
    return Proposal(
        flip_weights=scorer.informed_weights if informed else uniform_weights,
        swap_probability=scorer.config.swap_probability,
    )


def _starts(problem, config, stage_index):
    count = len(problem.edges)
    generator = np.random.default_rng(
        [config.seed + CHAIN_SEED_OFFSET, stage_index, count]
    )
    masks = (
        (True,) * count,
        (False,) * count,
        tuple(bool(value) for value in generator.integers(0, 2, count)),
    )
    return dict(zip(STARTS, masks))


def pooled_sample(chains, burn_in):
    """States of all chains after the burn-in: the sample of graphs."""
    return np.concatenate(
        [chain.states[burn_in + 1 :] for chain in chains.values()]
    )


def exact_edge_frequencies(exact):
    """Probability that a graph drawn from the target contains each edge."""
    count = int(np.log2(len(exact)))
    return np.asarray(exact) @ np.array(all_masks(count), dtype=float)


def exact_distribution(scorer, weight):
    """Target probabilities of all graphs; feasible for small edge sets."""
    loss = scorer.loss(weight)
    masks = all_masks(len(scorer.problem.edges))
    return gibbs_distribution([loss(mask) for mask in masks])


def run_stage(scorer, label, weight, informed, exact, stage_index):
    """Sample graphs from exp(-L) from three starts and pick the best one."""
    config = scorer.config
    loss = scorer.loss(weight)
    proposal = make_proposal(scorer, informed)
    chains = {}
    starts = _starts(scorer.problem, config, stage_index)
    for chain_index, (name, start) in enumerate(starts.items()):
        generator = np.random.default_rng(
            [config.seed + CHAIN_SEED_OFFSET, stage_index, chain_index]
        )
        chains[name] = metropolis_hastings(
            loss, proposal, start, config.steps, generator
        )
    selected, selected_loss = best_visited(chains.values())
    pooled = pooled_sample(chains, config.burn_in)
    return StageResult(
        label=label,
        weight=weight,
        proposal=proposal,
        chains=chains,
        selected=selected,
        selected_loss=selected_loss,
        edge_frequency=edge_frequencies(pooled),
        selected_share=float(np.all(pooled == selected, axis=1).mean()),
        exact=exact_distribution(scorer, weight) if exact else None,
    )


def run_problem(problem, config):
    """Stage 1: accuracy and complexity. Stage 2: plus the intervention."""
    scorer = GraphScorer(problem, config)
    count = len(problem.edges)
    print(f"{problem.name}: {count} edges, {2**count} graphs", flush=True)
    stages = {
        OBSERVATIONAL: run_stage(
            scorer, OBSERVATIONAL, 0.0, False, exact=True, stage_index=0
        ),
        INTERVENTIONAL: run_stage(
            scorer,
            INTERVENTIONAL,
            config.stability_weight,
            True,
            exact=count <= config.exact_stability_edges,
            stage_index=1,
        ),
    }
    return ProblemResult(problem, scorer, config, stages)


def chain_distance(stage, config):
    """Total variation between each chain (after burn-in) and the target."""
    if stage.exact is None:
        raise ValueError("The target of this stage was not enumerated")
    return {
        name: total_variation(
            visit_frequencies(chain.states[config.burn_in + 1 :]), stage.exact
        )
        for name, chain in stage.chains.items()
    }


def proposal_comparison(result):
    """Exact properties of the stage 2 chain for both proposals.

    Uses the full transition matrix, so the numbers carry no Monte Carlo
    error. Detailed balance is checked on the same matrix.
    """
    scorer, stage = result.scorer, result.stages[INTERVENTIONAL]
    if stage.exact is None:
        raise ValueError("The target of this stage was not enumerated")
    count = len(result.problem.edges)
    target = int(mask_index(result.selected))
    rows = []
    for label, informed in ((UNIFORM, False), (INFORMED, True)):
        matrix = transition_matrix(
            scorer.loss(stage.weight), make_proposal(scorer, informed), count
        )
        flow = stage.exact[:, None] * matrix
        hitting = expected_hitting_time(matrix, target)
        rows.append(
            {
                "proposal": label,
                "balance_error": float(np.abs(flow - flow.T).max()),
                "hit_from_full": float(hitting[-1]),
                "hit_from_empty": float(hitting[0]),
                "spectral_gap": spectral_gap(matrix),
                "acceptance": stationary_acceptance(matrix, stage.exact),
            }
        )
    return rows


def edge_report(result):
    """Per-edge criteria around the selected graph.

    An edge of the selected graph is analysed inside it; an absent candidate
    is analysed in the selected graph plus that edge. The laboratory 2 rule,
    the stability ratio and the cost of flipping the edge all refer to the
    same pair of graphs.
    """
    scorer, selected = result.scorer, result.selected
    stage = result.stages[INTERVENTIONAL]
    loss_before = scorer.loss(0.0)
    loss_after = scorer.loss(stage.weight)
    rows = []
    for index, edge in enumerate(result.problem.edges):
        neighbour = flip(selected, index)
        context = selected if selected[index] else neighbour
        stability = scorer.stability(context)
        decision = scorer.lab2_decisions(context)[index]
        log_ratio = log_acceptance_ratio(
            loss_after, stage.proposal, selected, neighbour
        )
        rows.append(
            {
                "edge": edge.label,
                "kind": edge.kind,
                "truth": edge.truth,
                "selected": selected[index],
                "ratio": float(stability.ratio[index]),
                "instability": float(stability.instability[index]),
                "delta_loss_before": float(
                    loss_before(neighbour) - loss_before(selected)
                ),
                "delta_loss_after": float(
                    loss_after(neighbour) - loss_after(selected)
                ),
                "flip_acceptance": float(np.exp(min(0.0, log_ratio))),
                "frequency": float(stage.edge_frequency[index]),
                "exact_frequency": (
                    None
                    if stage.exact is None
                    else float(exact_edge_frequencies(stage.exact)[index])
                ),
                "lab2_delta_rmse": decision["delta_rmse"],
                "lab2_tolerance": decision["tolerance"],
                "lab2_decision": decision["decision"],
            }
        )
    return rows


def lab2_rule_agrees(result):
    """Does the laboratory 2 rule keep exactly the edges of the selection?"""
    return all(
        (row["lab2_decision"] == KEEP) == row["selected"]
        for row in edge_report(result)
    )


def stability_cases(results):
    """Graphs for the stability figure: generating edges plus one extra.

    Only problems present in ``results`` are returned, so a partial run
    yields a partial figure; the notebook text is checked against both.
    """
    cases = []
    for name in STABILITY_PROBLEMS:
        if name in results:
            truth = results[name].problem.truth
            cases.append((results[name], flip(truth, truth.index(False))))
    return cases


def stability_profile(scorer, mask, levels):
    """Stability ratio of every edge of a graph against the noise level."""
    return np.array([scorer.stability(mask, alpha).ratio for alpha in levels])


def weight_sensitivity(scorer, mask, grid):
    """Exact probability of a graph for each stability weight."""
    index = int(mask_index(mask))
    return np.array(
        [exact_distribution(scorer, weight)[index] for weight in grid]
    )


def _plain(value):
    """Convert NumPy containers and scalars to JSON-serialisable objects."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def summarize(result):
    """Numbers of one problem for the tables, the slides and the saved file."""
    problem, config = result.problem, result.config
    scorer = result.scorer
    truth_index = int(mask_index(problem.truth))
    full = (True,) * len(problem.edges)
    summary = {
        "edges": [edge.label for edge in problem.edges],
        "truth": problem.describe(problem.truth),
        "validation_rmse": scorer.validation_rmse(result.selected),
        "tolerance": problem.dataset.tolerance,
        "edge_price": scorer.edge_price,
        "edge_report": edge_report(result),
        "lab2_rule_agrees": lab2_rule_agrees(result),
        "lab2_full_graph": {
            "boundary_fraction": scorer.stability(full).boundary_fraction,
            "identified": scorer.stability(full).identified,
            "records": list(scorer.lab2_decisions(full).values()),
        },
        "stages": {},
    }
    for label, stage in result.stages.items():
        states = np.concatenate([c.states for c in stage.chains.values()])
        record = {
            "selected": problem.describe(stage.selected),
            "selected_is_truth": stage.selected == problem.truth,
            "selected_loss": stage.selected_loss,
            "selected_share": stage.selected_share,
            "edge_frequency": stage.edge_frequency,
            "acceptance": {
                name: chain.acceptance_rate
                for name, chain in stage.chains.items()
            },
            "distinct_graphs": len({tuple(row) for row in states.tolist()}),
        }
        if stage.exact is not None:
            record["exact_truth_probability"] = stage.exact[truth_index]
            record["exact_edge_frequency"] = exact_edge_frequencies(
                stage.exact
            )
            record["total_variation"] = chain_distance(stage, config)
        summary["stages"][label] = record
    if result.stages[INTERVENTIONAL].exact is not None:
        summary["proposals"] = proposal_comparison(result)
        summary["weight_sensitivity"] = dict(
            zip(
                map(str, config.weight_grid),
                weight_sensitivity(
                    scorer, result.selected, config.weight_grid
                ),
            )
        )
    return summary


def save_results(root, results):
    """Store every sampled graph and the selection summary."""
    output = Path(root) / OUTPUT_PATH
    output.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, result in results.items():
        for label, stage in result.stages.items():
            for start, chain in stage.chains.items():
                key = f"{name}|{label}|{start}"
                arrays[f"{key}|states"] = chain.states
                arrays[f"{key}|losses"] = chain.losses
                arrays[f"{key}|proposed"] = chain.proposed
                arrays[f"{key}|accepted"] = chain.accepted
            if stage.exact is not None:
                arrays[f"{name}|{label}|exact"] = stage.exact
    np.savez_compressed(output / "chains.npz", **arrays)
    config = next(iter(results.values())).config
    summary = {
        "config": asdict(config),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "problems": {
            name: summarize(result) for name, result in results.items()
        },
        "stability_profile": [
            {
                "problem": result.problem.name,
                "graph": result.problem.describe(mask),
                "edges": [
                    edge.label
                    for edge, present in zip(result.problem.edges, mask)
                    if present
                ],
                "levels": config.stability_levels,
                "ratio": stability_profile(
                    result.scorer, mask, config.stability_levels
                )[:, np.array(mask)],
            }
            for result, mask in stability_cases(results)
        ],
    }
    summary = _plain(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
