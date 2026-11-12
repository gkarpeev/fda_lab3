"""Checks for the candidate graphs, the objective and the sampling stages."""

import ast
import importlib.util
from dataclasses import replace
from pathlib import Path

import nbformat
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from emlab.intervention import (
    conditional_noise,
    marginal_w1,
    residual_bootstrap,
    rlc_current,
)
from emlab.lab2_config import (
    DECISION_LOSS,
    FREQUENCY_HZ,
    L1_HENRY,
    L2_HENRY,
    R1_OHM,
)
from emlab.lab2_config import (
    DEFAULT_CONFIG as LAB2_CONFIG,
)
from emlab.lab2_data import SellmeierPredictor, prepare_datasets
from emlab.lab3_circuit import coupled_current
from emlab.lab3_config import DEFAULT_CONFIG, NOTEBOOK_PATH, Lab3Config
from emlab.lab3_experiment import (
    INTERVENTIONAL,
    OBSERVATIONAL,
    StageResult,
    chain_distance,
    edge_report,
    exact_edge_frequencies,
    pooled_sample,
    proposal_comparison,
    run_problem,
    save_results,
)
from emlab.lab3_graphs import LibraryPredictor, prepare_problems
from emlab.lab3_scoring import GraphScorer, boundary_share
from graphmh import Chain, uniform_weights

ROOT = Path(__file__).resolve().parents[1]
TEST_CONFIG = replace(
    DEFAULT_CONFIG,
    bootstrap_replicates=30,
    intervention_repeats=1,
    steps=1500,
    burn_in=200,
)


@pytest.fixture(scope="module")
def problems():
    return prepare_problems(ROOT)


@pytest.fixture(scope="module")
def silica(problems):
    return GraphScorer(problems["silica"], TEST_CONFIG)


@pytest.fixture(scope="module")
def rlc_result(problems):
    return run_problem(problems["rlc"], TEST_CONFIG)


def test_coupled_current_repeats_laboratory_2_without_extra_couplings():
    times = np.linspace(0, 2e-6, 151)
    for mutual, resistance in ((3.8, 4.0), (30.0, 8.0), (0.0, 1e-6)):
        np.testing.assert_array_equal(
            coupled_current(times, [mutual, resistance, 0.0, 0.0, 0.0]),
            rlc_current(times, [mutual, resistance]),
        )


def test_coupled_current_solves_the_charge_equation_with_all_couplings():
    """Independent check: integrate L q'' + R q' + S q = e(t) directly."""
    theta = [5.0, 3.0, 0.3, 40.0, 120.0]
    omega = 2 * np.pi * FREQUENCY_HZ
    inductance = np.array([[L1_HENRY, 5e-9], [5e-9, L2_HENRY]])
    resistance = np.array([[R1_OHM, 0.3], [0.3, 3.0]])
    elastance = np.array(
        [[omega**2 * L1_HENRY, 40e6], [40e6, omega**2 * L2_HENRY]]
    )

    def rhs(time, state):
        charge, current = state[:2], state[2:]
        source = np.array([1.0, 0.12]) * np.cos(omega * time)
        return np.r_[
            current,
            np.linalg.solve(
                inductance, source - resistance @ current - elastance @ charge
            ),
        ]

    times = np.linspace(0, 1e-6, 101)
    solution = solve_ivp(
        rhs,
        (0, times[-1]),
        np.zeros(4),
        t_eval=times,
        rtol=1e-10,
        atol=1e-18,
        method="DOP853",
    )
    expected = solution.y[3]
    np.testing.assert_allclose(
        coupled_current(times, theta),
        expected,
        atol=1e-7 * abs(expected).max(),
    )
    # Each coupling must change the current, otherwise its edge is idle.
    for index in (2, 3, 4):
        without = list(theta)
        without[index] = 0.0
        assert abs(coupled_current(times, without) - expected).max() > 1e-4


def test_library_predictor_extends_the_laboratory_2_offset_model():
    wavelengths = np.linspace(0.3, 3.0, 40)
    resonances = np.array([0.0684043, 0.1162414, 9.896161]) ** 2
    theta = np.array([0.7, 0.4, 0.9, 0.01])
    np.testing.assert_allclose(
        LibraryPredictor(SellmeierPredictor(resonances), (0,))(
            wavelengths, theta
        ),
        SellmeierPredictor(resonances, include_offset=True)(
            wavelengths, theta
        ),
    )
    design = LibraryPredictor(SellmeierPredictor(resonances), (2, -4)).design(
        wavelengths
    )
    np.testing.assert_allclose(design[:, 3], wavelengths**2)
    np.testing.assert_allclose(design[:, 4], wavelengths**-4.0)


def test_problems_keep_laboratory_2_data_and_parameters(problems):
    datasets = prepare_datasets(ROOT, LAB2_CONFIG)
    for name, problem in problems.items():
        original, extended = datasets[name].model, problem.dataset.model
        count = len(original.names)
        np.testing.assert_array_equal(extended.y, original.y)
        np.testing.assert_array_equal(extended.scale, original.scale)
        assert extended.names[:count] == tuple(original.names)
        assert problem.truth == (True,) * count + (False,) * (
            len(problem.edges) - count
        )
        assert [edge.ablation for edge in problem.edges[:count]] == [
            datasets[name].ablations[index] for index in range(count)
        ]


def test_laboratory_2_graph_reproduces_its_fit_and_bootstrap(problems):
    """The mask of the original edges is the laboratory 2 model itself."""
    problem = problems["rlc"]
    config = replace(
        DEFAULT_CONFIG,
        bootstrap_replicates=LAB2_CONFIG.bootstrap_replicates,
        intervention_repeats=LAB2_CONFIG.intervention_repeats,
    )
    scorer = GraphScorer(problem, config)
    original = prepare_datasets(ROOT, LAB2_CONFIG)["rlc"].model
    np.testing.assert_array_equal(
        scorer.fit(problem.truth).theta[:2], original.fit()[0]
    )
    submodel = scorer.submodel(problem.truth)
    draws, _, _ = residual_bootstrap(submodel, submodel.y, scorer.indices)
    with np.load(ROOT / "results/lab2/bootstrap_draws.npz") as saved:
        # Saved on another platform; the optimizer stops at a relative
        # tolerance of 1e-8 in the cost, i.e. ~1e-6 in the parameters. The
        # bootstrap spread is ~1e-2, so 1e-4 still pins every draw.
        np.testing.assert_allclose(
            draws, saved["rlc_linear_baseline"], rtol=1e-4
        )


def test_fit_keeps_absent_edges_at_their_ablation_values(silica, problems):
    mask = (True, True, True, False, True, False, False)
    theta = silica.fit(mask).theta
    np.testing.assert_array_equal(theta[[3, 5, 6]], 0.0)
    assert np.all(theta[[0, 1, 2]] > 0.1)
    # The only non-zero ablation value: the tag loop cannot have R2 = 0.
    circuit = GraphScorer(problems["rlc"], TEST_CONFIG)
    theta = circuit.fit((True, False, False, False, False)).theta
    np.testing.assert_array_equal(theta[1:], [1e-6, 0.0, 0.0, 0.0])
    empty = (False,) * 7
    model = silica.problem.dataset.model
    np.testing.assert_array_equal(silica.fit(empty).theta, 0.0)
    assert silica.fit(empty).chi2 == pytest.approx(
        float(((model.y / model.scale) ** 2).sum())
    )
    with pytest.raises(ValueError, match="empty graph"):
        silica.submodel(empty)


def test_stability_separates_redundant_edges_from_needed_ones(silica):
    mask = (True, True, True, True, False, False, False)
    stability = silica.stability(mask)
    assert np.all(stability.instability[:3] < 0.05)
    assert stability.ratio[3] > 1 and stability.instability[3] == 1.0
    assert np.all(np.isnan(stability.ratio[4:]))
    np.testing.assert_array_equal(stability.instability[4:], 0.0)
    # A stronger intervention moves every distribution further.
    weaker = silica.stability(mask, alpha=0.5).ratio[:4]
    assert np.all(weaker < stability.ratio[:4])


def test_stability_ratio_is_the_mean_shift_over_the_removal_distance(problems):
    config = replace(TEST_CONFIG, intervention_repeats=2)
    scorer = GraphScorer(problems["silica"], config)
    mask = (True, True, True, True, False, False, False)
    model = scorer.submodel(mask)
    baseline, _, _ = residual_bootstrap(
        model, model.y, scorer.indices, DECISION_LOSS
    )
    shifts = []
    for noise in scorer.noise:
        moved, _, _ = residual_bootstrap(
            model,
            conditional_noise(
                model.y, model.scale, config.intervention_alpha, noise
            ),
            scorer.indices,
            DECISION_LOSS,
        )
        shifts.append(marginal_w1(baseline, moved)[0])
    assert len(shifts) == 2 and not np.allclose(shifts[0], shifts[1])
    # Every ablation value of this problem is zero.
    expected = np.mean(shifts, axis=0) / np.abs(baseline).mean(axis=0)
    np.testing.assert_allclose(scorer.stability(mask).ratio[:4], expected)
    # For the redundant c0 the estimates straddle zero: the distance to
    # deletion is their mean absolute value, not the absolute mean.
    assert np.abs(baseline[:, 3]).mean() > 3 * abs(baseline[:, 3].mean())


def test_unidentified_parameter_is_declared_unstable(problems):
    """Losses alone do not affect a tag that nothing drives."""
    scorer = GraphScorer(problems["rlc"], TEST_CONFIG)
    stability = scorer.stability((False, True, False, False, False))
    assert not stability.identified[1]
    assert stability.instability[1] == 1.0


def test_parameter_stuck_to_a_bound_is_declared_unstable(problems):
    """In the full circuit graph M and Sm compensate each other."""
    scorer = GraphScorer(problems["rlc"], TEST_CONFIG)
    stability = scorer.stability((True,) * 5)
    stuck, free = [0, 3], [1, 2, 4]
    # Their distributions barely move, which would look like stability.
    assert np.all(stability.ratio[stuck] < 0.5)
    assert not stability.identified[stuck].any()
    np.testing.assert_array_equal(stability.instability[stuck], 1.0)
    assert stability.identified[free].all()
    np.testing.assert_array_equal(
        stability.instability[free], stability.ratio[free]
    )


def test_boundary_share_is_the_largest_over_all_bootstraps():
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    before = np.array([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7], [0.2, 0.8]])
    after = np.array([[0.5, 1.0], [0.0, 1.0], [0.3, 1.0], [0.2, 0.9]])
    np.testing.assert_allclose(boundary_share([before], bounds), [0.0, 0.0])
    np.testing.assert_allclose(
        boundary_share([before, after], bounds), [0.25, 0.75]
    )


def test_loss_adds_accuracy_complexity_and_stability(silica):
    mask = (True, True, True, True, False, False, False)
    price = 0.5 * np.log(len(silica.problem.dataset.model.y))
    parts = silica.parts(mask, weight=2.0)
    assert parts.accuracy == pytest.approx(0.5 * silica.fit(mask).chi2)
    assert parts.complexity == pytest.approx(4 * price)
    assert parts.stability == pytest.approx(
        2.0 * price * silica.stability(mask).instability.sum()
    )
    assert silica.loss(2.0)(mask) == pytest.approx(parts.total)
    assert silica.parts(mask, weight=0.0).stability == 0.0


def test_first_stage_runs_without_any_intervention(problems):
    scorer = GraphScorer(problems["silica"], TEST_CONFIG)
    scorer.loss(0.0)((True, True, True, True, False, False, False))
    assert scorer._fits and not scorer._stability


def test_informed_weights_follow_the_intervention(silica):
    mask = (True, True, True, True, False, False, False)
    weights = silica.informed_weights(mask)
    instability = silica.stability(mask).instability
    np.testing.assert_allclose(
        weights[:4], TEST_CONFIG.proposal_floor + instability[:4]
    )
    np.testing.assert_array_equal(weights[4:], 1.0)
    assert weights[3] > 5 * weights[:3].max()


def test_laboratory_2_rule_agrees_on_the_extended_graph(silica):
    mask = (True, True, True, True, False, False, False)
    decisions = silica.lab2_decisions(mask)
    assert [decisions[index]["decision"] for index in range(4)] == [
        "СОХРАНИТЬ",
        "СОХРАНИТЬ",
        "СОХРАНИТЬ",
        "УДАЛИТЬ",
    ]
    assert decisions[3]["parameter"] == "c0"
    with pytest.raises(ValueError, match="at least two edges"):
        silica.lab2_decisions((True,) + (False,) * 6)


def test_both_stages_select_the_generating_graph(rlc_result, problems):
    truth = problems["rlc"].truth
    for stage in rlc_result.stages.values():
        assert stage.selected == truth
        assert stage.exact.argmax() == 24  # binary 11000: {M, R2}
    before = rlc_result.stages[OBSERVATIONAL]
    after = rlc_result.stages[INTERVENTIONAL]
    assert after.exact[24] > before.exact[24] + 0.1
    assert np.all(after.edge_frequency[2:] < before.edge_frequency[2:])


def test_chains_agree_with_the_exact_target(rlc_result):
    # Only the first stage can show it: there a chain that never left the
    # selected graph would be 0.25 away from the target. In the second
    # stage 0.98 of the mass sits in one graph and any chain looks close.
    stage = rlc_result.stages[OBSERVATIONAL]
    assert 1 - stage.exact.max() > 0.2
    distances = chain_distance(stage, rlc_result.config)
    assert max(distances.values()) < 0.12


def test_second_stage_proposes_deletions_by_the_intervention(rlc_result):
    before = rlc_result.stages[OBSERVATIONAL].proposal
    after = rlc_result.stages[INTERVENTIONAL].proposal
    assert before.flip_weights is uniform_weights
    assert after.flip_weights == rlc_result.scorer.informed_weights
    assert before.swap_weights is after.swap_weights is uniform_weights
    assert before.swap_probability == after.swap_probability == 0.3


def _stage(states, exact):
    states = np.array(states, dtype=bool)
    steps = len(states) - 1
    chain = Chain(
        states,
        np.zeros(steps + 1),
        states[1:],
        np.zeros(steps, dtype=bool),
        np.zeros(steps),
    )
    return StageResult(
        label="",
        weight=0.0,
        proposal=None,
        chains={"start": chain},
        selected=(False,),
        selected_loss=0.0,
        edge_frequency=np.zeros(1),
        selected_share=0.0,
        exact=np.array(exact),
    )


def test_burn_in_is_left_out_of_the_sample():
    config = replace(TEST_CONFIG, steps=5, burn_in=2)
    stage = _stage([(True,)] * 3 + [(False,)] * 3, [1.0, 0.0])
    sample = pooled_sample(stage.chains, config.burn_in)
    assert sample.shape == (3, 1) and not sample.any()
    assert chain_distance(stage, config) == {"start": 0.0}


def test_exact_edge_frequencies_are_marginals_of_the_target():
    # Graphs are ordered as binary numbers: 00, 01, 10, 11.
    np.testing.assert_allclose(
        exact_edge_frequencies([0.1, 0.2, 0.3, 0.4]), [0.7, 0.6]
    )


def test_configuration_requires_a_replacement_move():
    for share in (0.0, 1.0):
        with pytest.raises(ValueError, match="swap probability"):
            Lab3Config(swap_probability=share)


def test_real_scorer_chain_satisfies_detailed_balance(rlc_result):
    rows = proposal_comparison(rlc_result)
    assert [row["proposal"] for row in rows] == [
        "равномерное",
        "по интервенции",
    ]
    for row in rows:
        assert row["balance_error"] < 1e-12
        assert row["hit_from_full"] > 1 and row["spectral_gap"] > 0


def test_edge_report_matches_the_selection(rlc_result):
    rows = edge_report(rlc_result)
    assert [row["selected"] for row in rows] == [
        True,
        True,
        False,
        False,
        False,
    ]
    for row in rows:
        keep = row["lab2_decision"] == "СОХРАНИТЬ"
        assert keep == row["selected"] == row["truth"]
        assert (row["instability"] < 0.1) == row["selected"]
        assert row["delta_loss_after"] > 0
    assert all(
        row["delta_loss_after"] > row["delta_loss_before"] for row in rows[2:]
    )


def test_saved_sample_contains_every_generated_graph(rlc_result, tmp_path):
    summary = save_results(tmp_path, {"rlc": rlc_result})
    with np.load(tmp_path / "results/lab3/chains.npz") as saved:
        for label, stage in rlc_result.stages.items():
            for start, chain in stage.chains.items():
                np.testing.assert_array_equal(
                    saved[f"rlc|{label}|{start}|states"], chain.states
                )
                assert len(chain.states) == TEST_CONFIG.steps + 1
    record = summary["problems"]["rlc"]["stages"][INTERVENTIONAL]
    assert record["selected"] == "{M, R₂}" and record["selected_is_truth"]
    (profile,) = summary["stability_profile"]
    assert profile["problem"] == "rlc" and profile["edges"] == [
        "M",
        "R₂",
        "Rₘ",
    ]
    assert np.shape(profile["ratio"]) == (len(profile["levels"]), 3)


def test_stored_notebook_is_the_executed_build():
    path = ROOT / "scripts/build_lab3.py"
    spec = importlib.util.spec_from_file_location("build_lab3", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    notebook = nbformat.read(ROOT / NOTEBOOK_PATH, as_version=4)
    stored = [(cell.cell_type, cell.source) for cell in notebook.cells]
    assert stored == builder.sources()
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert all(o.output_type != "error" for o in cell.outputs)


def test_notebook_contains_only_short_orchestration_cells():
    notebook = nbformat.read(ROOT / NOTEBOOK_PATH, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        assert len(cell.source.splitlines()) <= 12
        assert not any(
            isinstance(
                node, (ast.For, ast.While, ast.FunctionDef, ast.ClassDef)
            )
            for node in ast.walk(ast.parse(cell.source))
        )
