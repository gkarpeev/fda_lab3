"""The sampler must leave exp(-L) invariant for any positive proposal."""

import numpy as np
import pytest

from graphmh import (
    Chain,
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
    running_total_variation,
    spectral_gap,
    stationary_acceptance,
    total_variation,
    transition_matrix,
    visit_frequencies,
)

EDGES = 4


@pytest.fixture(scope="module")
def problem():
    """Random losses and strongly non-uniform, graph-dependent weights."""
    generator = np.random.default_rng(7)
    size = 2**EDGES
    losses = 2.0 * generator.normal(size=size)
    flip_table = generator.uniform(0.1, 3.0, size=(size, EDGES))
    swap_table = generator.uniform(0.1, 3.0, size=(size, EDGES))
    proposal = Proposal(
        flip_weights=lambda mask: flip_table[mask_index(mask)],
        swap_weights=lambda mask: swap_table[mask_index(mask)],
        swap_probability=0.4,
    )
    return (lambda mask: losses[mask_index(mask)]), proposal, losses


def test_proposal_is_a_distribution_with_symmetric_support(problem):
    _, proposal, _ = problem
    for mask in all_masks(EDGES):
        moves = proposal.distribution(mask)
        assert sum(moves.values()) == pytest.approx(1.0)
        assert all(probability > 0 for probability in moves.values())
        for candidate in moves:
            assert mask in proposal.distribution(candidate)


def test_moves_are_flips_and_edge_count_preserving_swaps(problem):
    _, proposal, _ = problem
    mask = (True, False, True, False)
    changes = sorted(
        (sum(a != b for a, b in zip(mask, candidate)), sum(candidate))
        for candidate in proposal.distribution(mask)
    )
    assert changes == [(1, 1)] * 2 + [(1, 3)] * 2 + [(2, 2)] * 4


def test_flips_and_swaps_use_their_own_weights(problem):
    _, proposal, _ = problem
    mask = (True, False, True, False)
    moves = proposal.distribution(mask)
    flips = np.asarray(proposal.flip_weights(mask))
    swaps = np.asarray(proposal.swap_weights(mask))
    assert moves[flip(mask, 1)] == pytest.approx(0.6 * flips[1] / flips.sum())
    assert moves[flip(mask, 0, 3)] == pytest.approx(
        0.4 * swaps[0] * swaps[3] / (swaps[[0, 2]].sum() * swaps[[1, 3]].sum())
    )


def test_swap_share_stays_in_place_when_no_swap_exists(problem):
    _, proposal, _ = problem
    for mask in ((False,) * EDGES, (True,) * EDGES):
        assert proposal.distribution(mask)[mask] == pytest.approx(0.4)


def test_detailed_balance_and_stationarity(problem):
    loss, proposal, losses = problem
    matrix = transition_matrix(loss, proposal, EDGES)
    target = gibbs_distribution(losses)
    flow = target[:, None] * matrix
    assert matrix.min() >= 0
    np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
    np.testing.assert_allclose(flow, flow.T, atol=1e-15)
    np.testing.assert_allclose(target @ matrix, target, atol=1e-15)


def test_chain_is_irreducible_and_aperiodic(problem):
    loss, proposal, _ = problem
    matrix = transition_matrix(loss, proposal, EDGES)
    # A primitive matrix has a strictly positive power: every graph is
    # reachable from every other one after the same number of steps.
    assert np.all(np.linalg.matrix_power(matrix, 2 * EDGES) > 0)


def test_chain_frequencies_converge_to_the_target(problem):
    loss, proposal, losses = problem
    chain = metropolis_hastings(
        loss, proposal, (True,) * EDGES, 40000, np.random.default_rng(11)
    )
    target = gibbs_distribution(losses)
    assert total_variation(visit_frequencies(chain.states), target) < 0.03
    assert 0 < chain.acceptance_rate < 1


def test_sampler_follows_the_exact_transition_matrix(problem):
    """The loop itself, not only its formulas: empirical one-step moves."""
    loss, proposal, _ = problem
    matrix = transition_matrix(loss, proposal, EDGES)
    chain = metropolis_hastings(
        loss, proposal, (False,) * EDGES, 120000, np.random.default_rng(5)
    )
    index = mask_index(chain.states)
    counts = np.zeros_like(matrix)
    np.add.at(counts, (index[:-1], index[1:]), 1)
    visits = counts.sum(axis=1)
    often = visits > 2000
    assert often.sum() >= 6
    empirical = counts[often] / visits[often, None]
    # Five standard errors of a frequency; its variance is at most 1/(4n).
    tolerance = 2.5 / np.sqrt(visits[often, None])
    assert np.all(np.abs(empirical - matrix[often]) < tolerance)


def test_chain_records_proposals_and_keeps_state_after_rejection(problem):
    loss, proposal, losses = problem
    chain = metropolis_hastings(
        loss, proposal, (False,) * EDGES, 500, np.random.default_rng(3)
    )
    moved = np.any(chain.states[1:] != chain.states[:-1], axis=1)
    np.testing.assert_array_equal(moved, chain.accepted)
    np.testing.assert_array_equal(
        chain.states[1:][chain.accepted], chain.proposed[chain.accepted]
    )
    np.testing.assert_allclose(chain.losses, losses[mask_index(chain.states)])


def test_acceptance_ratio_contains_the_proposal_correction(problem):
    loss, proposal, losses = problem
    mask = (True, False, False, True)
    candidate = flip(mask, 1)
    forward = proposal.distribution(mask)[candidate]
    backward = proposal.distribution(candidate)[mask]
    expected = (
        losses[mask_index(mask)]
        - losses[mask_index(candidate)]
        + np.log(backward / forward)
    )
    assert forward != pytest.approx(backward)
    assert log_acceptance_ratio(loss, proposal, mask, candidate) == (
        pytest.approx(expected)
    )


def test_invalid_weights_and_losses_are_rejected():
    start = (True, False)
    generator = np.random.default_rng(0)
    for weights in ([1.0, 0.0], [1.0, -1.0], [1.0, np.inf], [1.0]):
        with pytest.raises(ValueError, match="weight"):
            Proposal(flip_weights=lambda mask: weights).distribution(start)
    with pytest.raises(ValueError, match="not finite"):
        metropolis_hastings(
            lambda mask: np.inf, Proposal(), start, 1, generator
        )
    with pytest.raises(ValueError, match="Swap probability"):
        Proposal(swap_probability=1.0)


def test_exact_diagnostics_on_a_two_state_chain():
    # One edge: P = [[1 - a, a], [b, 1 - b]] with a = 1, b = exp(-1).
    matrix = transition_matrix(lambda mask: float(not mask[0]), Proposal(), 1)
    up, down = matrix[0, 1], matrix[1, 0]
    assert (up, down) == (pytest.approx(1.0), pytest.approx(np.exp(-1.0)))
    np.testing.assert_allclose(expected_hitting_time(matrix, 1), [1 / up, 0])
    np.testing.assert_allclose(expected_hitting_time(matrix, 0), [0, 1 / down])
    assert spectral_gap(matrix) == pytest.approx(1 - abs(1 - up - down))
    target = gibbs_distribution([1.0, 0.0])
    assert stationary_acceptance(matrix, target) == pytest.approx(
        target[0] * up + target[1] * down
    )


def test_selection_from_the_stored_sample(problem):
    loss, proposal, losses = problem
    chains = [
        metropolis_hastings(
            loss, proposal, start, 300, np.random.default_rng(seed)
        )
        for seed, start in enumerate(((True,) * EDGES, (False,) * EDGES))
    ]
    mask, value = best_visited(chains)
    visited = np.concatenate([chain.states for chain in chains])
    assert value == losses[mask_index(visited)].min()
    assert loss(mask) == value
    np.testing.assert_allclose(edge_frequencies(visited), visited.mean(axis=0))
    assert visit_frequencies(visited).sum() == pytest.approx(1.0)


def _stored(states, losses):
    """A chain that only carries the stored sample."""
    states = np.array(states, dtype=bool)
    steps = len(states) - 1
    return Chain(
        states,
        np.array(losses, dtype=float),
        states[1:],
        np.zeros(steps, dtype=bool),
        np.zeros(steps),
    )


def test_selection_takes_the_best_graph_of_all_chains():
    first = _stored([(True, False), (True, True)], [3.0, 2.0])
    second = _stored([(False, False), (False, True)], [5.0, 1.0])
    assert best_visited([first, second]) == ((False, True), 1.0)
    assert best_visited([second, first]) == ((False, True), 1.0)


def test_running_distance_uses_chain_prefixes():
    states = np.array([(False,), (False,), (True,), (True,)])
    exact = np.array([0.25, 0.75])
    np.testing.assert_allclose(
        running_total_variation(states, exact, [1, 2, 3, 4]),
        [0.75, 0.75, 5 / 12, 0.25],
    )
