"""Metropolis-Hastings over graphs encoded as masks of candidate edges.

A graph is a subset of K candidate edges, i.e. a vertex of the cube {0,1}^K.
The sampler knows nothing about the model behind a graph. It needs

* ``loss(mask)``: a finite number L(G); the target is pi(G) ~ exp(-L(G));
* a ``Proposal``: positive edge weights that say which edges the next
  move prefers to change.

Moves follow structure MCMC (Madigan & York, 1995): add an edge, delete an
edge, or replace one present edge by an absent one.

Loss and weights must be deterministic functions of the mask: the acceptance
ratio compares two evaluations of the same function, so a loss that changes
between calls would silently change the target.
"""

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Chain:
    """Sampled graphs; ``states[0]`` is the start, row t+1 follows step t.

    ``proposed[t]`` is the graph offered at step t whether or not it was
    accepted, so the whole generation sequence can be replayed.
    """

    states: np.ndarray
    losses: np.ndarray
    proposed: np.ndarray
    accepted: np.ndarray
    log_ratio: np.ndarray

    @property
    def acceptance_rate(self):
        return float(self.accepted.mean())


def uniform_weights(mask):
    """Proposal that treats every candidate edge equally."""
    return np.ones(len(mask))


def flip(mask, *edges):
    """Return the graph that differs from ``mask`` in the given edges."""
    return tuple(
        not present if index in edges else present
        for index, present in enumerate(mask)
    )


def _weights(function, mask):
    weights = np.asarray(function(mask), dtype=float)
    if weights.shape != (len(mask),):
        raise ValueError("One weight per candidate edge is required")
    # A zero weight would make a neighbour unreachable in one direction only
    # and break both reversibility and irreducibility.
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Edge weights must be positive and finite")
    return weights


@dataclass(frozen=True)
class Proposal:
    """Mixture of single-edge flips and replacements of one edge by another.

    Flip: edge e with probability w_e / sum(w), w = ``flip_weights(G)``.
    Swap (drawn with ``swap_probability``): a present edge r and an absent
    edge a with probability v_r v_a / (sum of v over present * sum of v
    over absent), v = ``swap_weights(G)``.
    """

    flip_weights: Callable = uniform_weights
    swap_weights: Callable = uniform_weights
    swap_probability: float = 0.0

    def __post_init__(self):
        if not 0 <= self.swap_probability < 1:
            raise ValueError("Swap probability must lie in [0, 1)")

    def distribution(self, mask):
        """Every graph that can be proposed from G, with q(G' | G).

        A flip changes the edge count and a swap does not, so each neighbour
        is produced by exactly one move and the values are the complete
        proposal probabilities. When no swap exists (empty or full graph)
        its share stays on G itself.
        """
        weights = _weights(self.flip_weights, mask)
        flip_share = (1 - self.swap_probability) / weights.sum()
        moves = {
            flip(mask, edge): flip_share * weights[edge]
            for edge in range(len(mask))
        }
        if self.swap_probability > 0:
            present = [edge for edge, value in enumerate(mask) if value]
            absent = [edge for edge, value in enumerate(mask) if not value]
            if present and absent:
                weights = _weights(self.swap_weights, mask)
                swap_share = self.swap_probability / (
                    weights[present].sum() * weights[absent].sum()
                )
                for removed in present:
                    for added in absent:
                        moves[flip(mask, removed, added)] = (
                            swap_share * weights[removed] * weights[added]
                        )
            else:
                moves[mask] = self.swap_probability
        return moves


def _finite_loss(loss, mask):
    value = float(loss(mask))
    # pi(G) > 0 for every graph is what makes the chain irreducible.
    if not np.isfinite(value):
        raise ValueError(f"Loss of graph {mask} is not finite")
    return value


def log_acceptance_ratio(loss, proposal, mask, candidate):
    """log [pi(G') q(G | G')] - log [pi(G) q(G' | G)].

    Both proposal probabilities come from ``Proposal.distribution``, so the
    sampler, the exact transition matrix and this ratio cannot disagree
    about what the proposal is.
    """
    forward = proposal.distribution(mask)[candidate]
    backward = proposal.distribution(candidate)[mask]
    return (
        _finite_loss(loss, mask)
        - _finite_loss(loss, candidate)
        + np.log(backward)
        - np.log(forward)
    )


def metropolis_hastings(loss, proposal, start, steps, rng):
    """Run one chain; keep every state, including repeats after rejection."""
    mask = tuple(bool(value) for value in start)
    states = np.empty((steps + 1, len(mask)), dtype=bool)
    losses = np.empty(steps + 1)
    proposed = np.empty((steps, len(mask)), dtype=bool)
    accepted = np.zeros(steps, dtype=bool)
    log_ratio = np.zeros(steps)
    states[0], losses[0] = mask, _finite_loss(loss, mask)
    for step in range(steps):
        moves = proposal.distribution(mask)
        candidates = list(moves)
        choice = rng.choice(len(candidates), p=list(moves.values()))
        candidate = candidates[choice]
        proposed[step] = candidate
        # candidate == mask only when a swap was drawn but none exists.
        if candidate != mask:
            log_ratio[step] = log_acceptance_ratio(
                loss, proposal, mask, candidate
            )
            if np.log(rng.uniform()) < log_ratio[step]:
                mask, accepted[step] = candidate, True
        states[step + 1], losses[step + 1] = mask, _finite_loss(loss, mask)
    return Chain(states, losses, proposed, accepted, log_ratio)


def all_masks(edge_count):
    """Every graph on the candidate edges, ordered as binary numbers."""
    return list(product((False, True), repeat=edge_count))


def mask_index(states):
    """Position of each mask in ``all_masks``; accepts one mask or an array."""
    states = np.asarray(states, dtype=bool)
    powers = 1 << np.arange(states.shape[-1] - 1, -1, -1)
    return states @ powers


def gibbs_distribution(losses):
    """Normalised exp(-L); the minimum is subtracted to avoid underflow."""
    losses = np.asarray(losses, dtype=float)
    weights = np.exp(-(losses - losses.min()))
    return weights / weights.sum()


def transition_matrix(loss, proposal, edge_count):
    """Exact transition probabilities of the chain between all graphs.

    Built from the same proposal and acceptance functions as the sampler, so
    detailed balance of the implementation can be verified numerically.
    """
    masks = all_masks(edge_count)
    matrix = np.zeros((len(masks), len(masks)))
    for row, mask in enumerate(masks):
        for candidate, probability in proposal.distribution(mask).items():
            if candidate == mask:
                continue
            ratio = log_acceptance_ratio(loss, proposal, mask, candidate)
            matrix[row, mask_index(candidate)] = probability * np.exp(
                min(0.0, ratio)
            )
        matrix[row, row] = 1.0 - matrix[row].sum()
    return matrix


def visit_frequencies(states):
    """Share of chain steps spent in each graph, indexed as ``all_masks``."""
    states = np.asarray(states, dtype=bool)
    counts = np.bincount(mask_index(states), minlength=2 ** states.shape[1])
    return counts / len(states)


def total_variation(first, second):
    """Half the L1 distance between two distributions over graphs."""
    return 0.5 * float(np.abs(np.asarray(first) - np.asarray(second)).sum())


def running_total_variation(states, exact, checkpoints):
    """Distance between the exact target and the chain prefix frequencies."""
    return np.array(
        [
            total_variation(visit_frequencies(states[:length]), exact)
            for length in checkpoints
        ]
    )


def expected_hitting_time(matrix, target):
    """Mean number of steps to reach ``target`` from every graph.

    Solves (I - Q) h = 1, where Q is the transition matrix without the
    target row and column; the entry of the target itself is zero.
    """
    keep = np.arange(len(matrix)) != target
    reduced = np.eye(keep.sum()) - matrix[np.ix_(keep, keep)]
    times = np.zeros(len(matrix))
    times[keep] = np.linalg.solve(reduced, np.ones(keep.sum()))
    return times


def spectral_gap(matrix):
    """1 - |second largest eigenvalue|: the geometric rate of convergence."""
    moduli = np.sort(np.abs(np.linalg.eigvals(matrix)))
    return float(1.0 - moduli[-2])


def stationary_acceptance(matrix, distribution):
    """Share of accepted proposals once the chain has reached its target."""
    return float(distribution @ (1.0 - np.diag(matrix)))


def edge_frequencies(states):
    """Share of sampled graphs that contain each candidate edge."""
    return np.asarray(states, dtype=bool).mean(axis=0)


def best_visited(chains):
    """Graph with the smallest loss among everything the chains stored."""
    chain = min(chains, key=lambda item: item.losses.min())
    step = int(chain.losses.argmin())
    mask = tuple(bool(value) for value in chain.states[step])
    return mask, float(chain.losses[step])
