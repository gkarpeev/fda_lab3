"""Reader/tag circuits with every candidate coupling between them.

The laboratory 1 equation written for loop charges q = (q1, q2):

    L q'' + R q' + S q = e(t),

    L = [[L1, M], [M, L2]]         inductive coupling M,
    R = [[R1, Rm], [Rm, R2]]       resistive (shared path) coupling Rm,
    S = [[1/C1, Sm], [Sm, 1/C2]]   capacitive coupling Sm (mutual elastance),
    e = (1, E2) cos(wt)            E2: source pickup directly in the tag loop.

Laboratory 2 used Rm = Sm = E2 = 0; with these values the function below
repeats ``intervention.rlc_current`` operation by operation.
"""

import numpy as np

from .lab2_config import (
    FREQUENCY_HZ,
    L1_HENRY,
    L2_HENRY,
    NANOHENRY_TO_HENRY,
    R1_OHM,
)

PER_MICROFARAD = 1e6
MILLIVOLT = 1e-3


def coupled_current(times, theta):
    """Tag current i2(t) from the zero state.

    theta = (M [nH], R2 [ohm], Rm [ohm], Sm [1/uF], E2 [mV]).
    """
    times = np.asarray(times, dtype=float)
    mutual_inductance = float(theta[0]) * NANOHENRY_TO_HENRY
    resistance = float(theta[1])
    mutual_resistance = float(theta[2])
    mutual_elastance = float(theta[3]) * PER_MICROFARAD
    pickup = float(theta[4]) * MILLIVOLT
    if (
        mutual_inductance < 0
        or resistance <= 0
        or mutual_inductance**2 >= L1_HENRY * L2_HENRY
    ):
        raise ValueError("Non-passive parameters")

    omega = 2 * np.pi * FREQUENCY_HZ
    capacitances = 1 / (omega**2 * np.array([L1_HENRY, L2_HENRY]))
    inverse_inductance = np.linalg.inv(
        [[L1_HENRY, mutual_inductance], [mutual_inductance, L2_HENRY]]
    )
    resistances = np.array(
        [[R1_OHM, mutual_resistance], [mutual_resistance, resistance]]
    )
    # The state keeps capacitor voltages v = q / C (as in laboratory 2), so
    # the elastance matrix acts on C v; without Sm this product is identity.
    elastance_times_capacitance = np.array(
        [
            [1.0, mutual_elastance * capacitances[1]],
            [mutual_elastance * capacitances[0], 1.0],
        ]
    )
    matrix = np.zeros((4, 4))
    matrix[:2, 2:] = np.diag(1 / capacitances)
    matrix[2:, :2] = -inverse_inductance @ elastance_times_capacitance
    matrix[2:, 2:] = -inverse_inductance @ resistances
    forcing = np.r_[0.0, 0.0, inverse_inductance @ np.array([1.0, pickup])]
    particular = np.linalg.solve(1j * omega * np.eye(4) - matrix, forcing)
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    weights = np.linalg.solve(eigenvectors, -particular.real)
    transient = (eigenvectors[3] * weights) @ np.exp(
        eigenvalues[:, None] * times
    )
    return (particular[3] * np.exp(1j * omega * times) + transient).real
