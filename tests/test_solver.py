"""Small deterministic regression tests for iLQR and its constrained extension."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pendulum_ilqr.controls.ilqr import (
    _control_gradient,
    alilqr,
    augmented_cost,
    ilqr,
    penalty_matrix,
    rollout,
    total_cost,
)
from pendulum_ilqr.plants.pendulum import GRAVITY_COEFFICIENT


def _independent_rollout(x_0, controls, dt):
    """Compute the stated equations without using the solver's rollout helper."""
    states = [np.array(x_0, dtype=float)]
    for control in controls:
        angle, rate = states[-1]
        states.append(
            np.array(
                [
                    angle + dt * rate,
                    rate + dt * (control[0] + GRAVITY_COEFFICIENT * np.sin(angle)),
                ]
            )
        )
    return np.array(states)


def _central_gradient(function, controls, epsilon=1e-5):
    gradient = np.empty_like(controls)
    for index in np.ndindex(controls.shape):
        perturbation = np.zeros_like(controls)
        perturbation[index] = epsilon
        gradient[index] = (
            function(controls + perturbation) - function(controls - perturbation)
        ) / (2 * epsilon)
    return gradient


@pytest.fixture
def tracking_problem():
    return {
        "x_0": np.array([0.2, -0.1]),
        "u_t": np.zeros((30, 1)),
        "W": np.diag([20.0, 1.0]),
        "R": np.array([[0.02]]),
        "W_f": np.diag([100.0, 10.0]),
        "x_ref": np.zeros(2),
        "dt": 0.04,
    }


def test_rollout_has_one_more_state_than_controls():
    controls = np.array([[0.2], [-0.3], [0.7]])
    x_0 = np.array([0.3, -0.4])
    states = rollout(x_0, controls, 0.03)
    assert states.shape == (4, 2)
    assert_array_equal(states[0], x_0)
    assert_allclose(states, _independent_rollout(x_0, controls, 0.03), atol=1e-14)


def test_total_cost_includes_every_running_control_and_terminal_state():
    states = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    controls = np.array([[2.0], [3.0]])
    reference = np.ones(2)
    # Running state terms: 3 + 2*4 + 3*9 = 38. Control: 4*(4+9)=52.
    # Terminal term: 5*16 + 6*25 = 230. The objective includes a factor 1/2.
    assert total_cost(
        states, controls, np.diag([2.0, 3.0]), np.array([[4.0]]), np.diag([5.0, 6.0]), reference
    ) == pytest.approx(160.0)


def test_penalty_switches_off_only_satisfied_zero_multiplier_constraints():
    constraints = np.array([-0.2, -0.2, 0.0, 0.3])
    multipliers = np.array([0.0, 2.0, 0.0, 0.0])
    penalties = np.array([3.0, 4.0, 5.0, 6.0])
    assert_array_equal(penalty_matrix(constraints, multipliers, penalties), np.diag([0, 4, 5, 6]))


def test_augmented_cost_adjoint_matches_control_finite_differences():
    x_0, dt = np.array([0.11, -0.3]), 0.04
    controls = np.linspace(-0.2, 0.1, 5)[:, None]
    states = _independent_rollout(x_0, controls, dt)
    multipliers = np.array([0.0, 0.7, 0.2, 0.8, 0.0, 0.1])[:, None]
    penalties = np.arange(2.0, 8.0)[:, None]
    w, r, w_f = np.diag([3.0, 0.7]), np.array([[0.2]]), np.diag([8.0, 2.0])
    reference = np.array([0.02, 0.0])

    def objective(trial_controls):
        return augmented_cost(
            _independent_rollout(x_0, trial_controls, dt),
            trial_controls,
            multipliers,
            penalties,
            w,
            r,
            w_f,
            reference,
            min_position=0.1,
        )

    gradient = _control_gradient(
        states, controls, multipliers, penalties, w, r, w_f, reference, dt, min_position=0.1
    )
    assert_allclose(gradient, _central_gradient(objective, controls), rtol=1e-6, atol=1e-8)


def test_ilqr_reduces_cost_and_returns_a_nonlinear_trajectory(tracking_problem):
    problem = tracking_problem
    initial_states = _independent_rollout(problem["x_0"], problem["u_t"], problem["dt"])
    initial_cost = total_cost(
        initial_states, problem["u_t"], problem["W"], problem["R"], problem["W_f"], problem["x_ref"]
    )
    states, controls, info = ilqr(**problem)
    assert info["converged"], info["status"]
    assert info["cost"] < 0.1 * initial_cost
    assert info["stationarity"] <= 1e-6
    assert states.shape == (31, 2)
    assert controls.shape == (30, 1)
    assert_allclose(
        states, _independent_rollout(problem["x_0"], controls, problem["dt"]), atol=1e-12
    )
    assert np.all(np.diff(info["cost_history"]) < 0)
    assert_allclose(info["xs"][-1], states)
    assert_allclose(info["us"][-1], controls)


@pytest.mark.parametrize("solver", [ilqr, alilqr])
def test_exact_equilibrium_converges_without_an_update(solver):
    states, controls, info = solver(
        np.zeros(2), np.zeros((3, 1)), np.eye(2), np.eye(1), np.zeros(2), 0.04
    )
    assert info["converged"]
    assert info["accepted_steps"] == 0
    assert info["cost"] == 0.0
    assert info["stationarity"] == 0.0
    assert_array_equal(states, np.zeros((4, 2)))
    assert_array_equal(controls, np.zeros((3, 1)))
    assert info["xs"].shape == (1, 4, 2)
    assert info["us"].shape == (1, 3, 1)


def test_terminal_angle_bound_and_reported_kkt_residuals():
    # With two controls, x[1,0] is fixed and only the terminal angle is active.
    problem = {
        "x_0": np.array([0.2, -2.0]),
        "u_t": np.zeros((2, 1)),
        "W": np.diag([20.0, 1.0]),
        "R": np.array([[0.02]]),
        "W_f": np.diag([100.0, 10.0]),
        "x_ref": np.zeros(2),
        "dt": 0.04,
    }
    bound = 0.08
    states, controls, info = alilqr(**problem, min_position=bound)
    assert info["converged"], info["status"]
    assert_allclose(
        states, _independent_rollout(problem["x_0"], controls, problem["dt"]), atol=1e-12
    )
    constraints = bound - states[:, [0]]
    assert np.max(constraints) <= info["constraint_tol"]
    assert states[-1, 0] == pytest.approx(bound, abs=1e-6)
    assert info["lambda_t"].shape == (3, 1)
    assert np.all(info["lambda_t"] >= 0.0)
    assert info["lambda_t"][-1, 0] > 1.0
    assert info["constraint_violation"] == pytest.approx(np.max(np.maximum(constraints, 0.0)))
    complementarity = np.max(np.abs(info["lambda_t"] * constraints))
    assert info["complementarity"] == pytest.approx(complementarity)
    assert complementarity <= info["constraint_tol"]

    def lagrangian(trial_controls):
        trial_states = _independent_rollout(problem["x_0"], trial_controls, problem["dt"])
        objective = total_cost(
            trial_states,
            trial_controls,
            problem["W"],
            problem["R"],
            problem["W_f"],
            problem["x_ref"],
        )
        return objective + float(np.sum(info["lambda_t"] * (bound - trial_states[:, [0]])))

    stationarity = np.max(np.abs(_central_gradient(lagrangian, controls)))
    assert stationarity <= info["gradient_tol"]
    assert info["stationarity"] == pytest.approx(stationarity, abs=1e-8)


@pytest.mark.parametrize("solver", [ilqr, alilqr])
def test_solver_does_not_mutate_inputs_and_can_disable_trajectory_history(solver, tracking_problem):
    original = {
        key: value.copy()
        for key, value in tracking_problem.items()
        if isinstance(value, np.ndarray)
    }
    states, controls, info = solver(**tracking_problem, record_history=False)
    for key, expected in original.items():
        assert_array_equal(tracking_problem[key], expected)
    assert not np.shares_memory(states, tracking_problem["x_0"])
    assert not np.shares_memory(controls, tracking_problem["u_t"])
    assert "xs" not in info and "us" not in info


@pytest.mark.parametrize("solver", [ilqr, alilqr])
def test_control_vector_is_normalized_to_one_column(solver):
    states, controls, info = solver(
        np.zeros(2), np.zeros(3), np.eye(2), np.eye(1), np.zeros(2), 0.04
    )
    assert states.shape == (4, 2)
    assert controls.shape == (3, 1)
    assert info["converged"]


@pytest.mark.parametrize("solver", [ilqr, alilqr])
@pytest.mark.parametrize(
    "change",
    [
        {"x_0": np.zeros(3)},
        {"x_ref": np.zeros((2, 1))},
        {"u_t": np.empty((0, 1))},
        {"u_t": np.zeros((3, 2))},
        {"u_t": np.array([[np.nan]])},
        {"x_0": np.array([np.inf, 0.0])},
        {"W": np.ones((1, 1))},
        {"W": np.array([[1.0, 1.0], [0.0, 1.0]])},
        {"W_f": np.diag([-1.0, 1.0])},
        {"R": np.zeros((1, 1))},
        {"R": np.array([[np.inf]])},
        {"dt": 0.0},
        {"dt": np.nan},
        {"gradient_tol": -1.0},
        {"rho_init": -1.0},
        {"max_inner_iters": 0},
        {"max_inner_iters": 1.5},
    ],
)
def test_invalid_problem_is_rejected(solver, tracking_problem, change):
    with pytest.raises(ValueError):
        solver(**(tracking_problem | change))


@pytest.mark.parametrize(
    "change",
    [
        {"x_0": np.array([-0.1, 0.0])},
        {"min_position": 0.3},
        {"mu_init": 0.0},
        {"phi": 1.0},
        {"constraint_tol": 0.0},
        {"max_outer_iters": 0},
    ],
)
def test_invalid_constrained_problem_is_rejected(tracking_problem, change):
    with pytest.raises(ValueError):
        alilqr(**(tracking_problem | change))


def test_unconstrained_solver_accepts_initial_angles_below_zero(tracking_problem):
    states, _, info = ilqr(**(tracking_problem | {"x_0": np.array([-0.1, 0.0])}))
    assert states[0, 0] == -0.1
    assert info["converged"], info["status"]


@pytest.mark.parametrize("solver", [ilqr, alilqr])
def test_large_damping_does_not_masquerade_as_convergence(solver, tracking_problem):
    options = {"max_inner_iters": 1, "rho_init": 1e15, "record_history": False}
    if solver is alilqr:
        options["max_outer_iters"] = 1
    states, controls, info = solver(**tracking_problem, **options)
    assert not info["converged"]
    assert info["status"] != "converged"
    assert info["stationarity"] > info["gradient_tol"]
    assert np.isfinite(states).all()
    assert np.isfinite(controls).all()


@pytest.mark.parametrize("solver", [ilqr, alilqr])
def test_inactive_constraints_do_not_overflow_a_controls_only_objective(solver):
    states, controls, info = solver(
        np.array([1e160, 0.0]),
        np.ones((2, 1)),
        np.zeros((2, 2)),
        np.eye(1),
        np.zeros(2),
        0.01,
        W_f=np.zeros((2, 2)),
    )
    assert info["converged"], info["status"]
    assert np.isfinite(states).all()
    assert np.isfinite(info["cost"])
    assert info["cost"] < 1e-10
    assert_allclose(controls, 0.0, atol=1e-6)
    assert info["stationarity"] <= info["gradient_tol"]


@pytest.mark.parametrize("solver", [ilqr, alilqr])
def test_finite_initial_trajectory_with_overflowing_cost_is_rejected(solver):
    initial_state, controls, dt = np.array([1e200, 0.0]), np.zeros((2, 1)), 0.01
    assert np.isfinite(_independent_rollout(initial_state, controls, dt)).all()
    with pytest.raises(ValueError):
        solver(initial_state, controls, np.eye(2), np.eye(1), np.zeros(2), dt)


def test_ilqr_solution_is_stationary_for_a_nonzero_tracking_reference():
    x_0, reference, dt = np.array([0.1, 0.0]), np.array([0.2, -0.1]), 0.04
    controls = np.zeros((8, 1))
    w, r, w_f = np.diag([4.0, 0.5]), np.array([[0.02]]), np.diag([10.0, 1.0])
    states, optimized_controls, info = ilqr(x_0, controls, w, r, reference, dt, W_f=w_f)
    assert info["converged"], info["status"]
    assert_allclose(states, _independent_rollout(x_0, optimized_controls, dt), atol=1e-12)

    def reference_tracking_objective(trial_controls):
        errors = _independent_rollout(x_0, trial_controls, dt) - reference
        running_states = np.einsum("ni,ij,nj->", errors[:-1], w, errors[:-1])
        running_controls = np.einsum("ni,ij,nj->", trial_controls, r, trial_controls)
        terminal = errors[-1] @ w_f @ errors[-1]
        return float(0.5 * (running_states + running_controls + terminal))

    assert reference_tracking_objective(optimized_controls) < reference_tracking_objective(controls)
    numerical_gradient = _central_gradient(reference_tracking_objective, optimized_controls)
    assert np.max(np.abs(numerical_gradient)) <= info["gradient_tol"]
    assert info["stationarity"] == pytest.approx(np.max(np.abs(numerical_gradient)), abs=1e-8)
