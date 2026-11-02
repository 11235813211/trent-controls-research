# Rocket plant model

The implementation in `src/rocket_landing` is a local free-flight model for
recovery-landing studies. The plant evaluates gravity, aerodynamics, and thrust;
the simulation layer propagates it and collects Jacobians. No rocket trajectory
optimizer or feedback controller is implemented.

## Frames, state, and input

Both frames are right handed. The world frame `W` is a local East-North-Up frame,
with positive altitude along +Z. The body frame `B` is fixed to the vehicle, with
+Z toward the nose. Position is the vehicle mass-center position in `W`.
The local world frame is treated as inertial over the modeled descent segment.

The stored state and commanded input are

```text
x = [r_W(3), v_W(3), q_BW(4), omega_B(3), m]   (14 entries)
u = T_B                                      (3 entries)
```

`r_W` is in meters, `v_W` in m/s, `omega_B` in rad/s, mass in kg, and thrust
in newtons. `omega_B` is the body-relative-world angular velocity, resolved in
body axes. Thrust is the force on the vehicle, so positive body-Z thrust points
toward the nose. Each control vector is held constant during one discrete step.

The quaternion has unit norm. Plant methods normalize nonzero quaternion inputs,
require finite inputs and positive mass, and do not mutate caller-owned state
or control arrays.

## Quaternion convention

The convention is **left, passive, scalar first**. Write a quaternion as
`q = [s, v]`, where `s` is scalar and `v` has three components. Define
`skew(a) @ b = a cross b`. The implemented left product is

```text
[s1, v1] ⊗L [s2, v2]
    = [s1*s2 - dot(v1, v2), s1*v2 + s2*v1 - cross(v1, v2)]
```

For a unit quaternion, the passive world-to-body rotation is

```text
v_B = C_BW(q) @ v_W
C_BW(q) = (s² - dot(v, v))*I + 2*v*v' - 2*s*skew(v)
C_BW(q ⊗L p) = C_BW(q) @ C_BW(p)
```

`C_BW(q).T` maps body vectors into world coordinates. For a positive
axis-angle vector `phi`, `from_rotvec(phi)` returns
`[cos(|phi|/2), sin(|phi|/2)*phi/|phi|]`, with its continuous identity limit.
Consequently, `C_BW(from_rotvec(phi)) = exp(-skew(phi))`.

The rate equation is

```text
q_dot = 0.5 * [0, omega_B] ⊗L q
C_BW_dot = -skew(omega_B) @ C_BW
```

Quaternions `q` and `-q` represent the same attitude. The implementation preserves
the quaternion sign during normalization and uses sign-independent local
attitude differences.

The word “left” describes the negative-cross quaternion algebra here; it does
not change either coordinate frame's handedness. JPL's
[quaternion convention discussion](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/q2m_c.html)
shows why element order, multiplication, and matrix interpretation must all be
specified. Its SPICE/Hamilton product uses a positive cross term. This project's
formulas above define its convention; its quaternion components must be converted
before being used with SPICE's quaternion-to-matrix routine.

## Loads

The mass center is the body origin for the load offsets. Let `r_A_B` denote the
pressure-center offset and `r_T_B` the thrust-application offset. Both are fixed
in the body. The three load families are evaluated separately by `plant.loads`.

NASA's [rocket force overview](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/four-rocket-forces/)
describes weight, thrust, aerodynamic loads at the pressure center, and thrust
moments from an offset line of action. The particular drag law and parameter
values below are modeling choices in this repository.

### Gravity

Gravity is uniform and downward in the local world frame:

```text
F_g_W = [0, 0, -m*g]
```

The default `g` is `9.80665 m/s²`. Gravity acts through the mass center, producing
no moment about that point. Earth rotation, curvature, and altitude-dependent
gravity are outside this local model.

### Aerodynamics

The atmosphere has uniform world-frame wind `w_W` and exponential density:

```text
rho(z) = rho0 * exp(-z/H)
v_A_B = C_BW(q) @ (v_W - w_W) + cross(omega_B, r_A_B)
D_B = diag(drag_area_body)
F_a_B = -0.5 * rho(z) * norm(v_A_B) * D_B @ v_A_B
M_a_B = cross(r_A_B, F_a_B)
```

`drag_area_body` contains three nonnegative `Cd*A` products in m². The velocity
at the pressure center includes rigid-body rotation. Since `D_B` is nonnegative,
`dot(F_a_B, v_A_B) <= 0`: the force dissipates energy relative to that local
flow. Anisotropic coefficients can make the force nonparallel to the relative
velocity.

This is a lumped force law, with no Mach/angle-of-attack coefficient tables,
explicit lift model, fin dynamics, independent roll damping, or CFD fit.
The density profile has no ground clamp or atmosphere-layer switching. Zero
air-density configuration disables aerodynamic loads.

### Thrust and optional mass depletion

The input is the applied body-frame force itself:

```text
F_t_B = u
M_t_B = cross(r_T_B, u)
```

There is no internal propulsive state, thrust lag, or gimbal actuator. The
input accepts any finite three-component force; feasible throttle and pointing
constraints belong in a later controls layer.

Mass stays constant by default (`specific_impulse=None`). A positive specific
impulse `Isp` in seconds enables

```text
m_dot = -norm(u) / (Isp * g0),  g0 = 9.80665 m/s²
```

This follows the thrust-to-propellant-weight-flow definition of
[specific impulse](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/specific-impulse/).
Here `g0` remains standard gravity even if the local gravity parameter changes.
This algebraic model changes translational mass only. Inertia, mass-center
location, and both load offsets remain fixed. No dry-mass boundary or propellant
exhaustion logic is modeled; evaluations reaching nonpositive mass are rejected.

The exact norm in mass flow is nondifferentiable at zero thrust. Both Jacobian
methods reject that operating point when depletion is enabled. Zero-thrust
simulation remains valid, and zero-thrust Jacobians are available with fixed
mass.

## Continuous dynamics and integration

With fixed symmetric positive-definite body inertia `J`, the derivative is

```text
r_W_dot = v_W
v_W_dot = (F_g_W + C_BW(q).T @ (F_a_B + u)) / m
q_dot = 0.5 * [0, omega_B] ⊗L q
omega_B_dot = solve(J, M_a_B + M_t_B - cross(omega_B, J @ omega_B))
m_dot = 0                                  (fixed-mass default)
      = -norm(u)/(Isp*g0)                   (optional depletion)
```

`step(x, u, dt)` uses classical four-stage RK4 with positive `dt` and held
thrust. Derivative evaluations normalize their quaternion inputs, and the
accepted output quaternion is normalized as well. No artificial norm-stabilizing
term is added to the differential equation. The integration map includes no
ground contact, clipping, control saturation, or touchdown event.

The default parameter set is illustrative:

| Parameter | Value |
| --- | --- |
| Inertia `J` | `diag(8000, 8000, 500) kg m²` |
| Thrust offset `r_T_B` | `[0, 0, -3] m` |
| Pressure-center offset `r_A_B` | `[0, 0, 1] m` |
| Drag-area products | `[8, 8, 0.5] m²` |
| Density at zero altitude | `1.225 kg/m³` |
| Density scale height | `8500 m` |
| Wind | `[0, 0, 0] m/s` |

These values do not describe an identified or flight-validated vehicle.

## Linearization interface

Both Jacobian methods use central finite differences, not analytic derivatives
or automatic differentiation. Their default perturbation scale is
`relative_step=1e-5`. Perturbations scale with `max(1, abs(component))`; attitude
perturbations use radians at unit scale. The mass perturbation is reduced when
needed to keep the perturbed mass positive.

`plant.linearize_continuous(x, u)` returns `A_c.shape == (14, 14)` and
`B_c.shape == (14, 3)`, differentiating the continuous derivative as implemented.
Normalization defines its extension away from the unit-quaternion surface.
The four quaternion columns therefore include a redundant radial coordinate;
they are not four independent physical attitude degrees of freedom.

`plant.linearize_discrete(x, u, dt)` returns `A.shape == (13, 13)` and
`B.shape == (13, 3)`. It differentiates the complete RK4 step in local coordinates:

```text
delta = [dr_W(3), dv_W(3), dphi_B(3), domega_B(3), dm]
x ⊞ delta:
    r_new = r + dr, v_new = v + dv
    q_new = from_rotvec(dphi_B) ⊗L q
    omega_new = omega + domega, m_new = m + dm

x ⊟ x_ref:
    attitude component = to_rotvec(q ⊗L conjugate(q_ref))
    all other components = ordinary subtraction

x_next = step(x, u, dt)
step(x ⊞ delta_x, u + delta_u, dt) ⊟ x_next
    ≈ A @ delta_x + B @ delta_u
```

The attitude logarithm uses its principal rotation vector, with norm at most
pi. The chart is local and cannot remain continuous across pi-radian errors.
Body angular-velocity components use direct addition in the stated body-axis
coordinate convention.

`rollout(plant, x0, controls, dt)` returns `N+1` states for `N` controls.
`linearize_trajectory(plant, states, controls, dt)` collects discrete matrices
with shapes `(N, 13, 13)` and `(N, 13, 3)`. It requires a dynamically consistent
nominal trajectory; inconsistent states are rejected because a defect term and
a different output chart would be needed. Generate the nominal states with
`rollout` before collecting these matrices.

These interfaces provide the plant and coordinate machinery a future iLQR or
AL-iLQR implementation can consume. Costs, constraints, optimizer iterations,
and feedback laws remain separate work. The existing pendulum solvers retain
their original two-state plant and are documented in
[the pendulum example](pendulum/README.md).

## Simulation example

The default `maneuver` example in the [repository README](../README.md) starts
with a 1000 kg upright vehicle at `(0, 0, 50) m`, velocity `(0, 0, -10) m/s`,
and zero angular rate. The default atmosphere and all load offsets remain
enabled. Five fixed body-thrust commands are held over a six-second flight,
using `dt=0.02 s` and `g=9.80665 m/s²`:

| Time (s) | Phase | Tx (N) | Ty (N) | Tz (N) |
| --- | --- | ---: | ---: | ---: |
| 0–0.8 | X steering pulse | 400 | 0 | 9806.65 |
| 0.8–1.6 | Reverse X / start Y steering | -400 | 280 | 11206.65 |
| 1.6–2.4 | Reverse Y steering | 0 | -280 | 11206.65 |
| 2.4–4.5 | Axial braking | 0 | 0 | 11806.65 |
| 4.5–6.0 | Small rate-trimming pulse | 80 | 45 | 11806.65 |

Each transverse pulse creates a moment through the tail offset `r_T_B`, as
well as a lateral force. The resulting tilt changes the direction of axial
thrust in the world frame. Opposing pulses limit the rotation, and increasing
axial thrust slows the descent. The last small, preselected pulse reduces the
angular rate produced by the aerodynamic moment. Every command is constructed
before integration; no command depends on the simulated state, and no optimizer
or controller runs.

The nominal final position is approximately `(-10.60, -6.12, 12.52) m`, with
vertical velocity `-0.66 m/s`. Tilt peaks at `6.30°` and ends at `2.00°`; final
angular speed is `0.16°/s`. The rocket stays above ground and keeps descending
throughout the segment. It still has lateral velocity at the end. These bounds
describe this finite nominal flight; the schedule provides no disturbance
rejection or touchdown logic.

The figure shows the spatial path and sampled body +Z directions, plus altitude,
world velocity, body thrust, tilt, angular velocity, and quaternion attitude.
The scalar `qw` and vector `[qx, qy, qz]` have separate panels to reveal their
different scales. These are the stored scalar-first, passive world-to-body
quaternion components. Steering and axial thrust use separately labeled force
scales so the small steering pulses remain visible. The CSV stores every held
input and state; the JSON includes the phase table and attitude extrema.

Use `--scenario braking` for the original 30 m, -10 m/s, four-second vertical
case with constant `[0, 0, m*(g + 2)] N` thrust. The `hover` and `coast` scenarios
provide thrust balance and unpowered descent examples. All scenarios use the
same free-flight equations. Contact and landing success criteria must be
introduced explicitly when the simulation expands.
