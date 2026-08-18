import numpy as np
from scipy.integrate import odeint, solve_ivp


def run_odeint(
    model_fn,
    x0,
    integration_time,
    params,
    time_points,
):
    """Run standard non-stiff ODE simulation using scipy.integrate.odeint (LSODA)."""
    def dynamics(x, t, params):
        dx, _ = model_fn(x, t, params)
        return dx

    t = np.linspace(0, integration_time, time_points)
    sol = odeint(dynamics, x0, t, args=(params,))
    outputs = [model_fn(sol[i], t[i], params)[1] for i in range(len(t))]
    return sol, outputs, t


def run_stiff(
    model_fn,
    x0,
    integration_time,
    params,
    time_points,
    method="Radau",
    rtol=1e-3,
    atol=1e-6,
):
    """Run stiff ODE simulation using scipy.integrate.solve_ivp (Radau by default)."""
    def dynamics(t, x):
        dx, _ = model_fn(x, t, params)
        return dx

    t_eval = np.linspace(0, integration_time, time_points)
    sol_res = solve_ivp(
        dynamics,
        (0, integration_time),
        x0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol_res.success:
        raise RuntimeError(f"Stiff solver failed: {sol_res.message}")
    sol = sol_res.y.T
    t = sol_res.t
    outputs = [model_fn(sol[i], t[i], params)[1] for i in range(len(t))]
    return sol, outputs, t
