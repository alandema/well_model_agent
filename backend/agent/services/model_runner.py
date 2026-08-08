import numpy as np
from scipy.integrate import odeint


def run_model(
    model_fn,
    x0,
    integration_time,
    params,
    time_points,
):
    def dynamics(
        x,
        t,
        params,
    ):
        dx, _ = model_fn(x, t, params)
        return dx

    t = np.linspace(0, integration_time, time_points)
    sol = odeint(dynamics, x0, t, args=(params,))
    outputs = [model_fn(sol[i], t[i], params)[1] for i in range(len(t))]
    return sol, outputs, t
