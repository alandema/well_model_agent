import numpy as np
from scipy.integrate import odeint


def run_model(model_fn, x0, t, u, param):
    """Integrate an ODE model over time and collect its outputs.

    The model function `model_fn(x, t, u, param)` must return a tuple
    `(dx, outputs)` where:
        - `dx` is the list of state derivatives used by the integrator.
        - `outputs` is a dict of algebraic quantities (e.g. pressures)
          computed from the current state.

    Args:
        model_fn: callable returning (dx, outputs).
        x0: initial state (array-like).
        t: time points (array-like).
        u: control inputs.
        param: model parameters.

    Returns:
        sol: state trajectory, shape (len(t), len(x0)).
        outputs: list of output dicts, one per time point.
    """
    def dynamics(x, t, u, param):
        dx, _ = model_fn(x, t, u, param)
        return dx

    sol = odeint(dynamics, x0, t, args=(u, param))
    outputs = [model_fn(sol[i], t[i], u, param)[1] for i in range(len(t))]
    return sol, outputs
