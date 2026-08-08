import numpy as np
from scipy.integrate import odeint


def run_model(
    model_fn,
    x0,
    t,
    gas_injection_rate,
    choke_opening,
    separator_pressure,
    reservoir_pressure,
    param,
):
    """Integrate an ODE model over time and collect its outputs.

    The model function `model_fn(x, t, gas_injection_rate, choke_opening,
    separator_pressure, reservoir_pressure, param)` must return a tuple
    `(dx, outputs)` where:
        - `dx` is the list of state derivatives used by the integrator.
        - `outputs` is a dict of algebraic quantities (e.g. pressures)
          computed from the current state.

    Args:
        model_fn: callable returning (dx, outputs).
        x0: initial state (array-like).
        t: time points (array-like).
        gas_injection_rate: Gas injection rate at standard conditions, in m³/day.
        choke_opening: Production choke opening, as a percentage from 0 to 100.
        separator_pressure: Separator pressure, in Pa.
        reservoir_pressure: Reservoir pressure, in Pa.
        param: model parameters.

    Returns:
        sol: state trajectory, shape (len(t), len(x0)).
        outputs: list of output dicts, one per time point.
    """
    def dynamics(
        x,
        t,
        gas_injection_rate,
        choke_opening,
        separator_pressure,
        reservoir_pressure,
        param,
    ):
        dx, _ = model_fn(
            x,
            t,
            gas_injection_rate,
            choke_opening,
            separator_pressure,
            reservoir_pressure,
            param,
        )
        return dx

    model_args = (
        gas_injection_rate,
        choke_opening,
        separator_pressure,
        reservoir_pressure,
        param,
    )
    sol = odeint(dynamics, x0, t, args=model_args)
    outputs = [model_fn(sol[i], t[i], *model_args)[1] for i in range(len(t))]
    return sol, outputs
