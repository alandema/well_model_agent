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
