import csv
import os
import uuid

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import tool

from agent.services.model_runner import run_model

# Directory where model outputs are written as CSV files.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", ".outputs")

# Default model parameters (from the reference FOWM script).
DEFAULT_PARAM = (np.multiply(1e2, [
    7.109820799914321, 0.000000234607899, 0.000058137935671, 0.901595193514190,
    0.000358225582262, 0.000010212053760, 0.000001766624933, 2.467164730000336,
])).tolist()

# Default initial state (from the reference FOWM script).
DEFAULT_X0 = (np.multiply(1e4, [
    0.762949953300966, 0.150646645264105, 2.024926259090548,
    0.213535823438009, 0.113058624767771, 1.519684541979486,
])).tolist()

# Default time grid (from the reference FOWM script).
DEFAULT_T = np.linspace(0, 100000, 1001).tolist()


def fowm(
    x,
    t,
    gas_injection_rate,
    choke_opening,
    separator_pressure,
    reservoir_pressure,
    param,
):
    """FOWM (Fast Offshore Well Model) dynamics and pressures.

    Returns a tuple `(dx, outputs)` where:
        - `dx` is the list of state derivatives (mass balances).
        - `outputs` is a dict of pressures computed from the state.

    Args:
        x: state variables [x1..x6].
        t: time (unused, the model is time-invariant).
        gas_injection_rate: Gas injection rate at standard conditions, in m³/day.
        choke_opening: Production choke opening, as a percentage from 0 to 100.
        separator_pressure: Separator pressure, in Pa.
        reservoir_pressure: Reservoir pressure, in Pa.
        param: parameters [mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr].
    """
    x1, x2, x3, x4, x5, x6 = x
    mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr = param

    # Physical constants
    Rol = 900.0
    R = 8314.0
    T = 298.0
    M = 18.0
    g = 9.81
    ALFAgw = 0.0188
    Romres = 891.9523

    # Geometry
    A = 0.018145839167135
    Vr = 81.601838734604499
    Vt = 28.963520770689396
    Va = 17.210272874895608
    La = 1118.0
    Hvgl = 916.0
    Hpdg = 1117.0
    Ht = 1279.0

    # Control inputs
    Ps = separator_pressure
    Pr = reservoir_pressure
    Wgc = gas_injection_rate * 101325.0 * M / (293.0 * R) / 3600.0 / 24.0
    z = choke_opening / 100.0

    # RISER + PIPELINE
    Peb = x1 * R * T / (M * Veb)
    Prt = x2 * R * T / (M * (Vr - (x3 + mlstill) / Rol))
    Prb = Prt + (x3 + mlstill) * g * 0.7071 / A

    ALFAg = x2 / (x2 + x3)
    ALFAl = 1.0 - ALFAg

    Wout = Cout * z * np.sqrt(max(0.0, Rol * (Prt - Ps)))
    Wlout = ALFAl * Wout
    Wgout = ALFAg * Wout
    Wg = Cg * max(0.0, (Peb - Prb))

    # TUBING
    Vgt = Vt - x6 / Rol
    ROgt = x5 / Vgt
    ROmt = (x5 + x6) / Vt

    Ptt = ROgt * R * T / M
    Ptb = Ptt + ROmt * g * Hvgl
    Ppdg = Ptb + Romres * g * (Hpdg - Hvgl)
    Pbh = Ppdg + Romres * g * (Ht - Hpdg)

    ALFAgt = x5 / (x6 + x5)

    Wwh = Kw * np.sqrt(ROmt * max(0.0, (Ptt - Prb)))
    Wwhg = Wwh * ALFAgt
    Wwhl = Wwh * (1.0 - ALFAgt)
    Wr = max(0.0, Kr * (1.0 - 0.2 * Pbh / Pr - 0.8 * (Pbh / Pr) ** 2))

    # ANNULAR
    Pai = ((R * T / (Va * M)) + (g * La / Va)) * x4
    ROai = M * Pai / (R * T)
    Wiv = Ka * np.sqrt(ROai * max(0.0, (Pai - Ptb)))

    # ODE (mass balances)
    dx1 = (1.0 - E) * Wwhg - Wg
    dx2 = E * Wwhg + Wg - Wgout
    dx3 = Wwhl - Wlout
    dx4 = Wgc - Wiv
    dx5 = Wr * ALFAgw + Wiv - Wwhg
    dx6 = Wr * (1.0 - ALFAgw) - Wwhl

    dx = [dx1, dx2, dx3, dx4, dx5, dx6]
    outputs = {"Ppdg": Ppdg, "Ptt": Ptt, "Prt": Prt, "Prb": Prb}
    return dx, outputs


class FowmModelInput(BaseModel):
    """Input schema for the FOWM model tool."""
    x0: list[float] = Field(
        default=None,
        description="Initial state variables [x1..x6]. If omitted, the default FOWM initial state is used.",
    )
    t: list[float] = Field(
        default=None,
        description="Time points to integrate over. If omitted, the default FOWM time grid is used.",
    )
    gas_injection_rate: float = Field(
        default=165000.0,
        ge=0.0,
        description=(
            "Gas injection rate at standard conditions, in m³/day. "
            "This is the gas injected into the annulus for gas lift."
        ),
    )
    choke_opening: float = Field(
        default=16.0,
        ge=0.0,
        le=100.0,
        description=(
            "Production choke opening as a percentage from 0 to 100. "
            "For example, 16 means the choke is 16% open."
        ),
    )
    separator_pressure: float = Field(
        default=101325.0,
        gt=0.0,
        description=(
            "Pressure at the separator or riser outlet, in Pa. "
            "This is the model's downstream pressure (Ps)."
        ),
    )
    reservoir_pressure: float = Field(
        default=2.25e7,
        gt=0.0,
        description=(
            "Reservoir pressure driving production inflow, in Pa. "
            "This is the model's reservoir pressure (Pr)."
        ),
    )
    param: list[float] = Field(
        default=None,
        description="Model parameters. If omitted, the default FOWM parameters are used.",
    )


@tool(
    args_schema=FowmModelInput,
    description=(
        "FOWM (Fast Offshore Well Model) tool for simulating offshore well dynamics. "
        "Use it to predict well behavior and pressures for a specified initial state, "
        "time grid, gas injection rate, choke opening, separator pressure, reservoir "
        "pressure, and model parameters."
    ),
)
def fowm_model(
    gas_injection_rate: float,
    choke_opening: float,
    separator_pressure: float,
    reservoir_pressure: float,
    x0: list[float] = None,
    t: list[float] = None,
    param: list[float] = None
) -> str:
    """Run the FOWM model and write the results to a CSV file.

    Returns the absolute path to the generated CSV file.
    """

    if x0 is None:
        x0 = DEFAULT_X0
    if t is None:
        t = DEFAULT_T
    if param is None:
        param = DEFAULT_PARAM

    sol, outputs = run_model(
        fowm,
        x0,
        t,
        gas_injection_rate,
        choke_opening,
        separator_pressure,
        reservoir_pressure,
        param,
    )

    # Build the CSV: one row per time point with t, states, then pressures.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(OUTPUT_DIR, f"fowm_{uuid.uuid4().hex}.csv")

    pressure_keys = list(outputs[0].keys())
    header = ["t", "x1", "x2", "x3", "x4", "x5", "x6"] + pressure_keys

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(t)):
            row = [t[i]] + list(sol[i]) + [outputs[i][k]
                                           for k in pressure_keys]
            writer.writerow(row)

    return os.path.abspath(file_path)
