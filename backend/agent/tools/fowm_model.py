import csv
import os
import datetime

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import tool
from langgraph.prebuilt import ToolRuntime
import pint

from agent.services.model_runner import run_model

# Directory where model outputs are written as CSV files.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", ".outputs")


def fowm(initial_conditions, t, params):
    x1, x2, x3, x4, x5, x6 = initial_conditions
    separator_pressure, reservoir_pressure, gas_injection_rate, choke_opening, mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr = params

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


class PhysicalValue(BaseModel):
    """A physical value with a unit."""
    value: float = Field(...,
                         description="The numerical value of the physical quantity.")
    unit: str = Field(...,
                      description="The unit of the physical quantity as per Pint python library (e.g., 'Pa', 'm³/day').")


class FowmModelInput(BaseModel):
    """Input schema for the FOWM model tool."""
    integration_time: float = Field(
        default=100000.0,
        gt=0,
        description="Simulation duration in seconds.",
    )
    time_points: int = Field(
        default=1001,
        ge=2,
        description="Number of time points to calculate, including the initial point.",
    )
    x0: list[float] = Field(
        default=[7629.4886447, 1506.47068049, 20249.37406149,
                 2135.35765486, 1130.58415814, 15196.85834102],
        description="Initial state variables [x1..x6].",
    )
    gas_injection_rate: PhysicalValue = Field(
        default=PhysicalValue(value=165000.0, unit="m^3/day"),
        description="Gas injected into the annulus for gas lift.",
    )
    choke_opening: PhysicalValue = Field(
        default=PhysicalValue(value=16.0, unit="percent"),
        description="Production choke opening.",
    )
    separator_pressure: PhysicalValue = Field(
        default=PhysicalValue(value=101325.0, unit="Pa"),
        description="Downstream pressure at the separator or riser outlet.",
    )
    reservoir_pressure: PhysicalValue = Field(
        default=PhysicalValue(value=2.25e7, unit="Pa"),
        description="Reservoir pressure driving production inflow.",
    )
    mlstill: PhysicalValue = Field(
        default=PhysicalValue(value=7.10982080e+02, unit="kg"),
        description="Retained liquid mass in the riser.",
    )
    Cg: PhysicalValue = Field(
        default=PhysicalValue(value=2.34607899e-05, unit="kg / (Pa * s)"),
        description="Expanded-bubble-to-riser gas transfer coefficient.",
    )
    Cout: PhysicalValue = Field(
        default=PhysicalValue(value=5.81379357e-03, unit="m^2"),
        description="Riser outlet flow coefficient.",
    )
    Veb: PhysicalValue = Field(
        default=PhysicalValue(value=9.01595194e+01, unit="m^3"),
        description="Expanded-bubble volume.",
    )
    E: PhysicalValue = Field(
        default=PhysicalValue(value=3.58225582e-02, unit="dimensionless"),
        description="Wellhead gas fraction bypassing the expanded bubble.",
    )
    Kw: PhysicalValue = Field(
        default=PhysicalValue(value=1.02120538e-03, unit="m^2"),
        description="Wellhead flow coefficient.",
    )
    Ka: PhysicalValue = Field(
        default=PhysicalValue(value=1.76662493e-04, unit="m^2"),
        description="Annulus-to-tubing gas flow coefficient.",
    )
    Kr: PhysicalValue = Field(
        default=PhysicalValue(value=2.46716473e+02, unit="kg / s"),
        description="Reservoir inflow coefficient.",
    )


@tool(
    args_schema=FowmModelInput,
    description=(
        """
Use this tool to simulate or analyze offshore deepwater and ultra-deepwater petroleum production systems using the Fast Offshore Wells Model (FOWM)[cite: 1].

WHEN TO USE:
* The query requires fast, real-time performance for monitoring, control, or optimization without numerical stiffness[cite: 1].
* The user needs a single, holistic model covering the entire system architecture (reservoir, tubing, gas lift annular, flowline, and riser)[cite: 1].
* The scenario involves analyzing severe slugging (limit cycles) generated simultaneously by casing heading (gas lift) and terrain/riser topography[cite: 1].
* The user needs estimations for unmeasured flow rates and pressures at key strategic points like PDG, TPT, and topside connections[cite: 1].

WHEN TO PREFER OTHER MODELS (e.g., OLGA or rigorous PDE simulators):
* The query strictly requires momentum and energy conservation balances, as FOWM relies solely on mass conservation ODEs[cite: 1].
* The user needs spatial variations of states within a control volume (PDEs) rather than a simplified lumped-parameter model[cite: 1].
* The analysis requires pressure drops calculated as a function of fluid velocities[cite: 1].
* The focus is on minor hydrodynamic slugs (wave formation due to phase slip) rather than severe slugging limit cycles[cite: 1].
"""
    ),
)
def fowm_model(
    integration_time: float,
    time_points: int,
    gas_injection_rate: PhysicalValue,
    choke_opening: PhysicalValue,
    separator_pressure: PhysicalValue,
    reservoir_pressure: PhysicalValue,
    mlstill: PhysicalValue,
    Cg: PhysicalValue,
    Cout: PhysicalValue,
    Veb: PhysicalValue,
    E: PhysicalValue,
    Kw: PhysicalValue,
    Ka: PhysicalValue,
    Kr: PhysicalValue,
    x0: list[float],
) -> str:
    """Run the FOWM model and write the results to a CSV file.

    Returns the absolute path to the generated CSV file.
    """

    try:
        ureg = pint.UnitRegistry()
        values = [separator_pressure, reservoir_pressure, gas_injection_rate,
                  choke_opening, mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr]
        target_units = ["Pa", "Pa", "m^3/day", "percent", "kg",
                        "kg / (Pa * s)", "m^2", "m^3", "dimensionless",
                        "m^2", "m^2", "kg / s"]
        params = [
            (value.value * ureg(value.unit)).to(unit).magnitude
            for value, unit in zip(values, target_units)
        ]
        sol, outputs, t = run_model(
            fowm,
            x0,
            integration_time,
            params,
            time_points,
        )
    except Exception as e:
        return f"Error running FOWM model: {str(e)}"

    # Build the CSV: one row per time point with t, states, then pressures.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(
        OUTPUT_DIR, f"fowm_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")

    pressure_keys = list(outputs[0].keys())
    header = ["t", "x1", "x2", "x3", "x4", "x5", "x6"] + pressure_keys

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(t)):
            row = [t[i]] + list(sol[i]) + [outputs[i][k]
                                           for k in pressure_keys]
            writer.writerow(row)

    return f"Results saved to {os.path.abspath(file_path)}. The CSV contains columns: {', '.join(header)}."
