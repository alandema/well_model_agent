import csv
import os
import datetime

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import tool
import pint

from agent.services.config import load_config
from agent.services.model_runner import run_odeint

# Directory where model outputs are written as CSV files.
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", ".outputs")
MODELS_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "models_config.json"
)


def fowm(initial_conditions, t, params):
    x1, x2, x3, x4, x5, x6 = initial_conditions

    input_values = params["input_values"]
    separator_pressure = input_values["separator_pressure"]
    reservoir_pressure = input_values["reservoir_pressure"]
    gas_injection_rate = input_values["gas_injection_rate"]
    choke_opening = input_values["choke_opening"]

    simulation_params = params["simulation_params"]
    mlstill = simulation_params["mlstill"]
    Cg = simulation_params["Cg"]
    Cout = simulation_params["Cout"]
    Veb = simulation_params["Veb"]
    E = simulation_params["E"]
    Kw = simulation_params["Kw"]
    Ka = simulation_params["Ka"]
    Kr = simulation_params["Kr"]

    physical_constants = params["physical_constants"]
    Rol = physical_constants["Rol"]
    R = physical_constants["R"]
    T = physical_constants["T"]
    M = physical_constants["M"]
    g = physical_constants["g"]
    ALFAgw = physical_constants["ALFAgw"]
    theta = np.radians(physical_constants["theta"])
    Romres = physical_constants["Romres"]

    geometry_params = params["geometry_params"]
    D_ss = geometry_params["D_ss"]
    D_t = geometry_params["D_t"]
    D_a = geometry_params["D_a"]
    L_r = geometry_params["L_r"]
    L_fl = geometry_params["L_fl"]
    L_t = geometry_params["L_t"]
    L_a = geometry_params["L_a"]
    Hvgl = geometry_params["Hvgl"]
    Hpdg = geometry_params["Hpdg"]
    Ht = geometry_params["Ht"]

    # Geometry
    A = np.pi * (D_ss ** 2) / 4.0
    Vr = (np.pi * (D_ss ** 2) * L_r / 4.0) + (np.pi * (D_ss ** 2) * L_fl / 4.0)
    Vt = np.pi * (D_t ** 2) * L_t / 4.0
    Va = np.pi * (D_a ** 2) * L_a / 4.0

    # Control inputs
    Ps = separator_pressure
    Pr = reservoir_pressure
    Wgc = gas_injection_rate * 101325.0 * M / (293.0 * R) / 3600.0 / 24.0
    z = choke_opening / 100.0

    # RISER + PIPELINE
    Peb = x1 * R * T / (M * Veb)
    Prt = x2 * R * T / (M * (Vr - (x3 + mlstill) / Rol))
    Prb = Prt + (x3 + mlstill) * g * np.sin(theta) / A

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
    Pai = ((R * T / (Va * M)) + (g * L_a / Va)) * x4
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
        default=[7629.49953301, 1506.46645264, 20249.26259091,
                 2135.35823438, 1130.58624768, 15196.84541979],
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
        default=PhysicalValue(value=1013250.0, unit="Pa"),
        description="Downstream pressure at the separator or riser outlet.",
    )
    reservoir_pressure: PhysicalValue = Field(
        default=PhysicalValue(value=2.25e7, unit="Pa"),
        description="Reservoir pressure driving production inflow.",
    )


def _convert_physical_value(value, target_unit, ureg):
    """Convert a physical-value object to a numeric model value."""
    physical_value = PhysicalValue.model_validate(value)
    return (physical_value.value * ureg(physical_value.unit)).to(target_unit).magnitude


def _convert_configured_values(configured_values, target_units, ureg):
    """Convert configured physical values to the model's calculation units."""
    return {
        name: _convert_physical_value(value, target_units[name], ureg)
        for name, value in configured_values.items()
        if name in target_units
    }


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
    x0: list[float],
    gas_injection_rate: PhysicalValue,
    choke_opening: PhysicalValue,
    separator_pressure: PhysicalValue,
    reservoir_pressure: PhysicalValue,
) -> str:
    """Run the FOWM model and write the results to a CSV file.

    Returns the absolute path to the generated CSV file.
    """

    try:
        config = load_config(MODELS_CONFIG_PATH)

        # ureg = pint.UnitRegistry()

        params = {
            "input_values": {
                "separator_pressure": separator_pressure.value,
                "reservoir_pressure": reservoir_pressure.value,
                "gas_injection_rate": gas_injection_rate.value,
                "choke_opening": choke_opening.value,
            },
            "simulation_params": {k:v['value'] for k, v in config["fowm_model"]["simulation_params"].items()},
            "physical_constants": {k:v['value'] for k, v in config["fowm_model"]["physical_constants"].items()},
            "geometry_params": {k:v['value'] for k, v in config["fowm_model"]["geometry_params"].items()}
        }
        sol, outputs, t = run_odeint(
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
        OUTPUT_DIR, f"fowm_{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).strftime('%Y-%m-%d_%H-%M-%S')}.csv")

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
