"""Standalone FOWM (Fast Offshore Wells Model) for the control room.

Same ODE as the agent tool version, but without LangChain dependencies:
``run_fowm`` returns ``(t, pdg_pressure)`` for the "PDG Pressure" sensor.
"""

import json
import os

import numpy as np

from services.model_runner import run_odeint

WELL_MODELS_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "model_config.json"
)


def fowm(initial_conditions, t, params):
    x1, x2, x3, x4, x5, x6 = initial_conditions

    input_values = params["input_values"]
    separator_pressure = input_values["separator_pressure"]
    reservoir_pressure = input_values["reservoir_pressure"]
    gas_injection_rate = input_values["gas_injection_rate"]
    choke_opening = input_values["choke_opening"]

    well_params = params["well_params"]
    mlstill = well_params["mlstill"]
    Cg = well_params["Cg"]
    Cout = well_params["Cout"]
    Veb = well_params["Veb"]
    E = well_params["E"]
    Kw = well_params["Kw"]
    Ka = well_params["Ka"]
    Kr = well_params["Kr"]

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
    outputs = {
        "PDG_pressure": Ppdg,
        "top_riser_pressure": Prt,
        "flowline_pressure": Prb,
        "oil_flow_rate": Wlout,
    }
    return dx, outputs


def load_params():
    """Build the FOWM params dict from model_config.json (x0 excluded)."""
    with open(WELL_MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    model = config["fowm_model"]
    return {
        "input_values": {
            k: v["value"]
            for k, v in model["input_params"].items()
            if k != "x0"
        },
        "well_params": {k: v["value"] for k, v in model["well_params"].items()},
        "physical_constants": {
            k: v["value"] for k, v in model["physical_constants"].items()
        },
        "geometry_params": {
            k: v["value"] for k, v in model["geometry_params"].items()
        },
    }


def load_initial_state():
    """Initial state x0 from model_config.json; only used at service start."""
    with open(WELL_MODELS_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config["fowm_model"]["input_params"]["x0"]["values"]


def run_fowm(x0, integration_time, time_points, params):
    """Run the FOWM model and return ``(t, pdg_pressure, states)``."""
    sol, outputs, t = run_odeint(
        fowm,
        list(x0),
        integration_time,
        params,
        time_points,
    )
    pdg_pressure = np.array([o["PDG_pressure"] for o in outputs])
    return t, pdg_pressure, sol


def run_fowm_step(x, step_seconds, params):
    """Simulate a single next state, ``step_seconds`` after state ``x``.

    Maps the writer's ``model_simulated_time`` onto the simulation grid:
    ``integration_time = step_seconds`` and ``time_points = 2`` (current and
    next state). Returns ``(next_state, pdg_pressure)``; feed ``next_state``
    back into the next call to keep the simulation continuous.
    """
    sol, outputs, _ = run_odeint(fowm, list(x), step_seconds, params, 2)
    return sol[-1], outputs[-1]
