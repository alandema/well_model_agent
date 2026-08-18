import csv
import os
import datetime

import numpy as np
from pydantic import BaseModel, Field
from langchain.tools import tool
import pint

from agent.services.model_runner import run_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", ".outputs")


class PhysicalValue(BaseModel):
    """A physical value with a unit."""

    value: float = Field(
        ..., description="The numerical value of the physical quantity."
    )
    unit: str = Field(
        ...,
        description="The unit of the physical quantity as per Pint python library (e.g., 'Pa', 'm³/day').",
    )


class MultiWellParameter(BaseModel):
    """Parameters for one well in the network."""

    mlstill: PhysicalValue = Field(
        default=PhysicalValue(value=7.10982080e02, unit="kg"),
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
        default=PhysicalValue(value=9.01595194e01, unit="m^3"),
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
        default=PhysicalValue(value=2.46716473e02, unit="kg / s"),
        description="Reservoir inflow coefficient.",
    )
    friction_factor_multiplier: PhysicalValue = Field(
        default=PhysicalValue(value=10.0, unit="dimensionless"),
        description="Multiplicative friction factor correction for the well.",
    )


def default_well_parameters():
    return [
        MultiWellParameter(
            mlstill=PhysicalValue(value=710.9820799914321, unit="kg"),
            Cg=PhysicalValue(value=2.3460789880222731e-05,
                             unit="kg / (Pa * s)"),
            Cout=PhysicalValue(value=5.8137935670717683e-03, unit="m^2"),
            Veb=PhysicalValue(value=90.15951935141904, unit="m^3"),
            E=PhysicalValue(value=0.0358225582262254, unit="dimensionless"),
            Kw=PhysicalValue(value=1.0212053760436238e-03, unit="m^2"),
            Ka=PhysicalValue(value=1.7666249329670688e-04, unit="m^2"),
            Kr=PhysicalValue(value=246.71647300003357, unit="kg / s"),
            friction_factor_multiplier=PhysicalValue(
                value=10.0, unit="dimensionless"),
        ),
        MultiWellParameter(
            mlstill=PhysicalValue(value=710.9820799914321, unit="kg"),
            Cg=PhysicalValue(value=2.3460789880222731e-05,
                             unit="kg / (Pa * s)"),
            Cout=PhysicalValue(value=0.5137935670717683e-03, unit="m^2"),
            Veb=PhysicalValue(value=90.15951935141904, unit="m^3"),
            E=PhysicalValue(value=0.0358225582262254, unit="dimensionless"),
            Kw=PhysicalValue(value=1.0212053760436238e-03, unit="m^2"),
            Ka=PhysicalValue(value=1.7666249329670688e-04, unit="m^2"),
            Kr=PhysicalValue(value=25.0, unit="kg / s"),
            friction_factor_multiplier=PhysicalValue(
                value=10.0, unit="dimensionless"),
        ),
        MultiWellParameter(
            mlstill=PhysicalValue(value=710.9820799914321, unit="kg"),
            Cg=PhysicalValue(value=2.3460789880222731e-05,
                             unit="kg / (Pa * s)"),
            Cout=PhysicalValue(value=0.8137935670717683e-03, unit="m^2"),
            Veb=PhysicalValue(value=90.15951935141904, unit="m^3"),
            E=PhysicalValue(value=0.0358225582262254, unit="dimensionless"),
            Kw=PhysicalValue(value=1.0212053760436238e-03, unit="m^2"),
            Ka=PhysicalValue(value=1.7666249329670688e-04, unit="m^2"),
            Kr=PhysicalValue(value=50.0, unit="kg / s"),
            friction_factor_multiplier=PhysicalValue(
                value=10.0, unit="dimensionless"),
        ),
        MultiWellParameter(
            mlstill=PhysicalValue(value=710.9820799914321, unit="kg"),
            Cg=PhysicalValue(value=2.3460789880222731e-05,
                             unit="kg / (Pa * s)"),
            Cout=PhysicalValue(value=2.5137935670717683e-03, unit="m^2"),
            Veb=PhysicalValue(value=90.15951935141904, unit="m^3"),
            E=PhysicalValue(value=0.0358225582262254, unit="dimensionless"),
            Kw=PhysicalValue(value=1.0212053760436238e-03, unit="m^2"),
            Ka=PhysicalValue(value=1.7666249329670688e-04, unit="m^2"),
            Kr=PhysicalValue(value=150.0, unit="kg / s"),
            friction_factor_multiplier=PhysicalValue(
                value=10.0, unit="dimensionless"),
        ),
        MultiWellParameter(
            mlstill=PhysicalValue(value=710.9820799914321, unit="kg"),
            Cg=PhysicalValue(value=2.3460789880222731e-05,
                             unit="kg / (Pa * s)"),
            Cout=PhysicalValue(value=8.5137935670717683e-03, unit="m^2"),
            Veb=PhysicalValue(value=90.15951935141904, unit="m^3"),
            E=PhysicalValue(value=0.0358225582262254, unit="dimensionless"),
            Kw=PhysicalValue(value=1.0212053760436238e-03, unit="m^2"),
            Ka=PhysicalValue(value=1.7666249329670688e-04, unit="m^2"),
            Kr=PhysicalValue(value=150.0, unit="kg / s"),
            friction_factor_multiplier=PhysicalValue(
                value=10.0, unit="dimensionless"),
        ),
        MultiWellParameter(
            mlstill=PhysicalValue(value=710.9820799914321, unit="kg"),
            Cg=PhysicalValue(value=2.3460789880222731e-05,
                             unit="kg / (Pa * s)"),
            Cout=PhysicalValue(value=6.5137935670717683e-03, unit="m^2"),
            Veb=PhysicalValue(value=90.15951935141904, unit="m^3"),
            E=PhysicalValue(value=0.0358225582262254, unit="dimensionless"),
            Kw=PhysicalValue(value=1.0212053760436238e-03, unit="m^2"),
            Ka=PhysicalValue(value=1.7666249329670688e-04, unit="m^2"),
            Kr=PhysicalValue(value=100.0, unit="kg / s"),
            friction_factor_multiplier=PhysicalValue(
                value=10.0, unit="dimensionless"),
        ),
    ]


def default_x0(well_count: int):
    base_state = [
        0.007371830553633,
        0.001428291136850,
        0.016951042814041,
        0.002021569964029,
        0.001184201653727,
        0.014054511806949,
    ]
    vector = []
    for _ in range(well_count):
        vector.extend(base_state)
    vector.extend([1.0e6, 97.959709025643093])
    return vector


class MultiWellModelInput(BaseModel):
    """Input schema for the multi-well network model tool."""

    integration_time: float = Field(
        default=200000.0,
        gt=0,
        description="Simulation duration in seconds.",
    )
    time_points: int = Field(
        default=1001,
        ge=2,
        description="Number of time points to calculate, including the initial point.",
    )
    well_count: int = Field(
        default=6,
        ge=1,
        description="Number of wells to include in the coupled network model.",
    )
    gas_injection_rates: list[PhysicalValue] = Field(
        default_factory=lambda: [
            PhysicalValue(value=165000.0, unit="m^3/day")
        ] * 6,
        description="Gas injection rate for each well in the network.",
    )
    choke_openings: list[PhysicalValue] = Field(
        default_factory=lambda: [PhysicalValue(
            value=10.0, unit="percent")] * 6,
        description="Choke opening for each well in the network.",
    )
    separator_pressure: PhysicalValue = Field(
        default=PhysicalValue(value=1013250.0, unit="Pa"),
        description="Separator pressure at the network outlet.",
    )
    well_parameters: list[MultiWellParameter] = Field(
        default_factory=default_well_parameters,
        description="Parameters for each well; the list length should match well_count.",
    )
    x0: list[float] = Field(
        default_factory=lambda: default_x0(6),
        description="Initial state vector for the coupled model, with 8 states per well plus 2 header states.",
    )


def _normalize_well_values(values, well_count, default_value):
    normalized = list(values)
    if len(normalized) < well_count:
        normalized.extend([default_value] * (well_count - len(normalized)))
    elif len(normalized) > well_count:
        normalized = normalized[:well_count]
    return normalized


def _to_numeric(value: PhysicalValue, unit: str):
    ureg = pint.UnitRegistry()
    return (value.value * ureg(value.unit)).to(unit).magnitude


def _normalize_x0(x0, well_count):
    state_vector = np.asarray(x0, dtype=float)
    required_size = well_count * 6 + 2
    if state_vector.size == required_size:
        return state_vector
    if state_vector.size == 0:
        return np.asarray(default_x0(well_count), dtype=float)
    if state_vector.size < required_size:
        pad_value = state_vector[-1] if state_vector.size else 0.0
        pad = np.full(required_size - state_vector.size,
                      pad_value, dtype=float)
        return np.concatenate([state_vector, pad])
    return state_vector[:required_size]


def _solve_x7(ptb, rogt, romt, wiv, alfagb, alfalmed, kr, fmf):
    """Solve the algebraic tubing pressure loss term."""

    def res(x7):
        ppdg = ptb + 891.9523 * 9.81 * (1117.0 - 916.0) + x7
        pbh = ppdg + 891.9523 * 9.81 * (1279.0 - 1117.0)
        wr = max(0.0, kr * (1.0 - 0.2 * pbh / 2.25e7 - 0.8 * (pbh / 2.25e7) ** 2))
        uslt = 4.0 * (1.0 - alfagb) * wr / (900.0 * np.pi * 0.150**2)
        usgt = 4.0 * (wiv + alfagb * wr) / (rogt * np.pi * 0.150**2)
        umt = uslt + usgt
        ret = max(1e-6, romt * umt * 0.150 / 1.43e-4)
        ft = fmf * \
            (-1.8 * np.log10((0.2e-5 / (3.7 * 0.150) ** 1.11) + 6.9 / ret)) ** (-2)
        return x7 - alfalmed * ft * romt * (umt**2) * 1639.0 / (2.0 * 0.150)

    r0, r1 = res(0.0), res(1e7)
    if r0 * r1 <= 0:
        from scipy.optimize import root_scalar

        return root_scalar(res, bracket=[0.0, 1e7], method="brentq").root
    return 0.0


def _solve_x8(xi, prt, mlstill, romt, ptt, alfagt, kw):
    """Solve the algebraic riser pressure loss term."""

    def res(x8):
        prb = prt + (xi[2] + mlstill) * 9.81 * \
            np.sin(np.pi / 4.0) / (0.152**2 * np.pi / 4.0) + x8
        wwh = kw * np.sqrt(romt * max(0.0, ptt - prb))
        wwhg = wwh * alfagt
        wwhl = wwh * (1.0 - alfagt)
        uslr = wwhl / (900.0 * (0.152**2 * np.pi / 4.0))
        rogr = max(1e-6, xi[1] / max(1e-6, 4497.0 *
                   np.pi * 0.152**2 / 4.0 - xi[2] / 900.0))
        usgr = wwhg / (rogr * (0.152**2 * np.pi / 4.0))
        umr = uslr + usgr
        romr = max(1e-6, (xi[1] + xi[2]) / (4497.0 * np.pi * 0.152**2 / 4.0))
        alfalr = max(
            0.0, min(1.0, xi[2] / (4497.0 * np.pi * 0.152**2 / 4.0 * 900.0)))
        mumr = alfalr * 1.43e-4 + (1.0 - alfalr) * 1.39e-5
        rer = max(1e-6, romr * umr * 0.152 / mumr)
        fr = (-1.8 * np.log10((0.2e-5 / (3.7 * 0.152) ** 1.11) + 6.9 / rer)) ** (-2)
        return x8 - fr * romr * (umr**2) * 4497.0 / (2.0 * 0.152)

    r0, r1 = res(0.0), res(1e7)
    if r0 * r1 <= 0:
        from scipy.optimize import root_scalar

        return root_scalar(res, bracket=[0.0, 1e7], method="brentq").root
    return 0.0


def model_well(xi, ui, param):
    """Calculates derivatives and flows for a single well."""

    mlstill, cg, cout, veb, e, kw, ka, kr, fmf = param
    zg = ui[0] * 101325.0 * 18.0 / (293.0 * 8314.0) / 3600.0 / 24.0
    z = ui[1] * 0.01
    ps = ui[2]

    peb = xi[0] * 8314.0 * 298.0 / (18.0 * veb)
    prt = xi[1] * 8314.0 * 298.0 / \
        (18.0 * (4497.0 * np.pi * 0.152**2 / 4.0 - (xi[2] + mlstill) / 900.0))
    alfag = xi[1] / (xi[1] + xi[2])
    alfal = 1.0 - alfag
    wout = cout * z * np.sqrt(max(0.0, 900.0 * (prt - ps)))
    wlout = alfal * wout
    wgout = alfag * wout

    vgt = max(1e-6, 1639.0 * 0.150**2 * np.pi / 4.0 - xi[5] / 900.0)
    rogt = max(1e-6, xi[4] / vgt)
    romt = max(1e-6, (xi[4] + xi[5]) / (1639.0 * 0.150**2 * np.pi / 4.0))
    ptt = rogt * 8314.0 * 298.0 / 18.0
    ptb = ptt + romt * 9.81 * 916.0

    pai = ((8314.0 * 298.0 / (1118.0 * 18.0 * 0.140**2 * np.pi / 4.0)) +
           (9.81 * 1118.0 / (0.140**2 * np.pi / 4.0))) * xi[3]
    roai = max(1e-6, 18.0 * pai / (8314.0 * 298.0))
    wiv = ka * np.sqrt(roai * max(0.0, pai - ptb))

    alfagb = min(1.0, max(0.0, 0.0188 / (0.0188 + 1.0)))
    alfalmed = max(
        0.0, min(1.0, max(0.0, xi[5]) / ((1639.0 * 0.150**2 * np.pi / 4.0) * 900.0)))
    alfagt = xi[4] / (xi[5] + xi[4])

    x7 = _solve_x7(ptb, rogt, romt, wiv, alfagb, alfalmed, kr, fmf)
    x8 = _solve_x8(xi, prt, mlstill, romt, ptt, alfagt, kw)

    ppdg = ptb + 891.9523 * 9.81 * (1117.0 - 916.0) + x7
    pbh = ppdg + 891.9523 * 9.81 * (1279.0 - 1117.0)
    wr = max(0.0, kr * (1.0 - 0.2 * pbh / 2.25e7 - 0.8 * (pbh / 2.25e7) ** 2))

    prb = prt + (xi[2] + mlstill) * 9.81 * \
        np.sin(np.pi / 4.0) / (0.152**2 * np.pi / 4.0) + x8
    wwh = kw * np.sqrt(romt * max(0.0, ptt - prb))
    wwhg = wwh * alfagt
    wwhl = wwh * (1.0 - alfagt)
    wg = cg * max(0.0, peb - prb)

    dxi = np.array([
        (1.0 - e) * wwhg - wg,
        e * wwhg + wg - wgout,
        wwhl - wlout,
        zg - wiv,
        wr * 0.0188 + wiv - wwhg,
        wr * (1.0 - 0.0188) - wwhl,
    ])
    return dxi, wlout, wgout, x7, x8


def model_header(xh, liq, gas, separator_pressure):
    """Calculates derivatives for the manifold / header."""

    vh = 12.5 * 0.4572**2 * np.pi / 4.0
    vhg = max(0.001, min(vh, vh - xh[1] / 900.0))
    mgh = (xh[0] * vhg * 18.0) / (8314.0 * 298.0)
    roh = (mgh + xh[1]) / vh
    alphahgout = mgh / (mgh + xh[1])
    whout = 3e-4 * np.sqrt(roh * max(0.0, xh[0] - separator_pressure))
    dxh1 = (
        8314.0 * 298.0 * (gas - whout * alphahgout) / 18.0
        - xh[0] / 900.0 * (liq - whout * (1.0 - alphahgout))
    ) / vhg
    dxh2 = liq - whout * (1.0 - alphahgout)
    return np.array([dxh1, dxh2])


def multi_well(initial_conditions, t, params):
    """Model for a network of wells connected to a common header."""

    well_count = params["well_count"]
    gas_rates = params["gas_rates"]
    choke_openings = params["choke_openings"]
    separator_pressure = params["separator_pressure"]
    well_params = params["well_params"]

    state = np.asarray(initial_conditions, dtype=float)
    header_state = state[-2:]
    dx = np.empty(well_count * 6 + 2, dtype=float)
    liq = 0.0
    gas = 0.0

    for i in range(well_count):
        xi = state[i * 6: (i + 1) * 6]
        ui = [gas_rates[i], choke_openings[i], header_state[0]]
        dxi, wlout, wgout, _, _ = model_well(xi, ui, well_params[i])
        dx[i * 6: (i + 1) * 6] = dxi
        liq += wlout
        gas += wgout

    dx[well_count * 6:] = model_header(header_state,
                                       liq, gas, separator_pressure)
    outputs = {
        "liquid_flow": float(liq),
        "gas_flow": float(gas),
        "header_pressure": float(header_state[0]),
        "header_liquid_mass": float(header_state[1]),
    }
    return dx, outputs


@tool(
    args_schema=MultiWellModelInput,
    description=(
        """
Run a coupled multi-well model for a network of production wells connected to a common header.

Use this tool when the user asks to simulate a production network with several wells instead of a single well.
The number of wells is explicitly controlled by the well_count field, and the model returns a CSV summary of well states and header variables.
"""
    ),
)
def multi_well_model(
    integration_time: float,
    time_points: int,
    well_count: int,
    gas_injection_rates: list[PhysicalValue],
    choke_openings: list[PhysicalValue],
    separator_pressure: PhysicalValue,
    well_parameters: list[MultiWellParameter],
    x0: list[float],
) -> str:
    """Run the multi-well model and write the results to a CSV file."""

    try:
        gas_rates = [
            _to_numeric(value, "m^3/day")
            for value in _normalize_well_values(
                gas_injection_rates, well_count, PhysicalValue(
                    value=165000.0, unit="m^3/day")
            )
        ]
        choke_values = [
            _to_numeric(value, "percent")
            for value in _normalize_well_values(
                choke_openings, well_count, PhysicalValue(
                    value=10.0, unit="percent")
            )
        ]
        normalized_parameters = []
        for i in range(well_count):
            if i < len(well_parameters):
                normalized_parameters.append(well_parameters[i])
            else:
                normalized_parameters.append(default_well_parameters()[-1])

        default_separator = (separator_pressure.value * pint.UnitRegistry()
                             (separator_pressure.unit)).to("Pa").magnitude
        params = {
            "well_count": well_count,
            "gas_rates": gas_rates,
            "choke_openings": choke_values,
            "separator_pressure": default_separator,
            "well_params": [
                [
                    _to_numeric(param.mlstill, "kg"),
                    _to_numeric(param.Cg, "kg / (Pa * s)"),
                    _to_numeric(param.Cout, "m^2"),
                    _to_numeric(param.Veb, "m^3"),
                    _to_numeric(param.E, "dimensionless"),
                    _to_numeric(param.Kw, "m^2"),
                    _to_numeric(param.Ka, "m^2"),
                    _to_numeric(param.Kr, "kg / s"),
                    _to_numeric(param.friction_factor_multiplier,
                                "dimensionless"),
                ]
                for param in normalized_parameters
            ],
        }
        state_vector = _normalize_x0(x0, well_count)
        sol, outputs, t = run_model(
            multi_well, state_vector, integration_time, params, time_points)
    except Exception as e:
        return f"Error running multi-well model: {str(e)}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_path = os.path.join(
        OUTPUT_DIR,
        f"multi_well_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv",
    )

    state_names = [
        f"well_{i + 1}_x{j}"
        for i in range(well_count)
        for j in range(1, 7)
    ] + ["header_x1", "header_x2"]
    output_keys = list(outputs[0].keys())
    header = ["t"] + state_names + output_keys

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(t)):
            row = [t[i]] + list(sol[i]) + [outputs[i][k] for k in output_keys]
            writer.writerow(row)

    return f"Results saved to {os.path.abspath(file_path)}. The CSV contains columns: {', '.join(header)}."
