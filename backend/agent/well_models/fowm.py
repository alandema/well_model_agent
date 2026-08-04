"""Fast Offshore Wells Model (FOWM).

A practical dynamic model for multiphase oil production systems in deepwater
and ultra-deepwater scenarios.

The model integrates a system of six ODEs describing the mass balance of gas
and liquid in three coupled volumes (pipeline/riser, tubing and annulus) and
exposes the resulting pressures (PDG, tubing top, riser top, riser bottom) at
every time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import odeint


# ---------------------------------------------------------------------------
# Physical constants shared by the model
# ---------------------------------------------------------------------------
_ROL = 900.0          # liquid density [kg/m3]
_R = 8314.0           # universal gas constant [J/(kmol*K)]
_T = 298.0            # temperature [K]
_M = 18.0             # gas molar mass [kg/kmol]
_G = 9.81             # gravity [m/s2]
_ALFAGW = 0.0188      # reservoir gas fraction
_ROMRES = 891.9523    # reservoir mixture density [kg/m3]

# Geometric / operational constants
_A = 0.018145839167135
_VR = 81.601838734604499
_VT = 28.963520770689396
_VA = 17.210272874895608
_HVGL = 916.0
_HPDG = 1117.0
_HT = 1279.0
_LA = 1118.0


# ---------------------------------------------------------------------------
# Default configuration (taken from the original notebook)
# ---------------------------------------------------------------------------
_DEFAULT_PARAM = np.multiply(
    1e2,
    [
        7.109820799914321,
        0.000000234607899,
        0.000058137935671,
        0.901595193514190,
        0.000358225582262,
        0.000010212053760,
        0.000001766624933,
        2.467164730000336,
    ],
)

_DEFAULT_U = [165000.0, 16.0, 1013250.0, 2.25e7]

_DEFAULT_X0 = np.multiply(
    1e4,
    [
        0.762949953300966,
        0.150646645264105,
        2.024926259090548,
        0.213535823438009,
        0.113058624767771,
        1.519684541979486,
    ],
)

_DEFAULT_T = np.linspace(0, 100000, 1001)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass
class FOWMConfig:
    """User-configurable inputs for a FOWM run.

    Every field has a default, so a run can be executed with no arguments at
    all (the notebook defaults). The user may override any subset.
    """

    # Model parameters: [mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr]
    param: List[float] = field(default_factory=lambda: list(_DEFAULT_PARAM))
    # Operational inputs: [gas_lift_rate, choke_opening, separator_pressure, reservoir_pressure]
    u: List[float] = field(default_factory=lambda: list(_DEFAULT_U))
    # Initial state: [x1..x6] (masses in the six control volumes)
    x0: List[float] = field(default_factory=lambda: list(_DEFAULT_X0))
    # Time grid
    t: List[float] = field(default_factory=lambda: list(_DEFAULT_T))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FOWM:
    """Fast Offshore Wells Model.

    Usage::

        result = FOWM.run()                       # defaults
        result = FOWM.run(FOWMConfig(u=[...]))     # custom inputs

    The returned dict contains the time grid, the six state trajectories, the
    four pressure trajectories and the final steady-state values.
    """

    # Human-readable description used by the well_model_picker tool to
    # decide when this model is a good fit for a user request.
    id = "fowm"
    description = (
        "Fast Offshore Wells Model (FOWM): a dynamic multiphase model for "
        "oil production in offshore (deepwater / ultra-deepwater) wells. "
        "It integrates a six-ODE mass-balance system across the pipeline, "
        "riser, tubing and annulus, and returns time series for six mass "
        "states and four pressures (PDG, tubing top, riser top, riser "
        "bottom). Use this model for offshore well production, gas-lift, "
        "choke/separator/reservoir pressure scenarios, and transient "
        "multiphase flow analysis in subsea wells."
    )

    # State variable names (masses)
    STATE_NAMES = [
        "m_gas_bubble",      # x1 - gas mass in the pipeline bubble
        "m_gas_riser",       # x2 - gas mass in the riser
        "m_liq_riser",       # x3 - liquid mass in the riser
        "m_gas_annulus",     # x4 - gas mass in the annulus
        "m_gas_tubing",      # x5 - gas mass in the tubing
        "m_liq_tubing",      # x6 - liquid mass in the tubing
    ]

    # Pressure names
    PRESSURE_NAMES = ["Ppdg", "Ptt", "Prt", "Prb"]

    # User-facing parameters required to run the model, exposed to the
    # agent via the get_model_parameters tool. Each entry has a name,
    # description, unit and default value.
    parameters = [
        {
            "name": "gas_lift_rate",
            "description": "Gas-lift rate",
            "unit": "Sm3/day",
            "default": 165000.0,
        },
        {
            "name": "choke_opening",
            "description": "Choke opening",
            "unit": "% (0-100)",
            "default": 16.0,
        },
        {
            "name": "separator_pressure",
            "description": "Separator pressure",
            "unit": "Pa",
            "default": 1013250.0,
        },
        {
            "name": "reservoir_pressure",
            "description": "Reservoir pressure",
            "unit": "Pa",
            "default": 2.25e7,
        },
        {
            "name": "t_end",
            "description": "End time of the simulation",
            "unit": "s",
            "default": 100000.0,
        },
        {
            "name": "n_points",
            "description": "Number of time-grid points",
            "unit": "-",
            "default": 1001,
        },
    ]

    # ------------------------------------------------------------------
    # Core ODE + pressure computation (single source of truth)
    # ------------------------------------------------------------------
    @classmethod
    def build_config(cls, parameters: Dict) -> "FOWMConfig":
        """Build a :class:`FOWMConfig` from a dict of user-facing parameters.

        Recognised keys (all optional, defaults applied when missing):
        ``gas_lift_rate``, ``choke_opening``, ``separator_pressure``,
        ``reservoir_pressure``, ``t_end``, ``n_points``.
        """
        cfg = FOWMConfig()

        u = list(cfg.u)
        if parameters.get("gas_lift_rate") is not None:
            u[0] = parameters["gas_lift_rate"]
        if parameters.get("choke_opening") is not None:
            u[1] = parameters["choke_opening"]
        if parameters.get("separator_pressure") is not None:
            u[2] = parameters["separator_pressure"]
        if parameters.get("reservoir_pressure") is not None:
            u[3] = parameters["reservoir_pressure"]
        cfg.u = u

        if parameters.get("t_end") is not None or parameters.get("n_points") is not None:
            t_end = parameters.get("t_end") or 100000.0
            n_points = parameters.get("n_points") or 1001
            cfg.t = list(np.linspace(0, t_end, n_points))

        return cfg

    @staticmethod
    def _derivatives_and_pressures(
        x: np.ndarray, u: np.ndarray, param: np.ndarray
    ) -> Tuple[List[float], Dict[str, float]]:
        """Return (dx/dt, pressures) for a given state.

        This merges the old ``FOWM`` and ``calcpres`` functions so the
        pressures are always consistent with the state used for integration.
        """
        x1, x2, x3, x4, x5, x6 = x
        mlstill, Cg, Cout, Veb, E, Kw, Ka, Kr = param
        u1, u2, u3, u4 = u

        Ps = u3
        Pr = u4

        Wgc = u1 * 101325.0 * _M / (293.0 * _R) / 3600.0 / 24.0
        z = u2 / 100.0

        # RISER + PIPELINE
        Peb = x1 * _R * _T / (_M * Veb)
        Prt = x2 * _R * _T / (_M * (_VR - (x3 + mlstill) / _ROL))
        Prb = Prt + (x3 + mlstill) * _G * 0.7071 / _A

        ALFAg = x2 / (x2 + x3)
        ALFAl = 1.0 - ALFAg

        Wout = Cout * z * np.sqrt(max(0.0, _ROL * (Prt - Ps)))
        Wlout = ALFAl * Wout
        Wgout = ALFAg * Wout
        Wg = Cg * max(0.0, (Peb - Prb))

        # TUBING
        Vgt = _VT - x6 / _ROL
        ROgt = x5 / Vgt
        ROmt = (x5 + x6) / _VT

        Ptt = ROgt * _R * _T / _M
        Ptb = Ptt + ROmt * _G * _HVGL
        Ppdg = Ptb + _ROMRES * _G * (_HPDG - _HVGL)
        Pbh = Ppdg + _ROMRES * _G * (_HT - _HPDG)

        ALFAgt = x5 / (x6 + x5)

        Wwh = Kw * np.sqrt(ROmt * max(0.0, (Ptt - Prb)))
        Wwhg = Wwh * ALFAgt
        Wwhl = Wwh * (1.0 - ALFAgt)
        Wr = max(0.0, Kr * (1.0 - 0.2 * Pbh / Pr - 0.8 * (Pbh / Pr) ** 2))

        # ANNULUS
        Pai = ((_R * _T / (_VA * _M)) + (_G * _LA / _VA)) * x4
        ROai = _M * Pai / (_R * _T)
        Wiv = Ka * np.sqrt(ROai * max(0.0, (Pai - Ptb)))

        # ODE
        dx = [
            (1.0 - E) * Wwhg - Wg,            # dx1 - gas mass in bubble
            E * Wwhg + Wg - Wgout,            # dx2 - gas mass in riser
            Wwhl - Wlout,                     # dx3 - liquid mass in riser
            Wgc - Wiv,                        # dx4 - gas mass in annulus
            Wr * _ALFAGW + Wiv - Wwhg,        # dx5 - gas mass in tubing
            Wr * (1.0 - _ALFAGW) - Wwhl,      # dx6 - liquid mass in tubing
        ]

        pressures = {"Ppdg": Ppdg, "Ptt": Ptt, "Prt": Prt, "Prb": Prb}
        return dx, pressures

    @staticmethod
    def _ode_func(x, _t, u, param):
        """Wrapper with the signature expected by ``scipy.integrate.odeint``."""
        dx, _ = FOWM._derivatives_and_pressures(np.asarray(x), u, param)
        return dx

    # ------------------------------------------------------------------
    # Public runner
    # ------------------------------------------------------------------
    @staticmethod
    def run(config: "FOWMConfig | None" = None) -> Dict:
        """Integrate the FOWM ODE system and return the full trajectory.

        Parameters
        ----------
        config:
            Optional :class:`FOWMConfig` providing ``param``, ``u``, ``x0``
            and ``t``. When ``None`` the notebook defaults are used.

        Returns
        -------
        dict
            ``t``                - time grid (1D array)
            ``states``           - dict of state-name -> trajectory
            ``pressures``        - dict of pressure-name -> trajectory
            ``final_states``     - dict of state-name -> final value
            ``final_pressures``  - dict of pressure-name -> final value
            ``config``           - the :class:`FOWMConfig` used
        """
        cfg = config or FOWMConfig()

        t = np.asarray(cfg.t, dtype=float)
        x0 = np.asarray(cfg.x0, dtype=float)
        u = np.asarray(cfg.u, dtype=float)
        param = np.asarray(cfg.param, dtype=float)

        sol = odeint(FOWM._ode_func, x0, t, args=(u, param))

        # Compute pressures along the whole trajectory
        pressures = {name: np.empty_like(t) for name in FOWM.PRESSURE_NAMES}
        for i, state in enumerate(sol):
            _, p = FOWM._derivatives_and_pressures(state, u, param)
            for name, value in p.items():
                pressures[name][i] = value

        states = {
            name: sol[:, j] for j, name in enumerate(FOWM.STATE_NAMES)
        }

        final_states = {
            name: float(sol[-1, j]) for j, name in enumerate(FOWM.STATE_NAMES)
        }
        final_pressures = {
            name: float(pressures[name][-1]) for name in FOWM.PRESSURE_NAMES
        }

        return {
            "t": t,
            "states": states,
            "pressures": pressures,
            "final_states": final_states,
            "final_pressures": final_pressures,
            "config": cfg,
        }

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    @staticmethod
    def to_csv(result: Dict) -> str:
        """Serialize a :meth:`run` result to a CSV string.

        The CSV contains one row per time step with the columns::

            t, m_gas_bubble, m_gas_riser, m_liq_riser, m_gas_annulus,
            m_gas_tubing, m_liq_tubing, Ppdg, Ptt, Prt, Prb
        """
        import csv
        import io

        t = result["t"]
        states = result["states"]
        pressures = result["pressures"]

        columns = ["t"] + FOWM.STATE_NAMES + FOWM.PRESSURE_NAMES
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(columns)

        for i in range(len(t)):
            row = [t[i]]
            row += [states[name][i] for name in FOWM.STATE_NAMES]
            row += [pressures[name][i] for name in FOWM.PRESSURE_NAMES]
            writer.writerow(row)

        return buf.getvalue()