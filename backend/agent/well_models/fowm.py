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

    # ------------------------------------------------------------------
    # Core ODE + pressure computation (single source of truth)
    # ------------------------------------------------------------------
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