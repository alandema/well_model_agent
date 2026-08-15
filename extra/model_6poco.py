"""
Minimal, self-contained Python transcription of the 6-well network simulation (Dados_6Pold.m).

Transcribed faithfully from MATLAB model files:
- Dados_6Pold.m
- Model_110fa_ode_mp.m
- Model_110fa_ode_mp_i.m
- Model_Header.m
- HEADER.m
- calcliq.m
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
import pandas as pd
# ==============================================================================
# Physical & Geometric Constants
# ==============================================================================
ROL = 900.0          # Liquid density [kg/m3]
R = 8314.0           # Universal gas constant [J/(kmol K)]
T = 298.0            # Temperature [K]
M = 18.0             # Molecular weight [kg/kmol]
G = 9.81             # Gravity [m/s2]
THETA = np.pi / 4.0  # Riser inclination angle [rad]
PR = 2.25e7          # Reservoir pressure [Pa]
ALFA_GW = 0.0188     # Gas-to-liquid mass ratio from reservoir
ROM_RES = 891.9523   # Reservoir mixture density [kg/m3]
MU = 1.43e-4         # Mixture dynamic viscosity [Pa s]
RUGO = 0.2e-5        # Pipe roughness [m]
MUL = 1.43e-4        # Liquid viscosity [Pa s]
MUG = 1.39e-5        # Gas viscosity [Pa s]

L = 4497.0           # Riser length [m]
LT = 1639.0          # Tubing length [m]
LA = 1118.0          # Annulus length [m]
D = 0.152            # Riser inner diameter [m]
DT = 0.150           # Tubing inner diameter [m]
DA = 0.140           # Annulus inner diameter [m]
A = D * D * np.pi / 4.0
VR = L * np.pi * D * D / 4.0
VT = LT * DT * DT * np.pi / 4.0
VA = LA * DA * DA * np.pi / 4.0
HVGL = 916.0         # Gas lift valve height [m]
HPDG = 1117.0        # PDG sensor height [m]
HT = 1279.0          # Total well depth [m]

# Header Constants
PS_HDR = 1013250.0   # Separator pressure [Pa]
KH = 3e-4            # Header valve constant
LH = 12.5            # Header length [m]
DH = 0.4572          # Header diameter [m]
VH = LH * DH * DH * np.pi / 4.0


# ==============================================================================
# Model Equations
# ==============================================================================
def _solve_x7(ptb, rogt, romt, wiv, alfagb, alfalmed, kr, fmf):
    """Solve algebraic friction drop in tubing (x7)."""
    def res(x7):
        ppdg = ptb + ROM_RES * G * (HPDG - HVGL) + x7
        pbh = ppdg + ROM_RES * G * (HT - HPDG)
        wr = max(0.0, kr * (1.0 - 0.2 * pbh / PR - 0.8 * (pbh / PR) ** 2))
        uslt = 4.0 * (1.0 - alfagb) * wr / (ROL * np.pi * DT**2)
        usgt = 4.0 * (wiv + alfagb * wr) / (rogt * np.pi * DT**2)
        umt = uslt + usgt
        ret = max(1e-6, romt * umt * DT / MU)
        ft = fmf * (-1.8 * np.log10((RUGO / (3.7 * DT) ** 1.11) + 6.9 / ret)) ** (-2)
        return x7 - alfalmed * ft * romt * (umt**2) * LT / (2.0 * DT)

    r0, r1 = res(0.0), res(1e7)
    return root_scalar(res, bracket=[0.0, 1e7], method="brentq").root if r0 * r1 <= 0 else 0.0


def _solve_x8(xi, prt, mlstill, romt, ptt, alfagt, kw):
    """Solve algebraic friction drop in riser (x8)."""
    def res(x8):
        prb = prt + (xi[2] + mlstill) * G * np.sin(THETA) / A + x8
        wwh = kw * np.sqrt(romt * max(0.0, ptt - prb))
        wwhg = wwh * alfagt
        wwhl = wwh * (1.0 - alfagt)
        uslr = wwhl / (ROL * A)
        rogr = max(1e-6, xi[1] / max(1e-6, VR - xi[2] / ROL))
        usgr = wwhg / (rogr * A)
        umr = uslr + usgr
        romr = max(1e-6, (xi[1] + xi[2]) / VR)
        alfalr = max(0.0, min(1.0, xi[2] / (VR * ROL)))
        mumr = alfalr * MUL + (1.0 - alfalr) * MUG
        rer = max(1e-6, romr * umr * D / mumr)
        fr = (-1.8 * np.log10((RUGO / (3.7 * D) ** 1.11) + 6.9 / rer)) ** (-2)
        return x8 - fr * romr * (umr**2) * L / (2.0 * D)

    r0, r1 = res(0.0), res(1e7)
    return root_scalar(res, bracket=[0.0, 1e7], method="brentq").root if r0 * r1 <= 0 else 0.0


def model_well(xi, ui, param):
    """Calculates derivatives and flows for a single well."""
    mlstill, cg, cout, veb, e, kw, ka, kr, fmf = param
    zg = ui[0] * 101325.0 * M / (293.0 * R) / 3600.0 / 24.0
    z = ui[1] * 0.01
    ps = ui[2]

    peb = xi[0] * R * T / (M * veb)
    prt = xi[1] * R * T / (M * (VR - (xi[2] + mlstill) / ROL))
    alfag = xi[1] / (xi[1] + xi[2])
    alfal = 1.0 - alfag
    wout = cout * z * np.sqrt(max(0.0, ROL * (prt - ps)))
    wlout = alfal * wout
    wgout = alfag * wout

    vgt = max(1e-6, VT - xi[5] / ROL)
    rogt = max(1e-6, xi[4] / vgt)
    romt = max(1e-6, (xi[4] + xi[5]) / VT)
    ptt = rogt * R * T / M
    ptb = ptt + romt * G * HVGL

    pai = ((R * T / (VA * M)) + (G * LA / VA)) * xi[3]
    roai = max(1e-6, M * pai / (R * T))
    wiv = ka * np.sqrt(roai * max(0.0, pai - ptb))

    alfagb = min(1.0, max(0.0, ALFA_GW / (ALFA_GW + 1.0)))
    alfalmed = max(0.0, min(1.0, max(0.0, xi[5]) / (VT * ROL)))
    alfagt = xi[4] / (xi[5] + xi[4])

    x7 = _solve_x7(ptb, rogt, romt, wiv, alfagb, alfalmed, kr, fmf)
    x8 = _solve_x8(xi, prt, mlstill, romt, ptt, alfagt, kw)

    ppdg = ptb + ROM_RES * G * (HPDG - HVGL) + x7
    pbh = ppdg + ROM_RES * G * (HT - HPDG)
    wr = max(0.0, kr * (1.0 - 0.2 * pbh / PR - 0.8 * (pbh / PR) ** 2))

    prb = prt + (xi[2] + mlstill) * G * np.sin(THETA) / A + x8
    wwh = kw * np.sqrt(romt * max(0.0, ptt - prb))
    wwhg = wwh * alfagt
    wwhl = wwh * (1.0 - alfagt)
    wg = cg * max(0.0, peb - prb)

    dxi = np.array([
        (1.0 - e) * wwhg - wg,           # dx1: gas mass in bubble
        e * wwhg + wg - wgout,           # dx2: gas mass in riser
        wwhl - wlout,                    # dx3: liquid mass in riser
        zg - wiv,                        # dx4: gas mass in annulus
        wr * ALFA_GW + wiv - wwhg,       # dx5: gas mass in tubing
        wr * (1.0 - ALFA_GW) - wwhl      # dx6: liquid mass in tubing
    ])
    return dxi, wlout, wgout, x7, x8


def model_header(xh, liq, gas):
    """Calculates derivatives for the manifold / header."""
    vhg = max(0.001, min(VH, VH - xh[1] / ROL))
    mgh = (xh[0] * vhg * M) / (R * T)
    roh = (mgh + xh[1]) / VH
    alphahgout = mgh / (mgh + xh[1])
    whout = KH * np.sqrt(roh * max(0.0, xh[0] - PS_HDR))
    dxh1 = (R * T * (gas - whout * alphahgout) / M - xh[0] / ROL * (liq - whout * (1.0 - alphahgout))) / vhg
    dxh2 = liq - whout * (1.0 - alphahgout)
    return np.array([dxh1, dxh2])


# ==============================================================================
# Simulation Runner
# ==============================================================================
def run_simulation(t_span=(0, 2e5), t_step=1000.0):
    """Runs the 6-well network simulation matching Dados_6Pold.m."""
    params = [
        [710.9820799914321, 2.3460789880222731e-05, 5.8137935670717683e-03, 90.15951935141904, 0.0358225582262254, 1.0212053760436238e-03, 1.7666249329670688e-04, 246.71647300003357, 10.0],
        [710.9820799914321, 2.3460789880222731e-05, 0.5137935670717683e-03, 90.15951935141904, 0.0358225582262254, 1.0212053760436238e-03, 1.7666249329670688e-04, 25.0, 10.0],
        [710.9820799914321, 2.3460789880222731e-05, 0.8137935670717683e-03, 90.15951935141904, 0.0358225582262254, 1.0212053760436238e-03, 1.7666249329670688e-04, 50.0, 10.0],
        [710.9820799914321, 2.3460789880222731e-05, 2.5137935670717683e-03, 90.15951935141904, 0.0358225582262254, 1.0212053760436238e-03, 1.7666249329670688e-04, 150.0, 10.0],
        [710.9820799914321, 2.3460789880222731e-05, 8.5137935670717683e-03, 90.15951935141904, 0.0358225582262254, 1.0212053760436238e-03, 1.7666249329670688e-04, 150.0, 10.0],
        [710.9820799914321, 2.3460789880222731e-05, 6.5137935670717683e-03, 90.15951935141904, 0.0358225582262254, 1.0212053760436238e-03, 1.7666249329670688e-04, 100.0, 10.0],
    ]
    u = [165000.0] * 6 + [10.0, 10.0, 10.0, 10.0, 10.0, 5.0]

    # Initial conditions from Dados_6Pold.m
    xx0 = 1e6 * np.array([
        0.007371830553633, 0.001428291136850, 0.016951042814041, 0.002021569964029, 0.001184201653727, 0.014054511806949, 1.223710827043528, 1.243924864994374,
        0.009912207160316, 0.005215537414924, 0.011123263153522, 0.002096473798538, 0.002528163432676, 0.005391837646180, 0.043189217068306, 0.050805887699325,
        0.009678410384208, 0.004195873764198, 0.014245455287811, 0.002126193070966, 0.002220003119176, 0.007537165893975, 0.095767127880593, 0.078125353716295,
        0.008561410454283, 0.002315476693005, 0.018313643808126, 0.002104166915800, 0.001523409475252, 0.012048992937728, 0.492022388447048, 0.457403591815644,
        0.007628969748398, 0.001690687298404, 0.017391630781485, 0.002010461760320, 0.001282923612724, 0.013197059384998, 0.875861084508333, 0.872376655157983,
        0.007799121005374, 0.001944069376985, 0.017259434462245, 0.001990461071670, 0.001382990468046, 0.012278190535320, 0.630809789570587, 0.624627108286633,
        1.307648174979114, 0.000138537805285
    ])

    # Extract 38 differential state initial values (6 per well + 2 header)
    y0_38 = np.empty(38, dtype=float)
    for i in range(6):
        y0_38[i * 6 : (i + 1) * 6] = xx0[i * 8 : i * 8 + 6]
    y0_38[36:38] = xx0[48:50]

    def ode_system(t, y38):
        dy = np.empty(38, dtype=float)
        liq, gas = 0.0, 0.0
        ps_hdr = y38[36]
        for i in range(6):
            xi = y38[i * 6 : (i + 1) * 6]
            ui = [u[i], u[i + 6], ps_hdr]
            dxi, wlout, wgout, _, _ = model_well(xi, ui, params[i])
            dy[i * 6 : (i + 1) * 6] = dxi
            liq += wlout
            gas += wgout
        dy[36:38] = model_header(y38[36:38], liq, gas)
        return dy

    t_eval = np.arange(t_span[0], t_span[1] + 1.0, t_step)
    sol = solve_ivp(ode_system, t_span, y0_38, t_eval=t_eval, method="Radau", rtol=1e-3, atol=1e-6)

    # Reconstruct all 50 states (including x7 and x8 for each well)
    num_steps = len(sol.t)
    y50 = np.empty((num_steps, 50), dtype=float)
    for k in range(num_steps):
        ps_hdr = sol.y[36, k]
        for i in range(6):
            xi = sol.y[i * 6 : (i + 1) * 6, k]
            ui = [u[i], u[i + 6], ps_hdr]
            _, _, _, x7, x8 = model_well(xi, ui, params[i])
            y50[k, i * 8 : i * 8 + 6] = xi
            y50[k, i * 8 + 6] = x7
            y50[k, i * 8 + 7] = x8
        y50[k, 48:50] = sol.y[36:38, k]

    return sol.t, y50


if __name__ == "__main__":
    t, y = run_simulation()
    df = pd.DataFrame(y, columns=[f"x{i}" for i in range(50)])
    df.to_csv("6poco/simulation_output.csv", index=False)
    print(f"Simulation completed successfully. Output shape: {y.shape}")
    print(f"Time span: t={t[0]:.0f} to t={t[-1]:.0f} ({len(t)} points)")