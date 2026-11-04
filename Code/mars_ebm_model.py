from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------
# This file lives in <project_root>/Code/. Inputs live in
# <project_root>/Input/ and outputs go to <project_root>/Output/, so the
# defaults below are resolved relative to this file rather than to the
# current working directory -- the script then works the same whether
# you run it from Code/, from the project root, or from anywhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "Input" / "mars_input.dat"
DEFAULT_OUTPUT = PROJECT_ROOT / "Output" / "mars_ebm_output.csv"

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
NB = 18                                    # number of 10-deg latitude bands
SOL_SECONDS = 88775.244                    # length of one Martian solar day (s)
MARS_YEAR_SOLS = 668.6                     # sols per Martian year
MARS_YEAR_SECONDS = MARS_YEAR_SOLS * SOL_SECONDS

SCALAR_KEYS = [
    "xlum", "exc", "obl", "xlsper", "sma", "s0",
    "rplanet", "grav", "ps_mean", "ps_ref", "cpdry", "rdry", "gama",
    "difs_scale", "alfrost", "tsup_frost", "tinf_frost",
    "alc0", "alc1", "a1", "a2", "b1", "b2", "cc0", "dd0",
]


@dataclass
class MarsEBMParams:
    xlum: float; exc: float; obl: float; xlsper: float; sma: float; s0: float
    rplanet: float; grav: float; ps_mean: float; ps_ref: float
    cpdry: float; rdry: float; gama: float; difs_scale: float
    alfrost: float; tsup_frost: float; tinf_frost: float
    alc0: float; alc1: float
    a1: float; a2: float; b1: float; b2: float; cc0: float; dd0: float

    phi: np.ndarray
    ts_ini: np.ndarray
    albedo: np.ndarray
    fcloud: np.ndarray
    elev: np.ndarray
    difs: np.ndarray
    cheat: np.ndarray

    xlat: np.ndarray = field(default=None)
    surf: np.ndarray = field(default=None)     # fractional band area (sums to 1)
    phi2: np.ndarray = field(default=None)      # band boundary latitudes (deg)


# ---------------------------------------------------------------------
# Input file parsing (equivalent of the READ(20,*) block in CONST_EBM)
# ---------------------------------------------------------------------
def load_input(path) -> MarsEBMParams:
    scalars: dict[str, float] = {}
    table_rows: list[list[float]] = []

    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" in line:
                name, value = line.split(":", 1)
                scalars[name.strip().lower()] = float(value.strip())
            else:
                table_rows.append([float(x) for x in line.split()])

    if len(table_rows) != NB:
        raise ValueError(
            f"Expected {NB} latitude-band rows in {path}, found {len(table_rows)}"
        )
    missing = [k for k in SCALAR_KEYS if k not in scalars]
    if missing:
        raise ValueError(f"Missing input parameters in {path}: {missing}")

    table = np.array(table_rows, dtype=float)
    phi, ts_ini, albedo, fcloud, elev, difs, cheat = table.T

    order = np.argsort(phi)
    phi, ts_ini, albedo, fcloud, elev, difs, cheat = (
        arr[order] for arr in (phi, ts_ini, albedo, fcloud, elev, difs, cheat)
    )

    params = MarsEBMParams(
        **{k: scalars[k] for k in SCALAR_KEYS},
        phi=phi, ts_ini=ts_ini, albedo=albedo, fcloud=fcloud,
        elev=elev, difs=difs * scalars["difs_scale"], cheat=cheat,
    )
    _compute_geometry(params)
    return params


def _compute_geometry(p: MarsEBMParams) -> None:
    """Equal-area band geometry, ported from CONST_EBM."""
    p.xlat = np.sin(np.radians(p.phi))

    phi2 = np.empty(NB)
    phi2[:-1] = (p.phi[:-1] + p.phi[1:]) / 2.0
    phi2[-1] = 90.0
    p.phi2 = phi2

    xlat2 = np.sin(np.radians(phi2))
    xlat1 = np.concatenate(([-1.0], xlat2[:-1]))
    surf = (xlat2 - xlat1) / 2.0
    p.surf = surf / surf.sum()


# ---------------------------------------------------------------------
# Seasonal insolation (ported from SOLDAY)
# ---------------------------------------------------------------------
def daily_insolation(p: MarsEBMParams, dn: float):
    """Daily-mean top-of-atmosphere insolation per band for day-of-year dn
    (sols since Ls = 0, i.e. northern-hemisphere spring equinox)."""
    obl_r = math.radians(p.obl)
    xlsper_r = math.radians(p.xlsper)

    slg = (2.0 * math.pi / MARS_YEAR_SOLS) * dn
    if slg < 0.0:
        slg += 2.0 * math.pi
    th = slg - xlsper_r
    sth = math.sin(th)
    xls = (slg + 2.0 * sth * p.exc + 2.5 * sth * math.cos(th) * p.exc ** 2
           + (3.0 * sth - 13.0 * sth ** 3 / 3.0) * p.exc ** 3)
    theta = xls - xlsper_r
    rm = (1.0 - p.exc ** 2) / (1.0 + p.exc * math.cos(theta))
    sdel = math.sin(obl_r) * math.sin(xls)
    cdel = math.sqrt(max(1.0 - sdel ** 2, 0.0))
    cte = p.s0 / (rm * rm)                       # flux at true Mars-Sun distance

    sphi = p.xlat
    cphi = np.sqrt(np.maximum(1.0 - sphi ** 2, 1e-20))
    ch0 = np.clip(-sphi * sdel / (cphi * cdel), -1.0, 1.0)
    h0 = np.arccos(ch0)

    cz0 = (h0 * sphi * sdel + cphi * cdel * np.sin(h0)) / math.pi
    sold = cte * cz0
    ls = xls % (2.0 * math.pi)
    return sold, ls


# ---------------------------------------------------------------------
# Surface pressure / potential temperature (ported from givedrv)
# ---------------------------------------------------------------------
def surface_pressure_and_ptem(temp: np.ndarray, p: MarsEBMParams):
    puis_p = p.grav / (p.gama * p.rdry)
    puis_t = p.rdry / p.cpdry
    tem0 = temp + p.gama * p.elev
    ps = p.ps_mean * (temp / tem0) ** puis_p
    ptem = temp * (p.ps_ref / ps) ** puis_t
    return ps, ptem


# ---------------------------------------------------------------------
# Albedo / frost feedback (ported from ALBEDO_BUD)
# ---------------------------------------------------------------------
def compute_albedo(temp: np.ndarray, p: MarsEBMParams):
    fi = np.clip(
        (p.tsup_frost - temp) / (p.tsup_frost - p.tinf_frost), 0.0, 1.0
    )
    alb_surf = (1.0 - fi) * p.albedo + fi * p.alfrost
    alcr = p.alc1 + (1.0 - p.alc1) ** 2 * alb_surf         # clear-sky planetary albedo
    alcy = (p.alc1 / 2.0) + (1.0 - p.alc1 / 2.0) ** 2 * p.alc0  # cloudy-column albedo
    alb = p.fcloud * alcy + (1.0 - p.fcloud) * alcr
    return alb, alb_surf, fi


# ---------------------------------------------------------------------
# Outgoing longwave radiation (ported from the FIR line in givedrv)
# ---------------------------------------------------------------------
def compute_olr(temp: np.ndarray, p: MarsEBMParams):
    tcels = temp - 273.15
    lnp = math.log(p.ps_mean / p.ps_ref) if p.ps_mean > 0 else 0.0
    a0 = p.a1 + p.a2 * lnp
    b0 = p.b1 + p.b2 * lnp
    fir = a0 + b0 * tcels - (p.cc0 + p.dd0 * tcels) * p.fcloud
    return fir


# ---------------------------------------------------------------------
# Meridional heat-flux divergence (ported from DFLMER)
# ---------------------------------------------------------------------
def meridional_flux_divergence(ptem: np.ndarray, p: MarsEBMParams):
    hsen = (p.ps_mean * p.cpdry / p.grav) * ptem
    facgeo = 2.0 * math.pi * p.rplanet ** 2

    fmer = np.zeros(NB + 1)     # fmer[i] = flux across the boundary north of band i (1-indexed like Fortran)
    dfmer = np.zeros(NB)
    xl1 = -1.0
    for i in range(NB - 1):
        xl2 = math.sin(math.radians(p.phi2[i]))
        dxl2 = p.xlat[i + 1] - p.xlat[i]
        fac2 = 2.0 * math.pi * (1.0 - xl2 ** 2)
        fmer[i + 1] = fac2 * p.difs[i] * (hsen[i + 1] - hsen[i]) / dxl2
        dfmer[i] = (fmer[i + 1] - fmer[i]) / ((xl2 - xl1) * facgeo)
        xl1 = xl2
    dfmer[NB - 1] = -fmer[NB - 1] / ((1.0 - xl1) * facgeo)
    return dfmer


# ---------------------------------------------------------------------
# Right-hand side of dT/dt (ported from givedrv)
# ---------------------------------------------------------------------
def rhs(t_seconds: float, temp: np.ndarray, p: MarsEBMParams, diag: dict) -> np.ndarray:
    dn = (t_seconds / SOL_SECONDS) % MARS_YEAR_SOLS
    sold, ls = daily_insolation(p, dn)
    ps, ptem = surface_pressure_and_ptem(temp, p)
    alb, alb_surf, fi = compute_albedo(temp, p)
    fir = compute_olr(temp, p)
    dfmer = meridional_flux_divergence(ptem, p)

    absorbed = p.xlum * sold * (1.0 - alb)
    dTdt = (absorbed - fir + dfmer) / p.cheat

    diag.update(dn=dn, ls_deg=math.degrees(ls) % 360.0, sold=sold,
                alb=alb, alb_surf=alb_surf, fi=fi, fir=fir, dfmer=dfmer,
                absorbed=absorbed)
    return dTdt


# ---------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------
def run_simulation(input_path, n_years: float, steps_per_year: int = 96):
    """Integrate the model for n_years Martian years, sampling
    steps_per_year output points per year (~ every 7 sols by default)."""
    p = load_input(input_path)
    y0 = p.ts_ini.copy()
    diag: dict = {}

    def f(t, y):
        return rhs(t, y, p, diag)

    t_end = n_years * MARS_YEAR_SECONDS
    dt_out = MARS_YEAR_SECONDS / steps_per_year
    t_eval = np.arange(0.0, t_end + dt_out / 2.0, dt_out)

    # The ice/frost-albedo feedback is a clipped (kinked) function of
    # temperature, so tight tolerances make the adaptive step size
    # collapse right at the melt/freeze thresholds. 0.1 K absolute /
    # 1e-3 relative accuracy is far more than sufficient for a
    # seasonally-averaged EBM and keeps the integration fast.
    sol = solve_ivp(
        f, (0.0, t_end), y0, method="RK45",
        t_eval=t_eval, max_step=dt_out, rtol=1e-3, atol=1e-1,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    rows = []
    for k in range(sol.t.size):
        t, y = sol.t[k], sol.y[:, k]
        rhs(t, y, p, diag)          # recompute diagnostics at this (t, y)
        row = {
            "time_years": t / MARS_YEAR_SECONDS,
            "sol_of_year": diag["dn"],
            "Ls_deg": diag["ls_deg"],
            "T_global_mean": float(np.average(y, weights=p.surf)),
            "albedo_global_mean": float(np.average(diag["alb"], weights=p.surf)),
            "frost_area_frac": float(np.average(diag["fi"], weights=p.surf)),
        }
        for i, lat in enumerate(p.phi):
            tag = f"{lat:.0f}"
            row[f"T_{tag}"] = y[i]
            row[f"alb_{tag}"] = diag["alb"][i]
            row[f"frost_{tag}"] = diag["fi"][i]
        rows.append(row)

    return p, rows


def write_csv(rows: list[dict], out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Run the Mars latitudinal EBM.")
    ap.add_argument("--input", default=str(DEFAULT_INPUT), help="input parameter file")
    ap.add_argument("--years", type=float, default=5.0, help="number of Mars years to simulate")
    ap.add_argument("--steps-per-year", type=int, default=96, help="output samples per Mars year")
    ap.add_argument("--out", default=str(DEFAULT_OUTPUT), help="output CSV path")
    args = ap.parse_args()

    print(f"Loading parameters from {args.input} ...")
    p, rows = run_simulation(args.input, args.years, args.steps_per_year)

    print(f"Ran {args.years:g} Mars years ({len(rows)} output samples).")
    print(f"Final global-mean temperature: {rows[-1]['T_global_mean']:.2f} K")
    print(f"Final global-mean albedo:      {rows[-1]['albedo_global_mean']:.3f}")

    write_csv(rows, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
