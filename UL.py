# UL.py — Universal Life monthly projection + sales tables (Level) + ALT investment report (with Rebalancing toggle)
# Files required next to UL.py:
#   - Tables/COI.csv       (columns: PolicyYear, and e.g. YRT70-F30, YRT70-F30S, YRT85-M45, LEVEL-F30, ...)
#   - Tables/Tax_Rates.csv (must include Province column and Personal_Income for marginal rate; optional CG_Inclusion)
#
# I/O:
# - Prompts: Province → Ownership → Gender → Age → Smoker → Face → COI Type → Premium Years → Base Rate → Safe Rate
# - Prints one table (Level DB) at four rates: Base−1%, Base, Base+1%, Base+2%.
# - Writes CSVs from the LEVEL base-case monthly sim:
#     Reports/LEVEL_Data.csv
#     Reports/LVL_Report.csv
# - ALT_Report.csv     : Alternative investment scenario (same premium & years as LEVEL Target) with:
#     Reports/ALT_Report.csv
#     Reports/ALT_Rebalance.csv (if rebalancing ON; detailed ledger)
#
# Notes:
# - IRR per year is computed as of EOY of each policy year using negative deposits paid up to that year
#   and a terminal inflow equal to the Level Death Benefit (Face Value) or Alt Net After-Tax value (for ALT).
# - TE IRR = IRR / (1 - Personal_Income tax rate for the selected province).
# - Age shown is END-OF-YEAR age (industry standard): Age = IssueAge + PolicyYear.

from __future__ import annotations

import os
import sys
import time
import threading
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ---------- Console helpers ----------

def supports_ansi() -> bool:
    return sys.stdout.isatty()

def status(msg: str) -> None:
    print(msg if not supports_ansi() else msg, end="" if supports_ansi() else "\n", flush=True)

def clear_status() -> None:
    if supports_ansi():
        print("\r\033[K", end="", flush=True)

def abbrev_money(x: float) -> str:
    n = float(x)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n >= 1_000_000_000:
        return f"{sign}${n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{sign}${n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{sign}${n/1_000:.0f}K"
    return f"{sign}${n:,.0f}"

def tax_equiv(irr_after_tax: float, tax_rate: float) -> float:
    denom = (1.0 - tax_rate)
    return irr_after_tax / denom if denom > 0 else float("nan")

def read_input(prompt: str, default: Optional[str] = None) -> str:
    """Always try to read from input; only fall back to default if EOF."""
    try:
        return input(prompt)
    except EOFError:
        return str(default) if default is not None else ""


def paidup_age(start_age: int, years: float) -> int:
    return int(round(start_age + years))  # EOY convention

def coi_limit_age(kind: str) -> int:
    return 70 if kind == "YRT70" else 85 if kind == "YRT85" else 100

def last_coi_months(start_age: int, kind: str) -> int:
    return max(0, coi_limit_age(kind) - start_age) * 12

def _no_lapse(df_monthly: pd.DataFrame, horizon_months: int) -> bool:
    if horizon_months <= 0:
        return True
    return float(df_monthly["FV_EOY"].iloc[:horizon_months].min()) >= 0.0

def ask_int_default(prompt: str, default: int) -> int:
    while True:
        s = read_input(prompt).strip()
        if s == "":
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("Please enter a whole number.")

def ask_float_default(prompt: str, default: float) -> float:
    while True:
        s = read_input(prompt).strip()
        if s == "":
            return float(default)
        try:
            return float(s)
        except ValueError:
            print("Please enter a number.")

# ---------- Robust IRR ----------

def irr(cashflows: List[float], lo: float = -0.9999, hi: float = 4.0, tol: float = 1e-7, max_iter: int = 200) -> float:
    has_pos = any(cf > 0 for cf in cashflows)
    has_neg = any(cf < 0 for cf in cashflows)
    if not (has_pos and has_neg):
        return float("nan")

    def npv(rate: float) -> float:
        if rate <= -0.999999:
            return float("inf")
        acc = cashflows[0]
        df = 1.0
        one_plus = 1.0 + rate
        for cf in cashflows[1:]:
            df *= one_plus
            acc += cf / df
        return acc

    f_lo, f_hi = npv(lo), npv(hi)
    for _ in range(12):
        if f_lo * f_hi <= 0:
            break
        hi *= 1.5
        f_hi = npv(hi)
    if f_lo * f_hi > 0:
        for _ in range(6):
            lo = min(lo + 0.1, -0.01)
            f_lo = npv(lo)
            if f_lo * f_hi <= 0:
                break
    if f_lo * f_hi > 0:
        return float("nan")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = npv(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)

# ---------- Three-dot animation ----------

_dot_stop_event: Optional[threading.Event] = None
_dot_thread: Optional[threading.Thread] = None

def _start_dots(text: str = "Calculating") -> None:
    if not supports_ansi():
        print(text, flush=True)
        return
    global _dot_stop_event, _dot_thread
    if _dot_thread and _dot_thread.is_alive():
        return
    _dot_stop_event = threading.Event()
    def run():
        i = 0
        dots = ["", ".", "..", "..."]
        while _dot_stop_event and not _dot_stop_event.is_set():
            sys.stdout.write(f"\r{text}{dots[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.4)
            i += 1
    _dot_thread = threading.Thread(target=run, daemon=True)
    _dot_thread.start()

def _stop_dots() -> None:
    global _dot_stop_event, _dot_thread
    if _dot_stop_event:
        _dot_stop_event.set()
    if _dot_thread:
        _dot_thread.join()
    if supports_ansi():
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

# ---------- Tax helpers from Tables/Tax_Rates.csv ----------

def _load_tax_csv() -> Optional[pd.DataFrame]:
    try:
        tax_path = Path(__file__).parent / "Tables" / "Tax_Rates.csv"
        if not tax_path.exists():
            return None
        df = pd.read_csv(tax_path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception:
        return None

def load_premium_tax(province_code: str, default: float = 0.033) -> float:
    try:
        df = _load_tax_csv()
        if df is None:
            return default
        prov_col = next((c for c in df.columns if c.lower() in {"province", "prov"}), None)
        if not prov_col:
            return default
        df[prov_col] = df[prov_col].astype(str).str.upper().str.strip()
        numeric_cols = [c for c in df.columns if c != prov_col]
        cand_cols = [c for c in numeric_cols if "premium" in c.lower() and "tax" in c.lower()] or numeric_cols
        for c in cand_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() < 0.6:
                continue
            row = df.loc[df[prov_col] == province_code]
            if row.empty:
                continue
            val = float(row.iloc[0][c])
            if val > 1.0:
                val /= 100.0
            if 0 <= val < 0.25:
                return val
        return default
    except Exception:
        return default

def load_personal_tax_rate(province_code: str, default: float = 0.50) -> float:
    try:
        df = _load_tax_csv()
        if df is None:
            return default
        prov_col = next((c for c in df.columns if c.lower() in {"province", "prov"}), None)
        if not prov_col:
            return default
        df[prov_col] = df[prov_col].astype(str).str.upper().str.strip()
        if "Personal_Income" in df.columns:
            row = df.loc[df[prov_col] == province_code]
            if not row.empty:
                val = float(pd.to_numeric(row.iloc[0]["Personal_Income"], errors="coerce"))
                if val > 1.0:
                    val /= 100.0
                if 0 <= val < 1:
                    return val
        def score(col: str) -> int:
            lc = col.lower(); s = 0
            if "personal_income" in lc: s += 100
            if "personal" in lc: s += 60
            if "marginal" in lc: s += 40
            if "income" in lc: s += 20
            if "tax" in lc: s += 10
            return s
        numeric_cols = [c for c in df.columns if c != prov_col]
        numeric_cols.sort(key=score, reverse=True)
        for c in numeric_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() < 0.6:
                continue
            row = df.loc[df[prov_col] == province_code]
            if row.empty:
                continue
            val = float(pd.to_numeric(row.iloc[0][c], errors="coerce"))
            if np.isnan(val):
                continue
            if val > 1.0:
                val /= 100.0
            if 0 <= val < 1.0:
                return val
        return default
    except Exception:
        return default

def load_cg_inclusion(province_code: str, default: float = 0.50) -> float:
    try:
        df = _load_tax_csv()
        if df is None:
            return default
        prov_col = next((c for c in df.columns if c.lower() in {"province", "prov"}), None)
        if not prov_col:
            return default
        df[prov_col] = df[prov_col].astype(str).str.upper().str.strip()
        if "CG_Inclusion" in df.columns:
            row = df.loc[df[prov_col] == province_code]
            if not row.empty:
                val = float(pd.to_numeric(row.iloc[0]["CG_Inclusion"], errors="coerce"))
                if val > 1.0:
                    val /= 100.0
                if 0 <= val <= 1:
                    return val
        return default
    except Exception:
        return default

# ---------- ALT engine (rebalancing toggle) ----------

def run_alt_report(
    start_age: int,
    prem_years: int,
    annual_prem: float,
    horizon_age: int,
    tax_rate: float,
    cg_inclusion: float,
    alloc_interest_pct: float,
    alloc_interest_rate: float,
    alloc_realized_pct: float,
    alloc_realized_rate: float,
    alloc_deferred_pct: float,
    alloc_deferred_rate: float,
    out_path: Path,
    rebalancing: bool = True,
    rebalance_log_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    ALT scenario; also emits a detailed annual rebalance log if `rebalance_log_path` is provided.
    """
    years = max(1, horizon_age - start_age)

    def to_frac(x: float) -> float:
        return x/100.0 if x > 1.0 else x

    aI = to_frac(alloc_interest_pct)
    aR = to_frac(alloc_realized_pct)
    aD = to_frac(alloc_deferred_pct)

    rI = alloc_interest_rate if alloc_interest_rate <= 1 else alloc_interest_rate/100.0
    rR = alloc_realized_rate if alloc_realized_rate <= 1 else alloc_realized_rate/100.0
    rD = alloc_deferred_rate if alloc_deferred_rate <= 1 else alloc_deferred_rate/100.0

    # Balances at end of previous year (start at 0)
    bal_I = bal_R = bal_D = 0.0
    deferred_tax_cum = 0.0  # running liability

    # Summary rows (existing ALT_Report)
    sum_rows = []

    # Detailed rebalance ledger rows
    log_rows = []

    for yr in range(1, years + 1):
        age = start_age + yr
        dep = annual_prem if yr <= prem_years else 0.0

        # --- BOY state BEFORE deposit/rebalance for the log
        boy_I, boy_R, boy_D = bal_I, bal_R, bal_D
        boy_total = boy_I + boy_R + boy_D

        # --- Apply deposit at BOY
        if rebalancing:
            # Deposit added to total, then rebalance to fixed weights
            total_boy_after_dep = boy_total + dep
            tgt_I = total_boy_after_dep * aI
            tgt_R = total_boy_after_dep * aR
            tgt_D = total_boy_after_dep * aD

            # Trades to move current balances to target
            trade_I = tgt_I - (boy_I + 0.0)
            trade_R = tgt_R - (boy_R + 0.0)
            trade_D = tgt_D - (boy_D + 0.0)
            # After trades, new starting balances for the return period:
            bal_I, bal_R, bal_D = tgt_I, tgt_R, tgt_D

        else:
            # No rebalancing: deposit split by weights; no trades
            dep_I = dep * aI
            dep_R = dep * aR
            dep_D = dep * aD
            bal_I = boy_I + dep_I
            bal_R = boy_R + dep_R
            bal_D = boy_D + dep_D

            tgt_I = bal_I
            tgt_R = bal_R
            tgt_D = bal_D
            trade_I = trade_R = trade_D = 0.0

        # --- In-year returns and taxes
        inc_I = bal_I * rI
        inc_R = bal_R * rR
        inc_D = bal_D * rD

        tax_I = inc_I * tax_rate
        tax_R = inc_R * tax_rate * cg_inclusion
        tax_current = -(tax_I + tax_R)  # negative (cash out for taxes)

        # Apply taxes where they’re generated
        bal_I += inc_I - tax_I
        bal_R += inc_R - tax_R
        bal_D += inc_D

        # Deferred capital gains tax accrual on the deferred sleeve
        accr_deferred = -(inc_D * tax_rate * cg_inclusion)      # negative number (liability increases)
        deferred_tax_cum += -accr_deferred                      # store as positive liability

        fund_before_deferred = bal_I + bal_R + bal_D
        net_after_tax = fund_before_deferred - deferred_tax_cum

        # --- Summary row (existing ALT_Report)
        sum_rows.append((
            yr, age, dep,
            inc_I + inc_R + inc_D,
            tax_current,
            accr_deferred,
            fund_before_deferred,
            net_after_tax
        ))

        # --- Detailed log row
        log_rows.append((
            yr, age,
            round(boy_I, 10), round(boy_R, 10), round(boy_D, 10), round(boy_total, 10),
            dep,
            aI, aR, aD,
            round(tgt_I, 10), round(tgt_R, 10), round(tgt_D, 10),
            round(trade_I, 10), round(trade_R, 10), round(trade_D, 10),
            inc_I, inc_R, inc_D,
            -tax_I, -tax_R,               # make taxes positive in the CSV
            -accr_deferred,               # positive accrual amount
            bal_I, bal_R, bal_D,
            fund_before_deferred,
            deferred_tax_cum,
            net_after_tax
        ))

    # Build the main summary DataFrame
    df = pd.DataFrame(sum_rows, columns=[
        "Year","Age","Deposit","Gross income","Tax","Accrued RTax deferred",
        "Fund Value before deferred","Net Fund Value After Tax"
    ])

    # Per-year After-Tax IRR (EOY), preserving timing (zeros kept)
    deposits = df["Deposit"].astype(float).tolist()
    net_vals = df["Net Fund Value After Tax"].astype(float).tolist()
    irr_series: List[str] = []
    neg_flows: List[float] = []
    for i in range(len(df)):
        dep = float(deposits[i])
        neg_flows.append(-dep if dep != 0 else 0.0)
        flows = neg_flows[: i+1]
        if not any(cf < 0 for cf in flows) or net_vals[i] <= 0:
            irr_series.append("")
            continue
        flows = flows + [float(net_vals[i])]
        r = irr(flows)
        irr_series.append(f"{r*100:.2f}%" if np.isfinite(r) else "")
    df["After-Tax IRR"] = irr_series

    # Round money columns to cents for the main summary
    money_cols = [
        "Deposit","Gross income","Tax","Accrued RTax deferred",
        "Fund Value before deferred","Net Fund Value After Tax"
    ]
    for c in money_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)

    # Write main summary
    out_path.write_text(df.to_csv(index=False))

    # Build and write the detailed rebalance log
    if rebalance_log_path is not None:
        log_cols = [
            "Year","Age",
            "BOY_I","BOY_R","BOY_D","BOY_Total",
            "Deposit",
            "Target_W_I","Target_W_R","Target_W_D",
            "Target_$ _I","Target_$ _R","Target_$ _D",
            "Trade_I","Trade_R","Trade_D",
            "Inc_I","Inc_R","Inc_D",
            "Tax_I_Paid","Tax_R_Paid","Deferred_CG_Tax_Accrued",
            "EOY_I","EOY_R","EOY_D",
            "Fund_Before_Deferred","Deferred_Tax_Cum","Net_After_Tax"
        ]
        log_df = pd.DataFrame(log_rows, columns=log_cols)

        # Round dollar columns to cents (leave weights as-is)
        dollar_cols = [c for c in log_cols if c not in ["Year","Age","Target_W_I","Target_W_R","Target_W_D"]]
        for c in dollar_cols:
            log_df[c] = pd.to_numeric(log_df[c], errors="coerce").round(2)

        rebalance_log_path.write_text(log_df.to_csv(index=False))

    return df, float(df.iloc[-1]["Net Fund Value After Tax"])


# ---------- Core ----------

def main() -> None:
    print("\n--- Please Enter Policy Details ---\n")

    def ask_int(prompt: str) -> int:    return int(input(prompt).strip())
    def ask_float(prompt: str) -> float:return float(input(prompt).strip())

    # --- Paths
    script_dir = os.path.dirname(__file__)
    settings_dir = Path(script_dir) / "Settings"
    reports_dir  = Path(script_dir) / "Reports"
    reports_dir.mkdir(exist_ok=True)

    prov_choice   = ask_int_default("Province (1=QC, 2=ON): ", 1)
    client_choice = ask_int_default("Ownership (1=Corp, 2=Personal): ", 2)
    gender_choice = ask_int_default("Gender (1=Male, 2=Female): ", 2)
    age_input     = ask_int_default("Issue Age (e.g., 30): ", 30)
    smoker_choice = ask_int_default("Smoker? (1=Non-Smoker, 2=Smoker): ", 1)

    face_amount   = ask_float_default("Face Amount (e.g., 10000000): ", 10000000)
    coi_choice    = ask_int_default("COI Type (1=YRT70, 2=YRT85, 3=L100): ", 1)
    prem_years    = ask_int_default("Planned Premium Years (e.g., 15): ", 15)
    base_rate_in  = ask_float_default("Base Annual Rate (e.g., 5.25): ", 5.25)
    safe_rate_in  = ask_float_default("Guaranteed Rate (e.g., 2.00): ", 2.00)

    prov_map = {1: "QC", 2: "ON"}
    province = prov_map.get(prov_choice, "QC")
    client_type = "Corporate" if client_choice == 1 else "Personal"
    gender      = "M" if gender_choice == 1 else "F"
    smk_suf     = "S" if smoker_choice == 2 else ""
    smk_display = "SM" if smoker_choice == 2 else "NS"
    start_age   = age_input
    coi_map     = {1: "YRT70", 2: "YRT85", 3: "LEVEL"}
    coi_type    = coi_map.get(coi_choice, "LEVEL")
    rate_base   = base_rate_in / 100.0
    rate_safe   = safe_rate_in / 100.0

    # --- Load taxes
    status("Calculating ...")
    coi_path = Path(script_dir) / "Tables" / "COI.csv"
    try:
        coi_df = pd.read_csv(coi_path)
        if "PolicyYear" not in coi_df.columns:
            if "Year" in coi_df.columns:
                coi_df = coi_df.rename(columns={"Year": "PolicyYear"})
            else:
                coi_df["PolicyYear"] = np.arange(1, len(coi_df) + 1, dtype=int)
        coi_df["PolicyYear"] = coi_df["PolicyYear"].astype(int)
    except Exception as e:
        clear_status()
        print(f"\nERROR loading COI.csv: {e}")
        sys.exit(1)

    premium_tax_rate  = load_premium_tax(province)
    personal_tax_rate = load_personal_tax_rate(province)

    coi_key = f"{coi_type}-{gender}{start_age}{smk_suf}"
    if coi_key not in coi_df.columns:
        alts = [coi_key, coi_key.replace("LEVEL", "Level"), coi_key.replace("Level", "LEVEL")]
        for k in alts:
            if k in coi_df.columns:
                coi_key = k
                break
        else:
            clear_status()
            print(f"\nERROR: COI column '{coi_key}' not found in COI.csv")
            sys.exit(1)
    coi_df[coi_key] = pd.to_numeric(coi_df[coi_key], errors="coerce").fillna(0.0)

    # --- Header
    clear_status()
    if supports_ansi():
        print(f"\n\033[1;97mPOLICY REPORT ({'QC' if province=='QC' else 'ON' if province=='ON' else province} / {client_type})\033[0m")
        print("\033[1;97m-------------------------------\033[0m")
    else:
        print(f"\nPOLICY REPORT ({'QC' if province=='QC' else 'ON' if province=='ON' else province} / {client_type})")
        print("-------------------------------")

    _start_dots("Calculating")

    # --- Sim setup
    months_total = (100 - start_age) * 12
    last_m = last_coi_months(start_age, coi_type)
    monthly_rate_cache = {}

    def mrate(annual_rate: float) -> float:
        if annual_rate not in monthly_rate_cache:
            monthly_rate_cache[annual_rate] = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
        return monthly_rate_cache[annual_rate]

    def coi_rate_for_year(policy_year: int) -> float:
        row = coi_df.loc[coi_df["PolicyYear"] == policy_year]
        return float((row[coi_key].iloc[0] if not row.empty else coi_df[coi_key].iloc[-1]))

    # -------- Simulation engine (monthly) --------

    def simulate_monthly(annual_prem: float, prem_years_int: int, annual_rate: float, db_option: int) -> Tuple[pd.DataFrame, int]:
        rows = []
        fv_boy = 0.0
        mr = mrate(annual_rate)

        for m in range(1, months_total + 1):
            py  = (m - 1) // 12 + 1
            moy = (m - 1) % 12 + 1

            deposit = annual_prem if (moy == 1 and py <= prem_years_int) else 0.0
            tax_amt = deposit * premium_tax_rate
            fv1 = (fv_boy + deposit) - tax_amt

            if db_option == 1:
                naar = max(0.0, face_amount - fv1)   # Level DB
            else:
                naar = face_amount                    # Fund+ DB

            coi_rate_ann = coi_rate_for_year(py) if (m <= last_m and last_m > 0) else 0.0
            coi_month = (naar / 1000.0) * (coi_rate_ann / 12.0)

            fv_net = fv1 - coi_month
            growth = fv_net * mr
            fv_eoy = fv_net + growth

            rows.append((py, moy, fv_boy, deposit, tax_amt, fv1,
                         face_amount, naar, coi_rate_ann, coi_month,
                         fv_net, annual_rate, growth, fv_eoy))
            fv_boy = fv_eoy

        cols = ["Year","Month","FV_BOY","Deposit","TAX","FV1","FaceValue","NAAR","COI_Rate","COI","FV_Net","AnnualRate","Growth","FV_EOY"]
        return pd.DataFrame(rows, columns=cols), last_m

    def fv_at_last_coi_fractional(annual_prem: float, years: float, annual_rate: float, db_option: int) -> float:
        whole = int(np.floor(years))
        frac = years - whole
        fv_boy = 0.0
        mr = mrate(annual_rate)
        for m in range(1, months_total + 1):
            py = (m - 1) // 12 + 1
            moy = (m - 1) % 12 + 1

            C = fv_boy
            if moy == 1:
                if py <= whole:
                    D = annual_prem
                elif py == whole + 1 and frac > 0:
                    D = annual_prem * frac
                else:
                    D = 0.0
            else:
                D = 0.0

            E = (C + D) - D * premium_tax_rate
            F = face_amount
            G = max(0.0, F - E) if db_option == 1 else F
            H = coi_rate_for_year(py) if (m <= last_m and last_m > 0) else 0.0
            I = (G / 1000.0) * (H / 12.0)
            J = E - I
            fv_boy = J * (1.0 + mr)

        return float(fv_boy if last_m > 0 else 0.0)

    def total_db_at_85(annual_prem: float, years: float, annual_rate: float, db_option: int) -> float:
        df, _ = simulate_monthly(annual_prem, int(np.floor(years + 1e-9)), annual_rate, db_option)
        idx = (85 - start_age) * 12
        fv85 = float(df.iloc[idx - 1]["FV_EOY"]) if idx > 0 else 0.0
        return float(face_amount if db_option == 1 else face_amount + fv85)

    def find_min_premium_for_years(years_int: int, annual_rate: float, db_option: int) -> int:
        low, high = 0.0, 1.0

        def ok_case(df):
            horizon = last_m
            return _no_lapse(df, horizon)

        for _ in range(40):
            df, _ = simulate_monthly(high, years_int, annual_rate, db_option)
            if ok_case(df):
                break
            high *= 2.0

        for _ in range(80):
            mid = (low + high) / 2.0
            df, _ = simulate_monthly(mid, years_int, annual_rate, db_option)
            if ok_case(df):
                high = mid
            else:
                low = mid
            if abs(high - low) < 0.01:
                break

        return int(round(high))

    def find_years_for_same_premium(annual_prem: float, annual_rate: float, db_option: int) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            if fv_at_last_coi_fractional(annual_prem, hi, annual_rate, db_option) >= 0: break
            hi *= 2.0
            if hi > 100: break
        best = hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fv_mid = fv_at_last_coi_fractional(annual_prem, mid, annual_rate, db_option)
            if fv_mid >= 0:
                best = mid; hi = mid
            else:
                lo = mid
            if abs(hi - lo) < 1e-3: break
        return round(best, 2)

    def irr_to_85(annual_prem: float, years: float, db85: float) -> float:
        horizon_years = max(1, 85 - start_age)
        y_paid = min(int(np.floor(years + 1e-9)), horizon_years)
        flows = [-annual_prem if t < y_paid else 0.0 for t in range(horizon_years)]
        flows.append(db85)
        r = irr(flows)
        return r if np.isfinite(r) else float("nan")

    def first_lapse_age(df_monthly: pd.DataFrame, start_age: int) -> Optional[int]:
        neg = df_monthly["FV_EOY"].to_numpy() < 0
        if not neg.any():
            return None
        m_idx = int(np.argmax(neg)) + 1
        pol_year = (m_idx + 11) // 12
        return start_age + pol_year

    def run_case_for_table(db_option: int, annual_prem: float, prem_years_int: int, annual_rate: float):
        df, _ = simulate_monthly(annual_prem, prem_years_int, annual_rate, db_option=db_option)
        lapse_age = first_lapse_age(df, start_age)

        if db_option == 1:
            db85 = total_db_at_85(annual_prem, prem_years_int, annual_rate, db_option=1)
            irr_tf = irr_to_85(annual_prem, prem_years_int, db85)
            irr_te = tax_equiv(irr_tf, personal_tax_rate)
            return (prem_years_int, paidup_age(start_age, prem_years_int), db85, irr_tf, irr_te, None)

        if lapse_age is not None:
            return (prem_years_int, paidup_age(start_age, prem_years_int), None, None, None, lapse_age)

        db85 = total_db_at_85(annual_prem, prem_years_int, annual_rate, db_option=2)
        irr_tf = irr_to_85(annual_prem, prem_years_int, db85)
        irr_te = tax_equiv(irr_tf, personal_tax_rate)
        return (prem_years_int, paidup_age(start_age, prem_years_int), db85, irr_tf, irr_te, None)

    def build_table_four_rates(db_option: int):
        r_m1 = max(0.0, rate_base - 0.01)
        r_0  = rate_base
        r_p1 = rate_base + 0.01
        r_p2 = rate_base + 0.02

        rows = []
        notes = []

        if db_option == 1:
            base_prem = find_min_premium_for_years(prem_years, r_0, db_option=1)
            def years_for_same_prem(r):
                return find_years_for_same_premium(base_prem, r, db_option=1)
            for r, label in [(r_m1, "−1%"), (r_0, "Target"), (r_p1, "+1%"), (r_p2, "+2%")]:
                yrs = years_for_same_prem(r)
                yrs_disp = f"~{int(round(yrs))}" if r != r_0 else prem_years
                db85 = total_db_at_85(base_prem, yrs, r, db_option=1)
                irr_tf = irr_to_85(base_prem, yrs, db85)
                irr_te = tax_equiv(irr_tf, personal_tax_rate)
                rows.append((label, base_prem, r*100, yrs_disp, paidup_age(start_age, yrs), db85, irr_tf, irr_te))
            return rows, None, base_prem, notes
        else:
            base_prem = find_min_premium_for_years(prem_years, r_0, db_option=2)
            for r, label in [(r_m1, "−1%"), (r_0, "Target"), (r_p1, "+1%"), (r_p2, "+2%")]:
                yrs, pu_age_val, db85, irr_tf, irr_te, lapse_age = run_case_for_table(
                    db_option=2, annual_prem=base_prem, prem_years_int=prem_years, annual_rate=r
                )
                if lapse_age is not None:
                    rows.append((label, base_prem, r*100, prem_years, pu_age_val, None, None, None))
                    notes.append(f"{label}: Lapses @ age {lapse_age}")
                else:
                    rows.append((label, base_prem, r*100, prem_years, pu_age_val, db85, irr_tf, irr_te))
            return rows, None, base_prem, notes

    def print_table_four_rates(title: str, rows: List[Tuple], notes: List[str]) -> None:
        if title.strip():
            print(title)
        cols = [
            ("Rate Case",        9),
            ("Rate",             7),
            ("Premium",         10),
            ("Years",            7),
            ("Paid-Up",         10),
            ("DB @ 85",         12),
            ("After-Tax IRR",    13),
            ("Pre-Tax IRR",     13),
        ]
        header = " | ".join(name.ljust(w) for name, w in cols)
        sep    = "-+-".join("-" * w for _, w in cols)
        print(header); print(sep)

        for label, prem, rate_pct, yrs, pu_age, db85, irr_tf, irr_te in rows:
            db_str = f"{abbrev_money(db85):>{cols[5][1]}}" if db85 is not None else " " * (cols[5][1]-2) + "N/A"
            irr_tf_str = f"{irr_tf*100:>{cols[6][1]-1}.2f}%" if irr_tf is not None else " " * (cols[6][1]-3) + "N/A"
            irr_te_str = f"{irr_te*100:>{cols[7][1]-1}.2f}%" if irr_te is not None else " " * (cols[7][1]-3) + "N/A"
            cells = [
                f"{label:<{cols[0][1]}}",
                f"{rate_pct:>{cols[1][1]-1}.2f}%",
                f"{abbrev_money(prem):<{cols[2][1]}}",
                f"{str(yrs):>{cols[3][1]}}",
                f"{pu_age:>{cols[4][1]}}",
                db_str,
                irr_tf_str,
                irr_te_str,
            ]
            if supports_ansi() and label == "Target":
                print("\033[1;32m" + " | ".join(cells) + "\033[0m")
            else:
                print(" | ".join(cells))

        if notes:
            print("\nNotes: " + " | ".join(notes))

    # ---- Main compute & CSV generation ----
    try:
        rows_level, _, base_prem_level, notes_level = build_table_four_rates(db_option=1)
        df_level_base, _ = simulate_monthly(base_prem_level, prem_years, rate_base, db_option=1)

        (Path(reports_dir) / "LEVEL_Data.csv").write_text(df_level_base.to_csv(index=False))

        ann = df_level_base.copy()
        for c in ["Year", "Deposit", "TAX", "COI", "Growth", "FV_EOY"]:
            ann[c] = pd.to_numeric(ann[c], errors="coerce").fillna(0.0)

        sums = ann.groupby("Year", as_index=False)[["Deposit", "TAX", "COI", "Growth"]].sum()
        last = ann.groupby("Year", as_index=False)[["FV_EOY"]].last()

        annual = sums.merge(last, on="Year", how="left")
        annual["Age"]        = (start_age + annual["Year"]).astype(int)
        annual["Cost"]       = (annual["COI"] + annual["TAX"]).astype(float)
        annual["Fund Value"] = annual["FV_EOY"].astype(float)
        annual["Face Value"] = float(face_amount)
        annual["Deposit"]    = annual["Deposit"].astype(float)
        annual["Income"]     = annual["Growth"].astype(float)

        out_annual = annual[["Year","Age","Deposit","Income","Cost","Fund Value","Face Value"]].copy()
        out_annual[["Deposit","Income","Cost","Fund Value","Face Value"]] = (
            out_annual[["Deposit","Income","Cost","Fund Value","Face Value"]].round(2)
        )

        if client_type == "Corporate":
            out_annual["NCPI"] = ""
            (Path(reports_dir) / "LVL_Report.csv").write_text(
                out_annual[["Year","Age","Deposit","Income","Cost","Fund Value","Face Value","NCPI"]].to_csv(index=False)
            )
        else:
            deposits_lvl = out_annual["Deposit"].astype(float).tolist()
            face_vals = out_annual["Face Value"].astype(float).tolist()
            irr_strs: List[str] = []
            te_strs:  List[str] = []
            neg_flows: List[float] = []
            for i in range(len(out_annual)):
                dep = float(deposits_lvl[i]); fv = float(face_vals[i])
                neg_flows.append(-dep if dep != 0 else 0.0)
                flows = [cf for cf in neg_flows[: i+1]]
                if all(abs(x) < 1e-9 for x in flows):
                    r = float("nan")
                else:
                    flows.append(fv)
                    r = irr(flows)
                if np.isfinite(r):
                    irr_strs.append(f"{r*100:.2f}%")
                    te_strs.append(f"{tax_equiv(r, personal_tax_rate)*100:.2f}%")
                else:
                    irr_strs.append("")
                    te_strs.append("")
            out_annual["IRR"] = irr_strs
            out_annual["TE IRR"] = te_strs
            (Path(reports_dir) / "LVL_Report.csv").write_text(
                out_annual[["Year","Age","Deposit","Income","Cost","Fund Value","Face Value","IRR","TE IRR"]].to_csv(index=False)
            )

    except Exception as e:
        _stop_dots()
        print("\nERROR:", e)
        traceback.print_exc()
        sys.exit(1)
    finally:
        _stop_dots()

    # ---- Console tables ----
    if supports_ansi():
        print(f"\n\033[1;97m${face_amount/1_000_000:.0f}M UL (LEVEL) / {('M' if gender=='M' else 'F')}{start_age} {smk_display} / {coi_type}\033[0m")
        # Bright yellow #FFF111 (RGB mode)
        print(f"\n\033[1;38;2;255;241;17mANNUAL PREMIUM: ${base_prem_level:,.0f}\033[0m\n")
    else:
        print(f"\n${face_amount/1_000_000:.0f}M UL (LEVEL) / {('M' if gender=='M' else 'F')}{start_age} {smk_display} / {coi_type}")
        print(f"\nANNUAL PREMIUM: ${base_prem_level:,.0f}\n")

    print_table_four_rates("", rows_level, [])

    safe_prem_level = find_min_premium_for_years(prem_years, rate_safe, db_option=1)
    if supports_ansi():
        print(f"\n\033[3mFor reference: a guaranteed rate of {safe_rate_in:.0f}% would require {prem_years} premiums of {abbrev_money(safe_prem_level)}.\033[0m\n")
    else:
        print(f"\nFor reference: a guaranteed rate of {safe_rate_in:.0f}% would require {prem_years} premiums of {abbrev_money(safe_prem_level)}.\n")

    # --- ALT (Alternative Investment) flow ---
    cg_inclusion = load_cg_inclusion(province)

    # ---- Defaults: load from Settings/Fund.csv if present ----
    # Expected columns (case-insensitive, flexible):
    #   Interest_W, Interest_Rate, Realized_W, Realized_Rate, Deferred_W, Deferred_Rate, Rebalance
    # Weights can be 0–1 or 0–100; Rates can be 0–1 or 0–100. Rebalance: 1=ON, 0=OFF
    def _to_frac(x: float) -> float:
        return x/100.0 if x > 1.0 else x

    alt_alloc_interest = 0.50   # default 50%
    alt_rate_interest  = 0.035  # default 3.5%
    alt_alloc_realized = 0.30   # default 30%
    alt_rate_realized  = 0.10   # default 10.0%
    alt_alloc_deferred = 0.20   # default 20%
    alt_rate_deferred  = 0.07   # default 7.0%
    alt_rebal_on       = True   # default: Rebalancing ON

    fund_csv = settings_dir / "Fund.csv"
    if fund_csv.exists():
        try:
            fund_df = pd.read_csv(fund_csv)
            if len(fund_df) >= 1:
                row = fund_df.iloc[0]
                def get(colnames, default=None):
                    for c in fund_df.columns:
                        if c.strip().lower() in [n.strip().lower() for n in colnames]:
                            v = row[c]
                            try:
                                return float(v)
                            except Exception:
                                return default
                    return default

                iw = get(["Interest_W","W_I","Weight_I","InterestWeight"], alt_alloc_interest*100)
                ir = get(["Interest_Rate","Rate_I","R_I"], alt_rate_interest*100)
                rw = get(["Realized_W","W_R","Weight_R","RealizedWeight"], alt_alloc_realized*100)
                rr = get(["Realized_Rate","Rate_R","R_R"], alt_rate_realized*100)
                dw = get(["Deferred_W","W_D","Weight_D","DeferredWeight"], alt_alloc_deferred*100)
                dr = get(["Deferred_Rate","Rate_D","R_D"], alt_rate_deferred*100)
                rb = get(["Rebalance","Balancing","Rebalancing"], 1.0)

                alt_alloc_interest = _to_frac(iw if iw is not None else 50.0)
                alt_rate_interest  = _to_frac(ir if ir is not None else 3.5)
                alt_alloc_realized = _to_frac(rw if rw is not None else 30.0)
                alt_rate_realized  = _to_frac(rr if rr is not None else 10.0)
                alt_alloc_deferred = _to_frac(dw if dw is not None else 20.0)
                alt_rate_deferred  = _to_frac(dr if dr is not None else 7.0)
                alt_rebal_on       = bool(int(rb)) if rb is not None else True

                # Normalize weights if they don't sum to 1
                wsum = alt_alloc_interest + alt_alloc_realized + alt_alloc_deferred
                if wsum > 0:
                    alt_alloc_interest /= wsum
                    alt_alloc_realized /= wsum
                    alt_alloc_deferred /= wsum
        except Exception as e:
            print(f"Warning: couldn't parse Settings/Fund.csv ({e}). Using built-in defaults.")

    def show_alt(ai, ri, ar, rr, ad, rd, rb):
        def p(x): return x*100 if x <= 1 else x
        def r(x): return x*100 if x <= 1 else x

        if supports_ansi():
            print(f"\n\033[1;97mALT INVESTMENT REPORT ({'QC' if province=='QC' else 'ON' if province=='ON' else province} / {client_type})\033[0m")
            print("\033[1;97m-------------------------------------\033[0m")
        else:
            print(f"\nALT INVESTMENT REPORT ({'QC' if province=='QC' else 'ON' if province=='ON' else province} / {client_type})")
            print("-------------------------------------")

        print(f"\n1) {p(ai):.0f}% Interest FI @ {r(ri):.1f}%")
        print(f"2) {p(ar):.0f}% Realized CG @ {r(rr):.1f}%")
        print(f"3) {p(ad):.0f}% Deferred CG @ {r(rd):.1f}%")
        print(f"\nRebalancing: {'On (Annual)' if rb else 'Off (Drift)'}\n")

    # Show defaults and prompt
    show_alt(
        alt_alloc_interest, alt_rate_interest,
        alt_alloc_realized, alt_rate_realized,
        alt_alloc_deferred, alt_rate_deferred,
        alt_rebal_on
    )
    choice = (read_input("To proceed press 1, to modify press 2: ", "1").strip() or "1")

    while choice == "2":
        try:
            ai = float(read_input("\n1) Fixed income allocation: ", f"{alt_alloc_interest*100:.0f}").strip() or f"{alt_alloc_interest*100:.0f}")
            ri = float(read_input("   Annual rate of return: ", f"{alt_rate_interest*100:.1f}").strip() or f"{alt_rate_interest*100:.1f}")
            ar = float(read_input("\n2) Realized CG allocation: ", f"{alt_alloc_realized*100:.0f}").strip() or f"{alt_alloc_realized*100:.0f}")
            rr = float(read_input("   Annual rate of return: ", f"{alt_rate_realized*100:.1f}").strip() or f"{alt_rate_realized*100:.1f}")
            ad = float(read_input("\n3) Deferred CG allocation: ", f"{alt_alloc_deferred*100:.0f}").strip() or f"{alt_alloc_deferred*100:.0f}")
            rd = float(read_input("   Annual rate of return: ", f"{alt_rate_deferred*100:.1f}").strip() or f"{alt_rate_deferred*100:.1f}")
            rb = read_input("\nRebalancing? (1 = Annual, 2 = Off): ", "1").strip()

            total_alloc = ai + ar + ad
            if abs(total_alloc - 100.0) > 1e-6:
                print("Please specify allocations to a total of 100%")
                continue

            alt_alloc_interest = ai / 100.0
            alt_rate_interest  = ri / (100.0 if ri > 1 else 1.0)
            alt_alloc_realized = ar / 100.0
            alt_rate_realized  = rr / (100.0 if rr > 1 else 1.0)
            alt_alloc_deferred = ad / 100.0
            alt_rate_deferred  = rd / (100.0 if rd > 1 else 1.0)
            alt_rebal_on       = (rb == "1")

            show_alt(alt_alloc_interest, alt_rate_interest, alt_alloc_realized, alt_rate_realized, alt_alloc_deferred, alt_rate_deferred, alt_rebal_on)
            choice = (read_input("To proceed press 1, to modify press 2: ", "1").strip() or "1")
        except Exception:
            print("Invalid entry. Please try again.")

    if choice == "1":
        horizon_age = 85
        alt_path = Path(reports_dir) / "ALT_Report.csv"
        rebalance_log_path = (Path(reports_dir) / "ALT_Rebalance.csv") if alt_rebal_on else None

        alt_df, alt_net_85 = run_alt_report(
            start_age=start_age,
            prem_years=prem_years,
            annual_prem=base_prem_level,
            horizon_age=horizon_age,
            tax_rate=personal_tax_rate,
            cg_inclusion=cg_inclusion,
            alloc_interest_pct=alt_alloc_interest,
            alloc_interest_rate=alt_rate_interest,
            alloc_realized_pct=alt_alloc_realized,
            alloc_realized_rate=alt_rate_realized,
            alloc_deferred_pct=alt_alloc_deferred,
            alloc_deferred_rate=alt_rate_deferred,
            out_path=alt_path,
            rebalancing=alt_rebal_on,
            rebalance_log_path=rebalance_log_path,
        )

        total_alt = alt_net_85
        total_ul  = float(face_amount)

        # Advantage vs ALT
        adv_dol = total_ul - total_alt
        adv_pct = (adv_dol / total_alt * 100.0) if total_alt != 0 else float("inf")
        pct_str = "∞" if total_alt == 0 else f"{adv_pct:+.0f}%"

        # Overall after-tax IRR to age 85 for ALT (one decimal place)
        horizon_years = max(1, 85 - start_age)
        y_paid = min(prem_years, horizon_years)
        flows = [-base_prem_level if t < y_paid else 0.0 for t in range(horizon_years)]
        flows.append(total_alt)
        alt_irr = irr(flows)
        alt_irr_str = f"{alt_irr*100:.1f}%" if np.isfinite(alt_irr) else ""

        # Colour: green if UL wins, red if loses
        if supports_ansi():
            color = "\033[1;92m" if adv_dol > 0 else "\033[1;91m"
            print(f"\n{color}AFTER-TAX VALUE AGE 85: ${total_alt:,.0f} (IRR {alt_irr_str})\033[0m")
        else:
            print(f"\nAFTER-TAX VALUE AGE 85: ${total_alt:,.0f} (IRR {alt_irr_str})")

        print(f"\nTotal UL Value age 85: ${total_ul:,.0f}")
        print(f"   --> UL vs Alt Fund: {adv_dol:+,.0f} ({pct_str})")

        if supports_ansi():
            print(
                f"\n\033[3mNote: annual rebalancing to fixed weights is "
                f"{'ON' if alt_rebal_on else 'OFF'}; realised CG taxed yearly; deferred CG accrues to age 85.\033[0m"
            )
        else:
            print(
                "\nNote: annual rebalancing to fixed weights is "
                f"{'ON' if alt_rebal_on else 'OFF'}; realised CG taxed yearly; deferred CG accrues to age 85."
            )

        print("\n")


if __name__ == "__main__":
    main()
