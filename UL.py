# UL.py — Universal Life monthly projection + sales tables (Level & Fund+)
# Files required next to UL.py:
#   - Tables/COI.csv       (columns: PolicyYear, and e.g. YRT70-F30, YRT70-F30S, YRT85-M45, LEVEL-F30, ...)
#   - Tables/Tax_Rates.csv (must include Province column and Personal_Income for personal marginal rate)
#
# I/O:
# - Prompts: Province → Ownership → Gender → Age → Smoker → Face → COI Type → Premium Years → Base Rate → Safe Rate
# - Prints two tables (Level DB and Fund+ DB) at four rates: Base−1%, Base, Base+1%, Base+2%.
# - Writes CSVs from the LEVEL base-case monthly sim:
#     LEVEL_Data.csv   : MONTHLY rows (unchanged): Year, Month, FV_BOY, Deposit, TAX, FV1, FaceValue, NAAR,
#                        COI_Rate, COI, FV_Net, AnnualRate, Growth, FV_EOY
#     LEVEL_Report.csv : ANNUAL summary (EOY ages):
#         Corporate → Year, Age, Deposit, Income, Cost, Fund Value, Face Value, NCPI(blank)
#         Personal  → Year, Age, Deposit, Income, Cost, Fund Value, Face Value, IRR, TE IRR
#
# Notes:
# - IRR per year is computed as of EOY of each policy year using negative deposits paid up to that year
#   and a terminal inflow equal to the Level Death Benefit (Face Value).
# - TE IRR = IRR / (1 - Personal_Income tax rate for the selected province).
# - IRR/TE IRR rendered as strings like "5.37%" in the annual report; monthly file unchanged.
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

def paidup_age(start_age: int, years: float) -> int:
    return int(round(start_age + years))  # EOY convention

def coi_limit_age(kind: str) -> int:
    return 70 if kind == "YRT70" else 85 if kind == "YRT85" else 100

def last_coi_months(start_age: int, kind: str) -> int:
    return max(0, coi_limit_age(kind) - start_age) * 12

def _no_lapse(df_monthly: pd.DataFrame, horizon_months: int) -> bool:
    """True if FV_EOY never goes negative up to horizon_months."""
    if horizon_months <= 0:
        return True
    return float(df_monthly["FV_EOY"].iloc[:horizon_months].min()) >= 0.0


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

# ---------- Core ----------

def main() -> None:
    print("\n--- Please Enter Policy Details ---\n")

    def ask_int(prompt: str) -> int:    return int(input(prompt).strip())
    def ask_float(prompt: str) -> float:return float(input(prompt).strip())

    prov_choice   = ask_int("Province (1=QC, 2=ON): ")
    client_choice = ask_int("Ownership (1=Corp, 2=Personal): ")
    gender_choice = ask_int("Gender (1=Male, 2=Female): ")
    age_input     = ask_int("Issue Age (e.g., 30): ")
    smoker_choice = ask_int("Smoker? (1=Non-Smoker, 2=Smoker): ")

    face_amount   = ask_float("Face Amount (e.g., 10000000): ")
    coi_choice    = ask_int("COI Type (1=YRT70, 2=YRT85, 3=L100): ")
    prem_years    = ask_int("Planned Premium Years (e.g., 15): ")
    base_rate_in  = ask_float("Base Annual Rate (e.g., 5.25): ")
    safe_rate_in  = ask_float("Guaranteed Rate (e.g., 2.00): ")

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

    premium_tax_rate  = load_premium_tax(province)
    personal_tax_rate = load_personal_tax_rate(province)

    # COI.csv
    status("Calculating ...")
    script_dir = os.path.dirname(__file__)
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

    clear_status()
    # print(f"\nProvince: {province} | Premium Tax: {premium_tax_rate*100:.2f}% | Client: {client_type}")
    # print(f"COI Column: {coi_key}")
    print(f"\nPOLICY REPORT ({'QC' if province=='QC' else 'ON' if province=='ON' else province} / {client_type})")
    print(f"-------------------------------")

    _start_dots("Calculating")

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
            horizon = last_m  # require survival through COI charge horizon
            return _no_lapse(df, horizon)

        # grow upper bound
        for _ in range(40):
            df, _ = simulate_monthly(high, years_int, annual_rate, db_option)
            if ok_case(df):
                break
            high *= 2.0

        # bisection
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
        horizon_years = max(1, 85 - start_age)  # EOY 85
        y_paid = min(int(np.floor(years + 1e-9)), horizon_years)
        flows = [-annual_prem if t < y_paid else 0.0 for t in range(horizon_years)]
        flows.append(db85)
        r = irr(flows)
        return r if np.isfinite(r) else float("nan")

    # --- helpers for 4-rate display & lapse detection ---

    def first_lapse_age(df_monthly: pd.DataFrame, start_age: int) -> Optional[int]:
        neg = df_monthly["FV_EOY"].to_numpy() < 0
        if not neg.any():
            return None
        m_idx = int(np.argmax(neg)) + 1  # 1-based
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
        # Rates
        r_m1 = max(0.0, rate_base - 0.01)
        r_0  = rate_base
        r_p1 = rate_base + 0.01
        r_p2 = rate_base + 0.02

        rows = []
        notes = []

        if db_option == 1:
            # LEVEL: base premium at target rate; flex years at other rates
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
            # FUND+: base premium at target rate; keep years fixed across all four rates
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
                # Bold and light green for visibility
                print("\033[1;32m" + " | ".join(cells) + "\033[0m")
            else:
                print(" | ".join(cells))

        if notes:
            print("\nNotes: " + " | ".join(notes))


    # ---- Main compute & CSV generation ----
    try:
        # LEVEL table + baseline monthly run
        rows_level, _, base_prem_level, notes_level = build_table_four_rates(db_option=1)
        df_level_base, _ = simulate_monthly(base_prem_level, prem_years, rate_base, db_option=1)

        # Save MONTHLY file exactly as produced
        (Path(script_dir) / "LEVEL_Data.csv").write_text(df_level_base.to_csv(index=False))

        # Build ANNUAL summary (for LEVEL_Report.csv)
        ann = df_level_base.copy()
        for c in ["Year", "Deposit", "TAX", "COI", "Growth", "FV_EOY"]:
            ann[c] = pd.to_numeric(ann[c], errors="coerce").fillna(0.0)

        sums = ann.groupby("Year", as_index=False)[["Deposit", "TAX", "COI", "Growth"]].sum()
        last = ann.groupby("Year", as_index=False)[["FV_EOY"]].last()

        annual = sums.merge(last, on="Year", how="left")
        annual["Age"]        = (start_age + annual["Year"]).astype(int)  # EOY age
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
            (Path(script_dir) / "LEVEL_Report.csv").write_text(
                out_annual[["Year","Age","Deposit","Income","Cost","Fund Value","Face Value","NCPI"]].to_csv(index=False)
            )
        else:
            # Personal-only IRR/TE IRR (per-year, EOY)
            deposits = out_annual["Deposit"].astype(float).tolist()
            face_vals = out_annual["Face Value"].astype(float).tolist()
            irr_strs: List[str] = []
            te_strs:  List[str] = []
            neg_flows: List[float] = []
            for i in range(len(out_annual)):
                dep = float(deposits[i]); fv = float(face_vals[i])
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
            (Path(script_dir) / "LEVEL_Report.csv").write_text(
                out_annual[["Year","Age","Deposit","Income","Cost","Fund Value","Face Value","IRR","TE IRR"]].to_csv(index=False)
            )

        # FUND+ table (console only)
        # rows_sf, _, base_prem_fund, notes_sf = build_table_four_rates(db_option=2)

    except Exception as e:
        _stop_dots()
        print("\nERROR:", e)
        traceback.print_exc()
        sys.exit(1)
    finally:
        _stop_dots()

    # ---- Console tables ----
    # ---- Console tables ----
    if supports_ansi():
        print(f"\n\033[1;97m${face_amount/1_000_000:.0f}M UL (LEVEL) / {('M' if gender=='M' else 'F')}{start_age} {smk_display} / {coi_type}")
        print(f"ANNUAL PREMIUM: ${base_prem_level:,.0f}\033[0m\n")
    else:
        print(f"\n${face_amount/1_000_000:.0f}M UL (LEVEL) / {('M' if gender=='M' else 'F')}{start_age} {smk_display} / {coi_type}")
        print(f"ANNUAL PREMIUM: ${base_prem_level:,.0f}\n")


   # head_sf = (
   #     f"UL (FUND+) / {('Male' if gender=='M' else 'Female')}{start_age} {smk_display} / {coi_type}\n"
   #     f"Annual Premium: ${base_prem_fund:,.0f}"
   # )

    # after the two heading prints
    print_table_four_rates("", rows_level, [])

    # print_table_four_rates(head_sf, rows_sf, notes_sf)

    # Guaranteed (safe) rate reference line
    safe_prem_level = find_min_premium_for_years(prem_years, rate_safe, db_option=1)
    if supports_ansi():
        print(f"\n\n\033[3mFor reference: a guaranteed rate of {safe_rate_in:.0f}% would require {prem_years} premiums of {abbrev_money(safe_prem_level)}.\033[0m\n\n")
    else:
        print(f"\n\nFor reference: a guaranteed rate of {safe_rate_in:.0f}% would require {prem_years} premiums of {abbrev_money(safe_prem_level)}.\n\n")

    

if __name__ == "__main__":
    main()
