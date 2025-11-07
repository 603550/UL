# UL.py — Universal Life monthly projection + two sales tables (Level & Sum+Fund)
# Files required next to UL.py:
#   - Tables/COI.csv       (columns: PolicyYear, and e.g. YRT70-F30, YRT70-F30S, YRT85-M45, LEVEL-F30, ...)
#   - Tables/Tax_Rates.csv (Province + premium tax rate; flexible headers/units)
#
# I/O:
# - Prompts: Province → Client Type → Gender → Age → Smoker → Face Amount → COI Type → Premium Years → Base Rate → Safe Rate
# - Prints two tables:
#     Level Death Benefit (constant DB): same premium at Base/+1%/+2% where +1/+2% solve for Years
#     Sum + Fund (growing DB): keeps the same # of premium years, but shows how DB increases at +1% and +2%.
# - Writes annual CSVs:
#     LEVEL_Data.csv  : Year, Age, Deposit, Income, Cost, Fund Value, Face Value
#     LEVEL_Report.csv: same; if Client Type = Corporate adds a blank NCPI column (to be filled later)
#
# Notes:
# - Keeps prior math/formatting intact; improves structure & resiliency. No UL_Projection.csv written anymore.

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

# ---------- Small utilities ----------

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

def tax_equiv(irr_after_tax: float, tax_rate: float = 0.5331) -> float:
    return irr_after_tax / (1.0 - tax_rate) if (1.0 - tax_rate) > 0 else float("nan")

def paidup_age(start_age: int, years: float) -> int:
    return int(round(start_age + years))

def coi_limit_age(kind: str) -> int:
    return 70 if kind == "YRT70" else 85 if kind == "YRT85" else 100

def last_coi_months(start_age: int, kind: str) -> int:
    return max(0, coi_limit_age(kind) - start_age) * 12

# Robust IRR (bisection) for numerical stability incl. edge cases
def irr(cashflows: List[float], lo: float = -0.9999, hi: float = 4.0, tol: float = 1e-7, max_iter: int = 200) -> float:
    has_pos = any(cf > 0 for cf in cashflows)
    has_neg = any(cf < 0 for cf in cashflows)
    if not (has_pos and has_neg):
        return float("nan")

    def npv(rate: float) -> float:
        if rate <= -0.999999:
            return float("inf")
        acc = cashflows[0]
        one_plus = 1.0 + rate
        df = 1.0
        for cf in cashflows[1:]:
            df *= one_plus
            acc += cf / df
        return acc

    f_lo, f_hi = npv(lo), npv(hi)
    expand = 0
    while f_lo * f_hi > 0 and expand < 10 and hi < 10.0:
        hi *= 1.5
        f_hi = npv(hi)
        expand += 1
    nudge = 0
    while f_lo * f_hi > 0 and nudge < 5:
        lo = lo + (0.1 if lo < -0.5 else 0.05)
        if lo >= -0.01:
            lo = -0.0099
        f_lo = npv(lo)
        nudge += 1
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

# ---------- Tiny three-dot animation ----------
_dot_stop_event: Optional[threading.Event] = None
_dot_thread: Optional[threading.Thread] = None

def _start_dots(text: str = "Calculating") -> None:
    """Show 'Calculating', 'Calculating.', 'Calculating..', 'Calculating...' loop."""
    global _dot_stop_event, _dot_thread
    if not supports_ansi():
        print(text, flush=True)
        return
    if _dot_thread and _dot_thread.is_alive():
        return
    _dot_stop_event = threading.Event()
    def run():
        dots = ["", ".", "..", "..."]
        i = 0
        while _dot_stop_event and not _dot_stop_event.is_set():
            sys.stdout.write(f"\r{text}{dots[i % 4]}")
            sys.stdout.flush()
            time.sleep(0.4)
            i += 1
    _dot_thread = threading.Thread(target=run, daemon=True)
    _dot_thread.start()

def _stop_dots() -> None:
    """Stop the animation and clear the line."""
    global _dot_stop_event, _dot_thread
    if _dot_stop_event:
        _dot_stop_event.set()
    if _dot_thread:
        _dot_thread.join()
    if supports_ansi():
        sys.stdout.write("\r" + " " * 40 + "\r")
        sys.stdout.flush()

# ---------- Load premium tax from Tables/Tax_Rates.csv ----------
def load_premium_tax(province_code: str, default: float = 0.033) -> float:
    """
    Returns the premium tax rate as a decimal (e.g., 0.033 for 3.3%).
    Flexible with column names/units in Tax_Rates.csv.

    Expected location: ./Tables/Tax_Rates.csv
    Recognized columns (any one is fine):
      - Province   (QC/ON/BC, case-insensitive; trims spaces)
      - PremiumTaxPercent  OR  Premium_Tax_Percent  OR  Premium_Tax_Rate  OR  PremiumTaxRate
        Values may be in percent (3.3) or decimal (0.033).
    """
    try:
        tax_path = Path(__file__).parent / "Tables" / "Tax_Rates.csv"
        if not tax_path.exists():
            return default

        df_tax = pd.read_csv(tax_path)
        df_tax.columns = [str(c).strip() for c in df_tax.columns]

        # Province column
        prov_col_candidates = [c for c in df_tax.columns if c.lower() in {"province", "prov"}]
        if not prov_col_candidates:
            return default
        prov_col = prov_col_candidates[0]
        df_tax[prov_col] = df_tax[prov_col].astype(str).str.strip().str.upper()

        # Rate column candidates
        rate_candidates = []
        for c in df_tax.columns:
            lc = c.lower()
            if any(k in lc for k in ["premiumtaxpercent", "premium_tax_percent", "premium_tax_rate", "premiumtaxrate"]):
                rate_candidates.append(c)
        if not rate_candidates:
            # Fall back to any column that looks numeric and NOT the province col
            rate_candidates = [c for c in df_tax.columns if c != prov_col]

        # Take the first numeric-looking candidate
        rate_col = None
        for c in rate_candidates:
            try:
                vals = pd.to_numeric(df_tax[c], errors="coerce")
                if vals.notna().mean() > 0.7:
                    rate_col = c
                    df_tax[c] = vals
                    break
            except Exception:
                continue
        if rate_col is None:
            return default

        row = df_tax.loc[df_tax[prov_col] == province_code]
        if row.empty:
            return default

        val = float(row.iloc[0][rate_col])

        # If value looks like a percent (e.g., 3.3), convert to decimal
        if val > 1.0:
            val = val / 100.0

        # Sanity clamp
        if not (0.0 <= val < 0.25):
            return default

        return val
    except Exception:
        return default

# ---------- Core ----------
def main() -> None:
    print("\n--- Please Enter Policy Details ---\n")

    def ask_int(prompt: str) -> int:   return int(input(prompt).strip())
    def ask_float(prompt: str) -> float: return float(input(prompt).strip())

    prov_choice   = ask_int("Province (1=QC, 2=ON): ")
    client_choice = ask_int("Ownership (1=Corp, 2=Personal): ")
    gender_choice = ask_int("Gender (1=Male, 2=Female): ")
    age_input     = ask_int("Issue Age (e.g., 30): ")
    smoker_choice = ask_int("Smoker? (1=Non-Smoker, 2=Smoker): ")

    face_amount   = ask_float("Face Amount (e.g., 10000000): ")
    coi_choice    = ask_int("COI Type (1=YRT70, 2=YRT85, 3=L100): ")
    prem_years    = ask_int("Planned Premium Years (e.g., 20): ")
    base_rate_in  = ask_float("Base Annual Rate (e.g., 4.25): ")
    safe_rate_in  = ask_float("Guaranteed Rate (e.g., 2.00): ")

    # Derived selections
    prov_map = {1: "QC", 2: "ON"}
    province = prov_map.get(prov_choice, "QC")

    client_type = "Corporate" if client_choice == 1 else "Personal"

    gender      = "M" if gender_choice == 1 else "F"
    smk_suf     = "S" if smoker_choice == 2 else ""
    smk_display = "SM" if smoker_choice == 2 else "NS"

    start_age = age_input
    coi_map   = {1: "YRT70", 2: "YRT85", 3: "LEVEL"}
    coi_type  = coi_map.get(coi_choice, "LEVEL")
    rate_base = base_rate_in / 100.0
    rate_safe = safe_rate_in / 100.0

    # Premium tax
    premium_tax_rate = load_premium_tax(province)

    # Load COI.csv
    status("Calculating ...")
    script_dir = os.path.dirname(__file__)
    coi_file_path = Path(script_dir) / "Tables" / "COI.csv"
    try:
        coi_df = pd.read_csv(coi_file_path)
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

    # Build COI key per smoker & age
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
    print(f"\nProvince: {province} | Premium Tax: {premium_tax_rate*100:.2f}% | Client: {client_type}")
    print(f"COI Column: {coi_key}")

    # Start three-dot animation here (slow section)
    _start_dots("Calculating")

    # Precompute constants for speed
    months_total = (100 - start_age) * 12
    last_m = last_coi_months(start_age, coi_type)
    monthly_rate_cache = {}

    def get_monthly_rate(annual_rate: float) -> float:
        if annual_rate not in monthly_rate_cache:
            monthly_rate_cache[annual_rate] = (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0
        return monthly_rate_cache[annual_rate]

    # COI lookup
    def get_coi_rate_for_year(policy_year: int) -> float:
        row = coi_df.loc[coi_df["PolicyYear"] == policy_year]
        return float((row[coi_key].iloc[0] if not row.empty else coi_df[coi_key].iloc[-1]))

    # -------- Simulation engine (monthly) --------
    def simulate_monthly(annual_prem: float, prem_years_int: int, annual_rate: float, db_option: int) -> Tuple[pd.DataFrame, int]:
        """
        Detailed monthly projection with TAX column.
        Columns:
        Year, Month, FV_BOY, Deposit, TAX, FV1, FaceValue, NAAR, COI_Rate, COI, FV_Net, AnnualRate, Growth, FV_EOY
        """
        rows = []
        fv_boy = 0.0
        m_rate = get_monthly_rate(annual_rate)

        for m in range(1, months_total + 1):
            py  = (m - 1) // 12 + 1
            moy = (m - 1) % 12 + 1

            deposit = annual_prem if (moy == 1 and py <= prem_years_int) else 0.0
            tax_amt = deposit * premium_tax_rate
            fv1 = (fv_boy + deposit) - tax_amt

            if db_option == 1:
                naar = max(0.0, face_amount - fv1)   # Level DB
            else:
                naar = face_amount                    # Sum + Fund DB

            coi_rate_ann = get_coi_rate_for_year(py) if (m <= last_m and last_m > 0) else 0.0
            coi_month = (naar / 1000.0) * (coi_rate_ann / 12.0)

            fv_net = fv1 - coi_month
            growth = fv_net * m_rate
            fv_eoy = fv_net + growth

            rows.append((
                py, moy,             # Year, Month
                fv_boy,              # FV_BOY
                deposit,             # Deposit
                tax_amt,             # TAX
                fv1,                 # FV1 (post-deposit, after tax)
                face_amount,         # FaceValue
                naar,                # NAAR
                coi_rate_ann,        # COI_Rate (annual per 1,000)
                coi_month,           # COI (monthly)
                fv_net,              # FV_Net
                annual_rate,         # AnnualRate (decimal)
                growth,              # Growth (monthly)
                fv_eoy               # FV_EOY
            ))
            fv_boy = fv_eoy

        df = pd.DataFrame(rows, columns=[
            "Year","Month","FV_BOY","Deposit","TAX","FV1",
            "FaceValue","NAAR","COI_Rate","COI","FV_Net","AnnualRate","Growth","FV_EOY"
        ])
        return df, last_m

    def fv_at_last_coi_fractional(annual_prem: float, years: float, annual_rate: float, db_option: int) -> float:
        whole = int(np.floor(years))
        frac = years - whole
        fv_boy = 0.0
        m_rate = get_monthly_rate(annual_rate)
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
            H = get_coi_rate_for_year(py) if (m <= last_m and last_m > 0) else 0.0
            I = (G / 1000.0) * (H / 12.0)
            J = E - I
            fv_boy = J * (1.0 + m_rate)

        return float(fv_boy if last_m > 0 else 0.0)

    def total_db_at_85(annual_prem: float, years: float, annual_rate: float, db_option: int) -> float:
        df, _ = simulate_monthly(annual_prem, int(np.floor(years + 1e-9)), annual_rate, db_option)
        idx = (85 - start_age) * 12
        fv85 = float(df.iloc[idx - 1]["FV_EOY"]) if idx > 0 else 0.0
        return float(face_amount if db_option == 1 else face_amount + fv85)

    def find_min_premium_for_years(years_int: int, annual_rate: float, db_option: int) -> int:
        low, high = 0.0, 1.0
        # Expand upper bound
        for _ in range(40):
            df, _ = simulate_monthly(high, years_int, annual_rate, db_option)
            ok = (last_m <= 0) or (float(df.iloc[last_m - 1]["FV_EOY"]) >= 0)
            if ok: break
            high *= 2.0
        # Bisection
        for _ in range(80):
            mid = (low + high) / 2.0
            df, _ = simulate_monthly(mid, years_int, annual_rate, db_option)
            ok = (last_m <= 0) or (float(df.iloc[last_m - 1]["FV_EOY"]) >= 0)
            if ok: high = mid
            else:  low = mid
            if abs(high - low) < 0.01: break
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

    rate_p1 = rate_base + 0.01
    rate_p2 = rate_base + 0.02

    def build_table(db_option: int):
        base_prem = find_min_premium_for_years(prem_years, rate_base, db_option)
        if db_option == 1:
            years_b = float(prem_years)
            years_1 = find_years_for_same_premium(base_prem, rate_p1, db_option)
            years_2 = find_years_for_same_premium(base_prem, rate_p2, db_option)
            years_disp_b = prem_years
            years_disp_1 = f"~{int(round(years_1))}"
            years_disp_2 = f"~{int(round(years_2))}"
        else:
            years_b = years_1 = years_2 = float(prem_years)
            years_disp_b = years_disp_1 = years_disp_2 = prem_years

        db85_b = total_db_at_85(base_prem, years_b, rate_base, db_option)
        db85_1 = total_db_at_85(base_prem, years_1, rate_p1, db_option)
        db85_2 = total_db_at_85(base_prem, years_2, rate_p2, db_option)

        irr_b = irr_to_85(base_prem, years_b, db85_b); irr_te_b = tax_equiv(irr_b)
        irr_1 = irr_to_85(base_prem, years_1, db85_1); irr_te_1 = tax_equiv(irr_1)
        irr_2 = irr_to_85(base_prem, years_2, db85_2); irr_te_2 = tax_equiv(irr_2)

        safe_prem_ref = find_min_premium_for_years(prem_years, rate_safe, db_option) if db_option == 1 else None

        rows = [
            (base_prem, rate_base*100, years_disp_b, paidup_age(start_age, years_b), db85_b, irr_b, irr_te_b),
            (base_prem, rate_p1*100,   years_disp_1, paidup_age(start_age, years_1), db85_1, irr_1, irr_te_1),
            (base_prem, rate_p2*100,   years_disp_2, paidup_age(start_age, years_2), db85_2, irr_2, irr_te_2),
        ]
        return rows, safe_prem_ref, base_prem

    # -------- pretty, fixed-width table printer --------
    def print_table(title: str, rows: List[Tuple]) -> None:
        print("\n" + title)

        cols = [
            ("Premium",        10),
            ("Rate",            7),
            ("Years",           7),
            ("Paid-Up",        10),
            ("DB @ 85",        12),
            ("Tax-Free IRR",   13),
            ("Pre-Tax IRR",    13),
        ]

        header = " | ".join(name.ljust(w) for name, w in cols)
        sep    = "-+-".join("-" * w for _, w in cols)
        print(header)
        print(sep)

        for prem, rate_pct, yrs, pu_age, db85, irr_tf, irr_te in rows:
            cells = [
                f"{abbrev_money(prem):<{cols[0][1]}}",
                f"{rate_pct:>{cols[1][1]-1}.2f}%",
                f"{str(yrs):>{cols[2][1]}}",
                f"{pu_age:>{cols[3][1]}}",
                f"{abbrev_money(db85):>{cols[4][1]}}",
                f"{irr_tf*100:>{cols[5][1]-1}.2f}%",
                f"{irr_te*100:>{cols[6][1]-1}.2f}%",
            ]
            print(" | ".join(cells))

    # Heavy work (ensure dots always stop)
    try:
        # LEVEL table + monthly baseline
        rows_level, safe_prem_level, base_prem_level = build_table(db_option=1)
        df_level_base, _ = simulate_monthly(base_prem_level, prem_years, rate_base, db_option=1)

        # -------- Annual CSVs from monthly LEVEL base-case --------
        ann = df_level_base.copy()
        # Ensure numeric types
        to_num = ["Year", "Deposit", "TAX", "COI", "NAAR", "COI_Rate", "Growth", "FV_EOY"]
        for c in to_num:
            ann[c] = pd.to_numeric(ann[c], errors="coerce").fillna(0.0)

        # Annual aggregates
        sums = ann.groupby("Year", as_index=False)[["Deposit", "TAX", "COI", "Growth"]].sum()
        last = ann.groupby("Year", as_index=False)[["FV_EOY"]].last()

        annual = sums.merge(last, on="Year", how="left")
        annual["Age"]        = (start_age + annual["Year"] - 1).astype(int)
        annual["Cost"]       = (annual["COI"] + annual["TAX"]).astype(float)
        annual["Fund Value"] = annual["FV_EOY"].astype(float)
        annual["Face Value"] = float(face_amount)
        annual["Deposit"]    = annual["Deposit"].astype(float)
        annual["Income"]     = annual["Growth"].astype(float)

        # Order & round for LEVEL_Data.csv
        out_data_cols = ["Year", "Age", "Deposit", "Income", "Cost", "Fund Value", "Face Value"]
        out_data = annual[out_data_cols].copy()
        out_data[["Deposit","Income","Cost","Fund Value","Face Value"]] = (
            out_data[["Deposit","Income","Cost","Fund Value","Face Value"]].round(2)
        )

        # Write LEVEL_Data.csv
        (Path(script_dir) / "LEVEL_Data.csv").write_text(out_data.to_csv(index=False))

        # LEVEL_Report.csv:
        #   - Personal: same as LEVEL_Data (no NCPI)
        #   - Corporate: same columns + empty NCPI column
        if client_type == "Corporate":
            out_report = out_data.copy()
            out_report["NCPI"] = ""  # blank for your teammate to fill
            cols_with_ncpi = out_data_cols + ["NCPI"]
            (Path(script_dir) / "LEVEL_Report.csv").write_text(out_report[cols_with_ncpi].to_csv(index=False))
        else:
            (Path(script_dir) / "LEVEL_Report.csv").write_text(out_data.to_csv(index=False))

        # FUND+ table for the console
        rows_sf, _, _ = build_table(db_option=2)

    except Exception as e:
        _stop_dots()
        print("\nERROR:", e)
        traceback.print_exc()
        sys.exit(1)
    finally:
        _stop_dots()

    # ---- Pretty console output ----
    head_level = (
        f"UL (LEVEL) / {('Male' if gender=='M' else 'Female')}{start_age} {smk_display} / {coi_type}\n"
        f"Annual Premium: ${base_prem_level:,.0f}"
    )
    head_sf = (
        f"UL (FUND+) / {('Male' if gender=='M' else 'Female')}{start_age} {smk_display} / {coi_type}\n"
        f"Annual Premium: ${rows_sf[0][0]:,.0f}"
    )

    print_table(head_level, rows_level)
    print(f"\nFor reference (Level): a guaranteed return of {safe_rate_in:.2f}% for {prem_years} premiums would require an annual premium of {abbrev_money(safe_prem_level)}.")
    print_table(head_sf, rows_sf)

    print(f"\nProvince = {'Quebec' if province=='QC' else 'Ontario' if province=='ON' else 'British Columbia'} | Client = {client_type}")
    print("\n(Annual CSVs saved: LEVEL_Data.csv and LEVEL_Report.csv)\n")

if __name__ == "__main__":
    main()
