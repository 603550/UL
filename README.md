# Universal Life Projection Model

This Python tool models **Universal Life (UL)** insurance projections using actual COI (Cost of Insurance) tables and provincial tax data.

It generates:
- `LEVEL_Data.csv` — detailed monthly projection  
- `LEVEL_Report.csv` — annual client summary  

## How It Works
1. Prompts for:
   - Province (QC, ON, BC)
   - Gender, Age, Smoker status
   - Face amount
   - COI type (YRT70, YRT85, LEVEL100)
   - Premium years
   - Base and guaranteed rates
2. Computes:
   - Fund growth, deposits, taxes, and cost of insurance
3. Outputs:
   - Annualized report with `Deposit`, `Income`, `Cost`, `Fund Value`, and `Face Value`.

## Requirements
Python 3.10+  
Required libraries:
```bash
pip install numpy pandas
