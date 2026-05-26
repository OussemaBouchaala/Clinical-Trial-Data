"""
Step 7: Quality metrics report.
PURPOSE: Compare before vs after harmonization across all fields.
Outputs a summary to console and saves to outputs/quality_report.txt.
"""

import pandas as pd
import json

# Load data
raw = pd.read_csv("data/raw/trials_sample.csv")
cleaned = pd.read_csv("data/processed/trials_cleaned.csv")

# Load dictionaries
with open("data/dictionaries/drug_dictionary.json", "r") as f:
    drug_dict = json.load(f)
with open("data/dictionaries/disease_dictionary.json", "r") as f:
    disease_dict = json.load(f)

report = []

def log(text=""):
    print(text)
    report.append(text)

log("=" * 60)
log("QUALITY METRICS REPORT")
log("=" * 60)

# --- Country ---
log("\n--- COUNTRY ---")
before = raw['country'].nunique()
after = cleaned['country_cleaned'].nunique()
individual_before = set()
for cell in raw['country'].dropna():
    for c in cell.split(","):
        individual_before.add(c.strip())
individual_after = set()
for cell in cleaned['country_cleaned'].dropna():
    for c in cell.split(","):
        individual_after.add(c.strip())
log(f"Unique combinations: {before} → {after} (reduced by {before - after})")
log(f"Individual countries: {len(individual_before)} → {len(individual_after)}")
log(f"Method: pycountry + manual pre/post-fix dictionaries")

# --- Sponsor ---
log("\n--- SPONSOR ---")
before = raw['sponsor'].nunique()
after = cleaned['sponsor_cleaned'].nunique()
log(f"Unique sponsors: {before} → {after} (reduced by {before - after})")
log(f"Method: suffix standardization + core extraction + fuzzy matching (pharma only)")

# --- Drug / Intervention ---
log("\n--- DRUG / INTERVENTION ---")
before = raw['intervention_name'].nunique()
after = cleaned['intervention_cleaned'].nunique()
brand_to_generic = sum(1 for k, v in drug_dict.items() if k.lower() != v.lower())
casing_fixes = sum(1 for k, v in drug_dict.items() if k.lower() == v.lower())
log(f"Unique values: {before} → {after} (reduced by {before - after})")
log(f"Dictionary: {len(drug_dict)} entries ({brand_to_generic} brand→generic, {casing_fixes} casing)")
log(f"Method: regex cleaning + RxNorm API (NIH)")

# --- Disease / Condition ---
log("\n--- DISEASE / CONDITION ---")
before_cells = raw['condition'].nunique()
after_cells = cleaned['condition_cleaned'].nunique()
before_individual = set()
for cell in raw['condition'].dropna():
    for c in cell.split("|"):
        before_individual.add(c.strip())
after_individual = set()
for cell in cleaned['condition_cleaned'].dropna():
    for c in cell.split("|"):
        after_individual.add(c.strip())
log(f"Unique cells: {before_cells} → {after_cells} (reduced by {before_cells - after_cells})")
log(f"Individual conditions: {len(before_individual)} → {len(after_individual)} (reduced by {len(before_individual) - len(after_individual)})")
log(f"Dictionary: {len(disease_dict)} entries")
log(f"Method: abbreviation expansion + casing + plurals + British→American + fuzzy matching")

# --- Biomarker ---
log("\n--- BIOMARKER EXTRACTION ---")
if 'biomarkers' in cleaned.columns:
    total = len(cleaned)
    has_biomarkers = cleaned['biomarkers'].apply(
        lambda x: len(eval(x)) > 0 if pd.notna(x) and x != '[]' else False
    ).sum()
    empty = total - has_biomarkers
    log(f"Total trials: {total}")
    log(f"Trials with biomarkers extracted: {has_biomarkers} ({has_biomarkers/total*100:.1f}%)")
    log(f"Trials with no biomarkers: {empty} ({empty/total*100:.1f}%)")
else:
    log(f"Biomarker column not present — extraction incomplete due to Kaggle crash.")
    log(f"Code is functional (tested: ~6% parse failure rate, ~7.9 sec/trial on T4 GPU)")
    log(f"Estimated full run: ~5.5 hours on dual T4 GPU setup")

log(f"\nModel: Shubh-0789/biomarker-qwen3.5-0.8b-lora-v2.1")
log(f"Libraries: transformers 5.2.0 + peft")
log(f"Hardware: Kaggle GPU T4 x2 (16GB VRAM each)")
log(f"Batching: single-trial loop (batch tokenization broken in transformers 5.2)")
log(f"Decoding: greedy (do_sample=False)")

log("\n" + "=" * 60)
log("END OF REPORT")
log("=" * 60)

# Save to file
with open("outputs/quality_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\nSaved to outputs/quality_report.txt")