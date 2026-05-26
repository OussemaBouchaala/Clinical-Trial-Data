"""
Step 8: Stretch goal — Before vs After comparison.
PURPOSE: Demonstrate the real-world value of harmonization.
Show how many trials you find for a specific drug or disease
before cleaning vs after. This is WHY this work matters.
"""

import pandas as pd

# Load data
raw = pd.read_csv("data/raw/trials_sample.csv")
cleaned = pd.read_csv("data/processed/trials_cleaned.csv")

print("=" * 60)
print("STRETCH GOAL: Before vs After Harmonization")
print("=" * 60)

# ============================================================
# DISEASE EXAMPLE: Lung Cancer
# ============================================================
print("\n--- DISEASE: Lung Cancer ---")
print("Search term: 'lung cancer'\n")

# Before: exact search on raw data
search = "lung cancer"
before_hits = raw[raw['condition'].str.contains(search, case=False, na=False)]
print(f"Before harmonization: {len(before_hits)} trials found")

# What variants exist in raw data?
raw_variants = set()
for cell in raw['condition'].dropna():
    for c in cell.split("|"):
        if "lung" in c.lower() and "cancer" in c.lower():
            raw_variants.add(c.strip())
print(f"Raw variants found: {len(raw_variants)}")
for v in sorted(raw_variants):
    print(f"  - '{v}'")

# After: same search on cleaned data
after_hits = cleaned[cleaned['condition_cleaned'].str.contains(search, case=False, na=False)]
print(f"\nAfter harmonization: {len(after_hits)} trials found")

after_variants = set()
for cell in cleaned['condition_cleaned'].dropna():
    for c in cell.split("|"):
        if "lung" in c.lower() and "cancer" in c.lower():
            after_variants.add(c.strip())
print(f"Cleaned variants: {len(after_variants)}")
for v in sorted(after_variants):
    print(f"  - '{v}'")

gain = len(after_hits) - len(before_hits)
print(f"\n>> Net gain: {gain} additional trials surfaced")

# ============================================================
# DRUG EXAMPLE: Bevacizumab (Avastin)
# ============================================================
print("\n\n--- DRUG: Bevacizumab ---")
print("Search term: 'bevacizumab'\n")

before_hits = raw[raw['intervention_name'].str.contains("bevacizumab", case=False, na=False)]
print(f"Before harmonization: {len(before_hits)} trials found")

# Check if Avastin exists in raw
avastin_hits = raw[raw['intervention_name'].str.contains("avastin", case=False, na=False)]
print(f"Searching 'Avastin' separately: {len(avastin_hits)} trials")

after_hits = cleaned[cleaned['intervention_cleaned'].str.contains("bevacizumab", case=False, na=False)]
print(f"\nAfter harmonization (one search): {len(after_hits)} trials found")
print(f">> Avastin trials now included under 'Bevacizumab'")

# ============================================================
# COUNTRY EXAMPLE: Turkey
# ============================================================
print("\n\n--- COUNTRY: Turkey ---")
print("Search term: 'Turkey'\n")

before_hits = raw[raw['country'].str.contains(r'\bTurkey   \b', case=False, na=False)]
usa_hits = raw[raw['country'].str.contains(r'\b(Türkiye)\b', case=False, na=False)]
#us_hits = raw[raw['country'].str.contains(r'\bUS\b', case=False, na=False)]
print(f"Before: 'Turkey' = {len(before_hits)}, 'Turkey (Türkiye)' = {len(usa_hits)}")

after_hits = cleaned[cleaned['country_cleaned'].str.contains("Turkey", case=False, na=False)]
print(f"After harmonization (one search --> Turkey): {len(after_hits)} trials found")

print("\n" + "=" * 60)
print("CONCLUSION: Harmonization makes previously invisible trials")
print("discoverable under a single consistent search term.")
print("=" * 60)