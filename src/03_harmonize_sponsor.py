"""
Step 3: Clean and harmonize sponsor names.
PURPOSE: Standardize corporate legal suffixes (Inc, Ltd, LLC, Corp, etc.)
to one canonical spelling, then group variations of the same company
using core name extraction + fuzzy matching (pharma only).

APPROACH:
  1. Manual map for weird edge cases (subsidiary sentences)
  2. Standardize legal suffixes (Inc., Incorporated → Inc)
  3. Extract core name (strip suffixes + division words) for comparison
  4. Fuzzy match pharma/biotech cores (threshold 90)
     - Skip institutions (universities, hospitals) to avoid false merges
  5. Pick shortest name as canonical
"""

import re
import pandas as pd
from collections import defaultdict
from thefuzz import fuzz


# --- MANUAL MAP: weird edge cases ---
SPONSOR_MANUAL_MAP = {
    "Pfizer's Upjohn has merged with Mylan to form Viatris Inc.": "Viatris Inc",
    "Metsera, a wholly owned subsidiary of Pfizer": "Pfizer",
    "Wyeth is now a wholly owned subsidiary of Pfizer": "Pfizer",
}

# --- SUFFIXES FOR CORE EXTRACTION (comparison only) ---
SUFFIXES_TO_STRIP_FOR_CORE = [
    r'\bCo\s*Ltd$',
    r'\bInc$',
    r'\bLtd$',
    r'\bLLC$',
    r'\bCorp$',
    r'\bGmbH$',
    r'\bSpA$',
    r'\bBV$',
    r'\bNV$',
    r'\bSA$',
    r'\bSL$',
    r'\bSAS$',
    r'\bAG$',
    r'\bAB$',
    r'\bA/S$',
    r'\bPLC$',
    r'\bSE$',
    r'\bPLLC$',
    r'\band Company$',
    r'\b& Company$',
    r'\bPharmaceuticals?$',
    r'\bPharma$',
    r'\bVaccines?$',
    r'\bBiosciences?$',
    r'\bBiotech$',
    r'\bTherapeutics?$',
    r'\bBiopharma$',
    r'\bBiopharmaceuticals?$',
]

# --- INSTITUTION KEYWORDS (skip fuzzy matching for these) ---
INSTITUTION_KEYWORDS = [
    'university', 'hospital', 'clinic', 'institute', 'college',
    'medical center', 'cancer center', 'health system', 'school',
    'foundation', 'ministry', 'department', 'va ', 'veterans',
    'national', 'centre', 'center', 'nhs',
]


def standardize_sponsor_suffix(name):
    """Standardize corporate legal suffixes to one canonical spelling."""
    if pd.isna(name):
        return name

    cleaned = name.strip()

    # Co., Ltd variations → Co Ltd
    cleaned = re.sub(r'\bCo\.\s*,?\s*Ltd\.?', 'Co Ltd', cleaned)
    # Pty Ltd → Ltd
    cleaned = re.sub(r',?\s*Pty\s+Ltd\.?\s*$', ' Ltd', cleaned)
    # Incorporated → Inc
    cleaned = re.sub(r',?\s*Incorporated\s*$', ' Inc', cleaned)
    # Inc. / , Inc → Inc
    cleaned = re.sub(r',?\s*Inc\.?\s*$', ' Inc', cleaned)
    # Limited / LTD → Ltd
    cleaned = re.sub(r',?\s*(?:Limited|LTD)\.?\s*$', ' Ltd', cleaned)
    cleaned = re.sub(r',?\s*Ltd\.?\s*$', ' Ltd', cleaned)
    # Corporation / Corp. → Corp
    cleaned = re.sub(r',?\s*Corporation\s*$', ' Corp', cleaned)
    cleaned = re.sub(r',?\s*Corp\.?\s*$', ' Corp', cleaned)
    # LLC. → LLC
    cleaned = re.sub(r',?\s*LLC\.?\s*$', ' LLC', cleaned)
    # gGmbH → GmbH
    cleaned = re.sub(r'\bgGmbH\s*$', 'GmbH', cleaned)
    # S.p.A → SpA
    cleaned = re.sub(r',?\s*S\.?p\.?A\.?\s*$', ' SpA', cleaned)
    # B.V → BV
    cleaned = re.sub(r',?\s*B\.?V\.?\s*$', ' BV', cleaned)
    # N.V → NV
    cleaned = re.sub(r',?\s*N\.?V\.?\s*$', ' NV', cleaned)
    # S.A → SA
    cleaned = re.sub(r',?\s*S\.?A\.?\s*$', ' SA', cleaned)
    # S.L → SL
    cleaned = re.sub(r',?\s*S\.?L\.?\s*$', ' SL', cleaned)

    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def get_core_name(name):
    """Strip all suffixes to get core name for comparison only."""
    if pd.isna(name):
        return name

    core = name.strip()

    for _ in range(2):
        for suffix in SUFFIXES_TO_STRIP_FOR_CORE:
            core = re.sub(suffix, '', core, flags=re.IGNORECASE).strip()

    core = core.rstrip('.,;: ')
    core = re.sub(r'\s+', ' ', core).strip()
    return core


def is_institution(name):
    """Check if sponsor is a university/hospital/institution."""
    lower = name.lower()
    return any(kw in lower for kw in INSTITUTION_KEYWORDS)


def build_sponsor_map(df):
    """Build mapping of sponsor variations → canonical name."""
    unique_sponsors = df['sponsor_cleaned'].unique()
    sponsor_cores = {s: get_core_name(s) for s in unique_sponsors}

    # Group by core
    core_to_sponsors = defaultdict(list)
    for sponsor, core in sponsor_cores.items():
        core_to_sponsors[core.lower()].append(sponsor)

    # Split pharma vs institution
    pharma_cores = {}
    institution_cores = {}
    for core, sponsors in core_to_sponsors.items():
        if any(is_institution(s) for s in sponsors):
            institution_cores[core] = sponsors
        else:
            pharma_cores[core] = sponsors

    # Fuzzy match pharma cores only (threshold 90)
    pharma_core_list = list(pharma_cores.keys())
    merged_cores = {}
    for i, core_a in enumerate(pharma_core_list):
        if core_a in merged_cores:
            continue
        for j in range(i + 1, len(pharma_core_list)):
            core_b = pharma_core_list[j]
            if core_b in merged_cores:
                continue
            score = fuzz.token_sort_ratio(core_a, core_b)
            if score >= 90:
                merged_cores[core_b] = core_a

    # Build final groups
    final_groups = defaultdict(list)
    for core, sponsors in pharma_cores.items():
        leader = merged_cores.get(core, core)
        final_groups[leader].extend(sponsors)
    for core, sponsors in institution_cores.items():
        final_groups[core].extend(sponsors)

    # Pick shortest name as canonical
    sponsor_map = {}
    for leader, members in final_groups.items():
        if len(members) > 1:
            canonical = min(members, key=len)
            for m in members:
                if m != canonical:
                    sponsor_map[m] = canonical

    return sponsor_map


def clean_sponsor(name, sponsor_map):
    """Full pipeline: manual map → standardize suffix → group remap."""
    if pd.isna(name):
        return name

    trimmed = name.strip()

    # Step 1: Manual map
    if trimmed in SPONSOR_MANUAL_MAP:
        trimmed = SPONSOR_MANUAL_MAP[trimmed]

    # Step 2: Standardize suffixes
    cleaned = standardize_sponsor_suffix(trimmed)

    # Step 3: Group remap
    if cleaned in sponsor_map:
        cleaned = sponsor_map[cleaned]

    return cleaned


def harmonize_sponsor(df):
    """Apply sponsor cleaning to the dataframe."""
    # First pass: standardize suffixes
    df['sponsor_cleaned'] = df['sponsor'].apply(
        lambda x: standardize_sponsor_suffix(SPONSOR_MANUAL_MAP.get(x.strip(), x.strip()))
        if pd.notna(x) else x
    )

    # Build mapping from standardized names
    sponsor_map = build_sponsor_map(df)

    # Second pass: apply group remapping
    df['sponsor_cleaned'] = df['sponsor'].apply(lambda x: clean_sponsor(x, sponsor_map))

    before = df['sponsor'].nunique()
    after = df['sponsor_cleaned'].nunique()
    print(f"Sponsor - Before: {before} unique")
    print(f"Sponsor - After:  {after} unique")
    print(f"Sponsor - Reduced by: {before - after}")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/trials_country_cleaned.csv")
    df = harmonize_sponsor(df)
    df.to_csv("data/processed/trials_sponsor_cleaned.csv", index=False)
    print("Saved to data/processed/trials_sponsor_cleaned.csv")