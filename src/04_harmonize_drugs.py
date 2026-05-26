"""
Step 4: Clean and harmonize drug/intervention names.
PURPOSE: Standardize drug names by:
  1. Basic cleaning (strip dosage, pharmaceutical forms, normalize casing)
  2. RxNorm API lookup for INN-suffix drugs + common brands (brand → generic)
  3. Apply dictionary to map all variations to canonical generic names.

APPROACH:
  - Split multi-drug cells by "+" separator
  - Clean each drug individually
  - Only look up DRUG/BIOLOGICAL intervention types in RxNorm
  - Preserve research codes in UPPERCASE (e.g., ABT-089, MK-3475)
  - Keep combo cell structure (drug1 + drug2) — don't flatten
"""

import re
import time
import json
import pandas as pd
import requests


# --- KNOWN BAD RXNORM MAPPINGS (manually verified as incorrect) ---
BAD_MAPPINGS = {
    'Bisphosphonates Combined With Vancomycin',
    'I.V Of Normal Saline',
    'Normal Saline',
    'Preservative Free Normal Saline',
    'Briplatin',
    'CG-P5 Peptide',
    'CNTX-2022 (Lidocaine',
    'Consolidation (Carboplatin',
    'Dexmedetomidine-Ketamine-Lidocaine',
    'Acyclovir-Penciclovir',
    'Statin',
    'Non-Thiamine',
    'Radioactive Iodine',
    'Velcade Cyclophosphamide Prednisone',
    'Velcade Prednisone',
    'SHR2554; Adabelimumab',
    'Naproxen-Esomeprazole',
    'Lidocaine-Prilocaine',
    'Sibutramine-Clarithromycin',
    'QL1706 Combined With Lenvatinib',
    'Ritonavir Boosted Atazanavir',
    'Yttrium Y 90 Basiliximab',
    'Ketamine Hydrochloride In Saline',
    'YL242; Pembrolizumab',
    'FOLFIRI 1-Bevacizumab',
    'Statin Plus Fenofibrate',
}

# --- COMMON BRAND NAMES TO LOOK UP (no INN suffix) ---
COMMON_BRANDS = [
    'Keytruda', 'KEYTRUDA', 'Opdivo', 'Herceptin', 'Avastin',
    'Rituxan', 'Humira', 'Remicade', 'Tecentriq', 'Imfinzi',
    'Yervoy', 'Bavencio', 'Libtayo', 'Erbitux', 'Vectibix',
    'Perjeta', 'Kadcyla', 'Darzalex', 'Revlimid', 'Velcade',
    'Gleevec', 'Xeloda', 'Taxol', 'Taxotere', 'Alimta',
    'Eloxatin', 'Nexavar', 'Sutent', 'Votrient', 'Inlyta',
    'Stivarga', 'Lynparza', 'Ibrance', 'Verzenio', 'Kisqali',
    'Tagrisso', 'Calquence', 'Imbruvica', 'Zytiga', 'Xtandi',
    'Lupron', 'Zoladex', 'Firmagon', 'Provenge', 'Jakafi',
    'Cosentyx', 'Stelara', 'Dupixent', 'Xolair', 'Nucala',
    'Entresto', 'Jardiance', 'Ozempic', 'Trulicity', 'Victoza',
    'Bridion', 'Aspirin', 'Tylenol', 'Advil', 'Motrin',
]

# --- INN DRUG SUFFIXES (International Nonproprietary Names) ---
DRUG_SUFFIXES = [
    'mab', 'nib', 'lib', 'tib', 'cib',
    'zole', 'pine', 'tine', 'dine', 'line',
    'pril', 'artan', 'olol', 'alol',
    'statin', 'fibrate',
    'cillin', 'mycin', 'cycline',
    'vir', 'navir', 'covir', 'bivir',
    'prazole', 'tidine',
    'amine', 'etine', 'oxetine',
    'solone', 'nisone', 'sone',
    'platin', 'rubicin',
    'gliptin', 'flozin', 'glutide',
    'parin', 'xaban',
    'caine', 'barb',
    'azine', 'azole',
    'ximab', 'zumab', 'mumab', 'limab',
    'tinib', 'anib', 'enib',
    'tide', 'retin',
    'afil', 'lukast',
    'fen', 'profen',
    'olone', 'onide',
    'sartan',
]

# --- NON-DRUG WORDS (for filtering out non-drug names) ---
NON_DRUG_WORDS = [
    'catheter', 'device', 'block', 'surgery', 'procedure', 'stent',
    'implant', 'pump', 'monitor', 'mask', 'bandage', 'dressing',
    'exercise', 'training', 'therapy', 'counseling', 'education',
    'program', 'intervention', 'questionnaire', 'interview', 'app',
    'guided', 'assisted', 'based', 'conventional', 'standard',
    'system', 'technique', 'method', 'protocol', 'regimen',
    'diet', 'meal', 'food', 'supplement', 'vitamin',
    'sham', 'control', 'placebo', 'usual care', 'no intervention',
    'needle', 'suture', 'graft', 'prosthesis', 'scaffold',
]


# ===================== CLEANING FUNCTIONS =====================

def smart_title_case(text, original):
    """Title case but preserve research codes based on original casing.
    If original word was ALL UPPERCASE → keep uppercase (likely a code).
    Otherwise → Title Case."""
    words = text.split()
    orig_words = original.split()

    result = []
    for i, word in enumerate(words):
        orig = orig_words[i] if i < len(orig_words) else word
        if orig.replace('-', '').replace('.', '').isupper() and any(c.isalpha() for c in orig):
            result.append(word.upper())
        else:
            result.append(word.title())

    return ' '.join(result)


def clean_drug_name(name):
    """Clean a single drug name: strip dosage, forms, normalize casing."""
    if pd.isna(name) or not name.strip():
        return name

    original = name.strip()
    cleaned = original

    # Remove dosage patterns (10 mg, 500mg, 1.5 mcg/kg, 0.9%)
    cleaned = re.sub(r'\d+\.?\d*\s*(?:mg|mcg|ug|ml|g\b|%|MG|ML|mcg/kg)\S*', '', cleaned)

    # Remove pharmaceutical forms
    cleaned = re.sub(
        r'\b(?:Injection|Capsule|Capsules|Tablet|Tablets|Solution|'
        r'Cream|Gel|Ointment|Syrup|Suspension|Powder)\b',
        '', cleaned, flags=re.IGNORECASE
    )

    # Remove frequency words
    cleaned = re.sub(
        r'\b(?:daily|weekly|once|twice|per day|per week|QD|BID|TID|QID)\b',
        '', cleaned, flags=re.IGNORECASE
    )

    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = cleaned.strip('.,;:- ')

    if not cleaned:
        return original

    # Smart casing: preserve research codes, title case the rest
    cleaned = smart_title_case(cleaned, original)

    return cleaned


def clean_drug_cell(cell):
    """Clean a cell containing one or more drugs separated by +."""
    if pd.isna(cell):
        return cell

    drugs = cell.split('+')
    cleaned = [clean_drug_name(d.strip()) for d in drugs]
    cleaned = [d for d in cleaned if d.strip()]

    return ' + '.join(cleaned)


# ===================== RXNORM FUNCTIONS =====================

def rxnorm_to_generic(drug_name):
    """Look up a drug in RxNorm and return its generic/ingredient name."""
    url = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    params = {"term": drug_name, "maxEntries": 1}

    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        candidates = data.get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return None

        rxcui = candidates[0].get("rxcui")
        if not rxcui:
            return None

        # Get related ingredients (generic names)
        rel_url = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/related.json"
        rel_params = {"tty": "IN"}
        rel_resp = requests.get(rel_url, params=rel_params, timeout=10)
        rel_data = rel_resp.json()

        groups = rel_data.get("relatedGroup", {}).get("conceptGroup", [])
        for group in groups:
            props = group.get("conceptProperties", [])
            if props:
                return props[0].get("name")

        return candidates[0].get("name")

    except Exception:
        return None


def has_drug_suffix(name):
    """Check if name ends with a known INN drug suffix."""
    lower = name.lower().strip()
    return any(lower.endswith(s) for s in DRUG_SUFFIXES)


def is_drug_candidate(name):
    """Check if name is worth looking up in RxNorm."""
    if not name or len(name) < 3 or len(name) > 40:
        return False
    if not name[0].isalpha():
        return False
    if len(name.split()) > 4:
        return False
    if any(c in name for c in '®™©'):
        return False
    lower = name.lower()
    for word in NON_DRUG_WORDS:
        if word in lower:
            return False
    return True


# ===================== DICTIONARY BUILDING =====================

def build_drug_dictionary(df):
    """Build drug dictionary by looking up INN-suffix drugs + brands in RxNorm."""

    # Collect unique individual drug names from DRUG/BIOLOGICAL rows
    drug_mask = df['intervention_type'].str.contains('DRUG|BIOLOGICAL', na=False)
    all_individual_drugs = set()
    for cell in df[drug_mask]['intervention_cleaned'].dropna():
        for drug in cell.split('+'):
            name = drug.strip()
            if name and is_drug_candidate(name):
                all_individual_drugs.add(name)

    # Filter to INN-suffix names
    inn_candidates = [d for d in all_individual_drugs if has_drug_suffix(d)]

    # Add common brands found in data
    brands_in_data = [
        b for b in COMMON_BRANDS
        if any(b.lower() in d.lower() for d in all_individual_drugs)
    ]

    # Combine lookup list
    all_lookups = sorted(set(inn_candidates + brands_in_data))
    print(f"Looking up {len(all_lookups)} names in RxNorm...")

    # Run RxNorm lookups
    raw_dict = {}
    start = time.time()
    for i, name in enumerate(all_lookups):
        result = rxnorm_to_generic(name)
        if result and not result.startswith("ERROR"):
            raw_dict[name] = result.title()

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            remaining = (elapsed / (i + 1)) * (len(all_lookups) - i - 1)
            print(f"  {i+1}/{len(all_lookups)} | ~{remaining:.0f}s remaining")

    elapsed = time.time() - start
    print(f"Done in {elapsed/60:.1f} minutes. Found: {len(raw_dict)}")

    # Clean up dictionary: remove combos, bad mappings, simplify formulations
    clean_dict = {}
    for raw, generic in raw_dict.items():
        if any(sep in raw for sep in [' And ', ' Plus ', ', ', '/']):
            continue
        if raw in BAD_MAPPINGS:
            continue
        # Simplify bracket formulations
        if '[' in generic:
            generic = generic.split('[')[0].strip()
        if 'Injectable Solution' in generic:
            generic = generic.replace('Injectable Solution', '').strip()
        clean_dict[raw] = generic

    return clean_dict


# ===================== APPLY TO DATAFRAME =====================

def apply_drug_dictionary(cell, drug_dict):
    """Apply drug dictionary to a cell with multiple drugs."""
    if pd.isna(cell):
        return cell

    drugs = cell.split('+')
    cleaned = []
    for drug in drugs:
        name = drug.strip()
        if name in drug_dict:
            cleaned.append(drug_dict[name])
        else:
            cleaned.append(name)

    return ' + '.join(cleaned)


def harmonize_drugs(df):
    """Full drug harmonization pipeline."""
    before = df['intervention_name'].nunique()

    # Step 1: Basic cleaning (dosage, casing, forms)
    df['intervention_cleaned'] = df['intervention_name'].apply(clean_drug_cell)

    # Step 2: Build dictionary via RxNorm
    drug_dict = build_drug_dictionary(df)

    # Step 3: Apply dictionary
    df['intervention_cleaned'] = df['intervention_cleaned'].apply(
        lambda x: apply_drug_dictionary(x, drug_dict)
    )

    after = df['intervention_cleaned'].nunique()
    same = sum(1 for r, g in drug_dict.items() if r.lower() == g.lower())
    diff = sum(1 for r, g in drug_dict.items() if r.lower() != g.lower())

    print(f"Drug - Before: {before} unique")
    print(f"Drug - After:  {after} unique")
    print(f"Drug - Dictionary: {len(drug_dict)} entries ({diff} brand→generic, {same} casing fixes)")

    return df, drug_dict


if __name__ == "__main__":
    df = pd.read_csv("data/processed/trials_sponsor_cleaned.csv")
    df, drug_dict = harmonize_drugs(df)

    # Save dictionary
    with open("data/dictionaries/drug_dictionary.json", "w") as f:
        json.dump(drug_dict, f, indent=2)

    df.to_csv("data/processed/trials_drugs_cleaned.csv", index=False)
    print("Saved to data/processed/")