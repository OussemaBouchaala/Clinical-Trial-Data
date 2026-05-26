"""
Step 5: Clean and harmonize disease/condition names.
PURPOSE: Standardize condition names by:
  1. Expanding common abbreviations (NSCLC → Non-Small Cell Lung Cancer)
  2. Normalizing casing with medical abbreviation preservation
  3. Stripping plurals (Diseases → Disease)
  4. British → American spelling (Leukaemia → Leukemia)
  5. Fuzzy matching for near-duplicates (threshold 95)

APPROACH:
  - Split multi-condition cells by "|" separator
  - Clean each condition individually
  - Abbreviation dictionary for ~45 common medical abbreviations
  - Safe fuzzy matching (excludes stage/grade mismatches and opposites)
  - Remove duplicates within each cell after cleaning
"""

import re
import json
import pandas as pd
from collections import Counter
from thefuzz import fuzz


# --- ABBREVIATION → FULL CANONICAL NAME ---
CONDITION_ABBREVIATIONS = {
    'nsclc': 'Non-Small Cell Lung Cancer',
    'sclc': 'Small Cell Lung Cancer',
    'aml': 'Acute Myeloid Leukemia',
    'all': 'Acute Lymphoblastic Leukemia',
    'cll': 'Chronic Lymphocytic Leukemia',
    'nhl': 'Non-Hodgkin Lymphoma',
    'dlbcl': 'Diffuse Large B-Cell Lymphoma',
    'adhd': 'Attention Deficit Hyperactivity Disorder',
    'copd': 'Chronic Obstructive Pulmonary Disease',
    'ckd': 'Chronic Kidney Disease',
    'hcc': 'Hepatocellular Carcinoma',
    'rcc': 'Renal Cell Carcinoma',
    'gist': 'Gastrointestinal Stromal Tumor',
    'mds': 'Myelodysplastic Syndrome',
    'tnbc': 'Triple-Negative Breast Cancer',
    'gerd': 'Gastroesophageal Reflux Disease',
    'ibd': 'Inflammatory Bowel Disease',
    'ibs': 'Irritable Bowel Syndrome',
    'ptsd': 'Post-Traumatic Stress Disorder',
    'ocd': 'Obsessive-Compulsive Disorder',
    'ms': 'Multiple Sclerosis',
    'ra': 'Rheumatoid Arthritis',
    'sle': 'Systemic Lupus Erythematosus',
    'osa': 'Obstructive Sleep Apnea',
    'af': 'Atrial Fibrillation',
    'mi': 'Myocardial Infarction',
    'ards': 'Acute Respiratory Distress Syndrome',
    'nafld': 'Non-Alcoholic Fatty Liver Disease',
    'nash': 'Non-Alcoholic Steatohepatitis',
    'pcos': 'Polycystic Ovary Syndrome',
    'bph': 'Benign Prostatic Hyperplasia',
    'uti': 'Urinary Tract Infection',
    'dvt': 'Deep Vein Thrombosis',
    'pe': 'Pulmonary Embolism',
    'aki': 'Acute Kidney Injury',
    'tbi': 'Traumatic Brain Injury',
    'als': 'Amyotrophic Lateral Sclerosis',
    'hiv': 'HIV Infection',
    'hcv': 'Hepatitis C',
    'hbv': 'Hepatitis B',
    't2dm': 'Type 2 Diabetes Mellitus',
    't1dm': 'Type 1 Diabetes Mellitus',
    'ad': 'Alzheimer Disease',
    'pd': 'Parkinson Disease',
    'cf': 'Cystic Fibrosis',
    'uc': 'Ulcerative Colitis',
    'cd': 'Crohn Disease',
    'adpkd': 'Autosomal Dominant Polycystic Kidney Disease',
}

# --- KNOWN MEDICAL ABBREVIATIONS (keep uppercase) ---
KNOWN_ABBREVIATIONS = {
    'HIV', 'AIDS', 'COPD', 'NSCLC', 'SCLC', 'AML', 'ALL', 'CLL',
    'NHL', 'DLBCL', 'ADHD', 'CKD', 'HCC', 'RCC', 'GIST', 'MDS',
    'TNBC', 'GERD', 'IBD', 'IBS', 'PTSD', 'OCD', 'MS', 'RA',
    'SLE', 'OSA', 'AF', 'MI', 'ARDS', 'NAFLD', 'NASH', 'PCOS',
    'BPH', 'UTI', 'DVT', 'PE', 'AKI', 'TBI', 'ALS', 'HCV', 'HBV',
    'ACL', 'ICU', 'CAR', 'ALK', 'EGFR', 'BRCA', 'PD', 'AD',
    'ADPKD', 'EBV', 'HPV', 'CMV', 'RSV', 'TB', 'MRSA', 'GI',
    'CNS', 'CVD', 'PAD', 'VTE', 'HER2', 'BCG', 'IV', 'II', 'III',
}


# ===================== CLEANING FUNCTIONS =====================

def smart_title_case_condition(text):
    """Title case but preserve known medical abbreviations."""
    words = text.split()
    result = []
    for word in words:
        if word.upper() in KNOWN_ABBREVIATIONS:
            result.append(word.upper())
        elif word.isupper() and len(word) > 1:
            result.append(word.title())
        else:
            result.append(word[0].upper() + word[1:] if word else word)
    return ' '.join(result)


def clean_condition(name):
    """Clean a single condition name."""
    if pd.isna(name) or not name.strip():
        return name

    cleaned = name.strip()

    # Step 1: Check pure abbreviation
    lower = cleaned.lower().strip()
    if lower in CONDITION_ABBREVIATIONS:
        return CONDITION_ABBREVIATIONS[lower]

    # Step 2: "ABBR - Full Name" pattern
    match = re.match(r'^([A-Z]{2,10})\s*[-–]\s*(.+)$', cleaned)
    if match:
        abbr = match.group(1).lower()
        if abbr in CONDITION_ABBREVIATIONS:
            return CONDITION_ABBREVIATIONS[abbr]

    # Step 3: "ABBR (Full Name)" pattern
    match = re.match(r'^([A-Z]{2,10})\s*\((.+)\)\s*$', cleaned)
    if match:
        abbr = match.group(1).lower()
        if abbr in CONDITION_ABBREVIATIONS:
            return CONDITION_ABBREVIATIONS[abbr]

    # Step 4: Remove trailing parenthetical abbreviation
    cleaned = re.sub(r'\s*\([A-Z]{2,10}\)\s*$', '', cleaned).strip()

    # Step 5: Smart Title Case
    cleaned = smart_title_case_condition(cleaned)

    # Step 6: Strip plurals
    cleaned = re.sub(
        r'(Disease|Disorder|Syndrome|Infection|Neoplasm|Fracture|Tumor|Ulcer)s\b',
        r'\1', cleaned
    )

    # Step 7: British → American spelling
    cleaned = cleaned.replace('Leukaemia', 'Leukemia')
    cleaned = cleaned.replace('Tumour', 'Tumor')
    cleaned = cleaned.replace('Haemophilia', 'Hemophilia')
    cleaned = cleaned.replace('Anaemia', 'Anemia')
    cleaned = cleaned.replace('Oesophageal', 'Esophageal')

    return cleaned


# ===================== FUZZY MATCHING =====================

def is_safe_fuzzy_match(rare, freq):
    """Check if a fuzzy match is safe (no stage/grade/opposite mismatch)."""
    rare_lower = rare.lower()
    freq_lower = freq.lower()

    # Stage mismatch check
    if 'stage' in rare_lower and 'stage' in freq_lower:
        rare_stage = re.findall(r'stage\s*(\w+)', rare_lower)
        freq_stage = re.findall(r'stage\s*(\w+)', freq_lower)
        if rare_stage and freq_stage and rare_stage[0] != freq_stage[0]:
            return False

    # Grade mismatch check
    if 'grade' in rare_lower and 'grade' in freq_lower:
        rare_grade = re.findall(r'grade\s*(\w+)', rare_lower)
        freq_grade = re.findall(r'grade\s*(\w+)', freq_lower)
        if rare_grade and freq_grade and rare_grade[0] != freq_grade[0]:
            return False

    # Opposite meaning check
    opposite_pairs = [
        ('stable', 'unstable'), ('pre', 'post'),
        ('acute', 'chronic'), ('early', 'late'),
        ('primary', 'secondary'), ('benign', 'malignant'),
    ]
    for a, b in opposite_pairs:
        if (a in rare_lower and b in freq_lower) or (b in rare_lower and a in freq_lower):
            return False

    # Token sort check
    if fuzz.token_sort_ratio(rare_lower, freq_lower) < 90:
        return False

    return True


# Known bad fuzzy matches to exclude
BAD_CONDITION_MATCHES = {
    'Prehabilitation',
    'B Acute Lymphoblastic Leukemia',
    'Metastatic Lung Small Cell Carcinoma',
    'Mental Health Wellness',
    'T-Prolymphocytic Leukemia',
    'Stage II Rectal Cancer Ajcc V7',
    'Stage III Rectal Cancer Ajcc V7',
    'Stage IV Colorectal Cancer Ajcc V7',
    'Prediabetes',
    'Caregiver',
    'Drug Interaction',
    'Mobile Application',
}


def build_condition_map(df):
    """Build fuzzy matching map for conditions."""
    # Count frequency of each cleaned individual condition
    cond_freq = Counter()
    for cell in df['condition_cleaned'].dropna():
        for c in cell.split('|'):
            cond_freq[c.strip()] += 1

    unique_conds = sorted(cond_freq.keys(), key=lambda x: -cond_freq[x])
    frequent_conds = [c for c in unique_conds if cond_freq[c] >= 2]
    rare_conds = [c for c in unique_conds if cond_freq[c] == 1]

    condition_map = {}
    for rare in rare_conds:
        if rare in BAD_CONDITION_MATCHES:
            continue
        best_score = 0
        best_match = None
        for freq in frequent_conds:
            score = fuzz.ratio(rare.lower(), freq.lower())
            if score > best_score and score >= 95:
                if is_safe_fuzzy_match(rare, freq):
                    best_score = score
                    best_match = freq
        if best_match:
            condition_map[rare] = best_match

    return condition_map


# ===================== APPLY TO DATAFRAME =====================

def clean_condition_cell(cell, condition_map):
    """Full pipeline: clean each condition, apply fuzzy map, deduplicate."""
    if pd.isna(cell):
        return cell

    conditions = cell.split('|')
    cleaned = []
    for c in conditions:
        name = clean_condition(c.strip())
        if name in condition_map:
            name = condition_map[name]
        cleaned.append(name)

    # Remove duplicates keeping order
    seen = set()
    unique = []
    for c in cleaned:
        lower = c.lower()
        if lower not in seen:
            seen.add(lower)
            unique.append(c)

    return ' | '.join(unique)


def harmonize_conditions(df):
    """Full condition harmonization pipeline."""
    # Step 1: Basic cleaning
    df['condition_cleaned'] = df['condition'].apply(
        lambda x: clean_condition_cell(x, {}) if pd.notna(x) else x
    )

    # Step 2: Build fuzzy map from cleaned data
    condition_map = build_condition_map(df)

    # Step 3: Re-apply with fuzzy map
    df['condition_cleaned'] = df['condition'].apply(
        lambda x: clean_condition_cell(x, condition_map) if pd.notna(x) else x
    )

    # Stats
    before_conds = set()
    for cell in df['condition'].dropna():
        for c in cell.split('|'):
            before_conds.add(c.strip())

    after_conds = set()
    for cell in df['condition_cleaned'].dropna():
        for c in cell.split('|'):
            after_conds.add(c.strip())

    print(f"Condition - Individual before: {len(before_conds)}")
    print(f"Condition - Individual after:  {len(after_conds)}")
    print(f"Condition - Reduced by: {len(before_conds) - len(after_conds)}")
    print(f"Condition - Cells before: {df['condition'].nunique()}")
    print(f"Condition - Cells after:  {df['condition_cleaned'].nunique()}")
    print(f"Condition - Fuzzy matches: {len(condition_map)}")

    return df, condition_map


if __name__ == "__main__":
    df = pd.read_csv("data/processed/trials_drugs_cleaned.csv")
    df, condition_map = harmonize_conditions(df)

    # Save disease dictionary (abbreviation + fuzzy maps)
    disease_dict = {**{k: v for k, v in CONDITION_ABBREVIATIONS.items()}, **condition_map}
    with open("data/dictionaries/disease_dictionary.json", "w") as f:
        json.dump(disease_dict, f, indent=2)

    df.to_csv("data/processed/trials_conditions_cleaned.csv", index=False)
    print("Saved to data/processed/")