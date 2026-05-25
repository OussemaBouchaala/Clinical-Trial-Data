"""
Step 2: Clean and harmonize country names.
PURPOSE: Standardize messy country names like "Turkey (Türkiye)", "USA",
"Brasil" into clean common names like "Turkey", "United States", "Brazil".
Uses pycountry library with manual pre-fixes and post-fixes on top.
"""

import pandas as pd
import pycountry

# PRE-FIX: names pycountry can't find at all
COUNTRY_PRE_FIX = {
    "turkey (türkiye)": "Turkey",
    "türkiye": "Turkey",
    "palestinian territories": "Palestine",
    "faroe islands": "Faroe Islands",
    "brasil": "Brazil",
}

# POST-FIX: pycountry returns ugly official names, we convert to common names
COUNTRY_POST_FIX = {
    "Iran, Islamic Republic of": "Iran",
    "Viet Nam": "Vietnam",
    "Russian Federation": "Russia",
    "Taiwan, Province of China": "Taiwan",
    "Tanzania, United Republic of": "Tanzania",
    "Syrian Arab Republic": "Syria",
    "Lao People's Democratic Republic": "Laos",
    "Venezuela, Bolivarian Republic of": "Venezuela",
    "Moldova, Republic of": "Moldova",
    "Réunion": "Reunion",
    "Korea, Republic of": "South Korea",
    "Korea, Democratic People's Republic of": "North Korea",
    "Congo, The Democratic Republic of the": "Democratic Republic of the Congo",
    "Bolivia, Plurinational State of": "Bolivia",
    "Micronesia, Federated States of": "Micronesia",
    "Palestine, State of": "Palestine",
    "Côte d'Ivoire": "Ivory Coast",
    "Czechia": "Czech Republic",
}


def clean_country(name):
    """Clean a single country name using: pre-fix → pycountry → post-fix."""
    if pd.isna(name):
        return name

    trimmed = name.strip()
    lower = trimmed.lower()

    # Step 1: Pre-fix
    if lower in COUNTRY_PRE_FIX:
        return COUNTRY_PRE_FIX[lower]

    # Step 2: Try pycountry exact match, then fuzzy
    result = pycountry.countries.get(name=trimmed)
    if not result:
        try:
            result = pycountry.countries.search_fuzzy(trimmed)[0]
        except LookupError:
            return trimmed

    official = result.name

    # Step 3: Post-fix
    if official in COUNTRY_POST_FIX:
        return COUNTRY_POST_FIX[official]

    return official


def clean_country_cell(cell):
    """Handle cells with multiple comma-separated countries."""
    if pd.isna(cell):
        return cell

    text = cell.strip()

    # Fix Côte d'Ivoire before comma split (it contains special chars)
    text = text.replace("Côte d\u2019Ivoire", "Ivory Coast")
    text = text.replace("Côte d'Ivoire", "Ivory Coast")

    # Split by comma, clean each, remove duplicates
    countries = [clean_country(c.strip()) for c in text.split(",")]
    seen = set()
    unique = []
    for c in countries:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return ", ".join(unique)


def harmonize_country(df):
    """Apply country cleaning to the dataframe. Returns df with new column."""
    before = df['country'].nunique()
    df['country_cleaned'] = df['country'].apply(clean_country_cell)
    after = df['country_cleaned'].nunique()

    all_countries = df['country_cleaned'].dropna().str.split(", ").explode().str.strip()

    print(f"Country - Before: {before} unique combinations")
    print(f"Country - After:  {after} unique combinations")
    print(f"Country - Unique individual countries: {all_countries.nunique()}")

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/trials_sample.csv")
    df = harmonize_country(df)
    df.to_csv("data/processed/trials_country_cleaned.csv", index=False)
    print("Saved to data/processed/trials_country_cleaned.csv")