"""
Project Data Extractor & Normalization Module
Extracts, normalizes, and validates project risk data from CSV spreadsheets and PDF documents.
Supports alias column mapping, PDF key-value text extraction, and strict 20-feature validation.
"""

import re
import io
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional

# Supported Categorical Options matching model training taxonomy
VALID_CATEGORICAL_OPTIONS = {
    "project_type": [
        "Software Development", "IT Infrastructure", "ERP Implementation", "Healthcare IT",
        "Financial Systems", "Construction", "Manufacturing", "Marketing Campaign", "R&D", "Telecom"
    ],
    "industry_sector": [
        "IT", "Finance", "Healthcare", "Manufacturing", "Retail", "Telecom", "Government", "Construction"
    ],
    "methodology": [
        "Agile", "Hybrid", "Waterfall"
    ],
    "region": [
        "North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa"
    ],
    "priority": [
        "High", "Medium", "Low", "Critical"
    ]
}

# Feature Naming Aliases Mapping Dictionary
FIELD_ALIASES = {
    "project_name": [
        "project_name", "project name", "projectname", "name of project", "project title", "title"
    ],
    "project_type": [
        "project_type", "project type", "projecttype", "type", "domain", "category"
    ],
    "industry_sector": [
        "industry_sector", "industry sector", "industrysector", "industry", "sector"
    ],
    "methodology": [
        "methodology", "dev_methodology", "development methodology", "process methodology", "process"
    ],
    "region": [
        "region", "operating_region", "location", "geography", "zone"
    ],
    "priority": [
        "priority", "project_priority", "urgency", "importance"
    ],
    "planned_duration_days": [
        "planned_duration_days", "planned duration days", "planned duration", "duration days",
        "duration", "timeline days", "planned_duration", "project duration"
    ],
    "budget_usd": [
        "budget_usd", "budget usd", "project budget", "budget", "cost usd", "total budget", "cost"
    ],
    "requirement_changes_count": [
        "requirement_changes_count", "requirement changes count", "requirement changes",
        "scope changes", "requirement_changes", "number of requirement changes"
    ],
    "vendor_dependency_count": [
        "vendor_dependency_count", "vendor dependency count", "vendor dependencies",
        "vendor_dependency", "vendors", "number of vendors", "vendor dependency"
    ],
    "milestones_missed": [
        "milestones_missed", "milestones missed", "missed milestones", "delayed milestones"
    ],
    "team_size": [
        "team_size", "team size", "number of team members", "headcount", "members", "team members"
    ],
    "team_avg_experience_years": [
        "team_avg_experience_years", "team avg experience years", "average team experience",
        "team experience", "avg experience years", "average experience", "experience"
    ],
    "team_turnover_pct": [
        "team_turnover_pct", "team turnover pct", "team turnover", "turnover rate",
        "turnover pct", "attrition rate", "turnover"
    ],
    "resource_availability_pct": [
        "resource_availability_pct", "resource availability pct", "resource availability",
        "availability pct", "resources available", "availability"
    ],
    "communication_score": [
        "communication_score", "communication score", "comm score", "team communication score", "communication"
    ],
    "sponsor_engagement_score": [
        "sponsor_engagement_score", "sponsor engagement score", "sponsor engagement",
        "stakeholder engagement score", "sponsor score"
    ],
    "tech_complexity_score": [
        "tech_complexity_score", "tech complexity score", "technical complexity score",
        "technical complexity", "complexity score", "tech complexity"
    ],
    "scope_clarity_score": [
        "scope_clarity_score", "scope clarity score", "scope clarity", "clarity score"
    ],
    "external_dependency_score": [
        "external_dependency_score", "external dependency score", "external dependency",
        "external dependencies score", "external dependency risk", "external dependencies"
    ],
    "defect_count": [
        "defect_count", "defect count", "defects", "bugs count", "total defects", "bugs"
    ]
}

# The 20 Required Model Features
REQUIRED_ML_FEATURES = [
    "project_type", "industry_sector", "methodology", "region", "priority",
    "planned_duration_days", "budget_usd", "requirement_changes_count",
    "vendor_dependency_count", "milestones_missed", "team_size",
    "team_avg_experience_years", "team_turnover_pct", "resource_availability_pct",
    "communication_score", "sponsor_engagement_score", "tech_complexity_score",
    "scope_clarity_score", "external_dependency_score", "defect_count"
]

NUMERICAL_FEATURES = [
    "planned_duration_days", "budget_usd", "requirement_changes_count",
    "vendor_dependency_count", "milestones_missed", "team_size",
    "team_avg_experience_years", "team_turnover_pct", "resource_availability_pct",
    "communication_score", "sponsor_engagement_score", "tech_complexity_score",
    "scope_clarity_score", "external_dependency_score", "defect_count"
]


def normalize_column_name(raw_name: str) -> Optional[str]:
    """Maps raw column/header string to standard internal feature field name."""
    clean_str = re.sub(r'[^a-zA-Z0-9]', ' ', str(raw_name)).strip().lower()
    clean_str_sub = re.sub(r'\s+', ' ', clean_str)

    # 1. Exact match check (highest priority)
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            clean_alias = re.sub(r'\s+', ' ', alias.lower())
            if clean_str_sub == clean_alias:
                return field

    # 2. Word boundary substring match (longer aliases first)
    all_alias_pairs = []
    for field, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            all_alias_pairs.append((field, alias))
    all_alias_pairs.sort(key=lambda x: len(x[1]), reverse=True)

    for field, alias in all_alias_pairs:
        clean_alias = re.sub(r'\s+', ' ', alias.lower())
        pattern = r'\b' + re.escape(clean_alias) + r'\b'
        if re.search(pattern, clean_str_sub):
            return field

    return None


def clean_numeric_value(val: Any) -> Optional[float]:
    """Cleans numeric string input (removes $, %, commas, units) and converts to float."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace(',', '')
    match = re.search(r'\d+(?:\.\d+)?', s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def match_categorical_option(field: str, val: Any) -> Optional[str]:
    """Matches raw categorical input value against valid model options."""
    if not val or pd.isna(val):
        return None

    raw_val = str(val).strip().lower()
    valid_opts = VALID_CATEGORICAL_OPTIONS.get(field, [])

    # Exact match
    for opt in valid_opts:
        if raw_val == opt.lower():
            return opt

    # Substring match
    for opt in valid_opts:
        if raw_val in opt.lower() or opt.lower() in raw_val:
            return opt

    return None


def extract_from_csv(file_buffer) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Reads a CSV spreadsheet, normalizes column headers, extracts row records,
    and converts numeric/categorical types.
    Returns: (list of normalized row dicts, error_message)
    """
    try:
        df = pd.read_csv(file_buffer)
        if df.empty:
            return [], "Uploaded CSV file is empty."

        # Map DataFrame column names to internal fields
        col_mapping = {}
        for col in df.columns:
            std_name = normalize_column_name(col)
            if std_name:
                col_mapping[col] = std_name

        if not col_mapping:
            return [], "Could not map any valid project risk features from CSV headers."

        df_mapped = df.rename(columns=col_mapping)
        records = []

        for idx, row in df_mapped.iterrows():
            rec = {}
            for field in list(FIELD_ALIASES.keys()):
                if field in row:
                    val = row[field]
                    if field in NUMERICAL_FEATURES:
                        rec[field] = clean_numeric_value(val)
                    elif field in VALID_CATEGORICAL_OPTIONS:
                        rec[field] = match_categorical_option(field, val)
                    else:
                        rec[field] = str(val).strip() if pd.notna(val) else None
            records.append(rec)

        return records, None

    except Exception as e:
        return [], f"Error processing CSV file: {str(e)}"


def extract_from_pdf(file_buffer) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Extracts readable text from PDF document using pypdf,
    parses key-value pairs, and normalizes feature values.
    Returns: (normalized feature dictionary, error_message)
    """
    text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(file_buffer)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    except Exception as e:
        return {}, f"Could not read PDF document: {str(e)}"

    if not text.strip():
        return {}, "PDF document contains no readable text or may be an image-only scan."

    extracted = {}
    lines = text.splitlines()

    for line in lines:
        if ":" in line or "=" in line:
            parts = re.split(r'[:=]', line, maxsplit=1)
            raw_key = parts[0].strip()
            raw_val = parts[1].strip()

            std_field = normalize_column_name(raw_key)
            if std_field and std_field not in extracted:
                if std_field in NUMERICAL_FEATURES:
                    extracted[std_field] = clean_numeric_value(raw_val)
                elif std_field in VALID_CATEGORICAL_OPTIONS:
                    extracted[std_field] = match_categorical_option(std_field, raw_val)
                else:
                    extracted[std_field] = raw_val

    # Regex search fallback across full body text if key-value splitting missed inline text
    if "project_name" not in extracted or not extracted["project_name"]:
        m = re.search(r'(?i)project\s*name\s*[:=-]?\s*([^\n\r]+)', text)
        if m:
            extracted["project_name"] = m.group(1).strip()

    for num_f in NUMERICAL_FEATURES:
        if num_f not in extracted or extracted[num_f] is None:
            pattern_aliases = FIELD_ALIASES[num_f]
            for alias in pattern_aliases:
                pattern = r'(?i)' + re.escape(alias) + r'\s*[:=-]?\s*\$?([0-9.,]+)'
                m = re.search(pattern, text)
                if m:
                    extracted[num_f] = clean_numeric_value(m.group(1))
                    break

    return extracted, None


def validate_extracted_features(extracted_dict: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Checks extracted project dictionary for the required 20 ML features.
    Returns: (list of missing_feature_names, cleaned_dict)
    """
    missing = []
    cleaned = {}

    for feat in REQUIRED_ML_FEATURES:
        val = extracted_dict.get(feat)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(feat)
        else:
            cleaned[feat] = val

    if "project_name" in extracted_dict and extracted_dict["project_name"]:
        cleaned["project_name"] = extracted_dict["project_name"]

    return missing, cleaned
