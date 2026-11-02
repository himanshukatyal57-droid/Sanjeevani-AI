# %% [markdown]
# # Sanjeevani AI - Dataset Cleaning & Exploratory Data Analysis
#
# **Scope:** load -> inspect -> clean headers -> EDA -> data-quality report -> export cleaned copy.
# **Out of scope:** model building, diagnosis, treatment suggestions, clinical validity claims.
#
# **Design rules followed throughout**
# - The original `df` is never modified; all cleaning happens on `df_clean` / `df_final`.
# - Nothing is deleted, imputed, merged or converted automatically unless you switch the option on in Section 14.
# - Columns that look like names, contact details, addresses or identifiers are **masked in every preview**.
# - No column names are hard-coded; everything is inferred from the dataset you load.
#
# Sections: 1 Imports - 2 Load - 3 Inspect - 4 Clean headers - 5 Mapping - 6 Data types - 7 Missing - 8 Duplicates - 9 Numeric - 10 Categorical - 11 Dates - 12 Correlation - 13 Quality report - 14 Export - 15 Verification

# %% [markdown]
# ## 1. Importing Libraries & Configuration
# **Purpose:** import Pandas/NumPy (Matplotlib/Seaborn only for plots), define thresholds and privacy helpers.
# **Expected output:** library versions.
# **Look for:** if Seaborn is missing the notebook still works (plain Matplotlib fallback).

# %%
import re
import json
import difflib
import unicodedata
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)
pd.set_option("display.float_format", "{:,.4f}".format)

# `display` exists in Jupyter/Colab; fall back to print for plain Python
try:
    display
except NameError:
    display = print

# ----------------------------- Configuration -----------------------------
HIGH_MISSING_PCT = 40.0        # columns at/above this % missing are flagged "unusually high"
CAT_MAX_UNIQUE_CAP = 100       # upper limit on unique values for a column to count as categorical
OUTLIER_IQR_MULT = 1.5         # IQR fence multiplier
MIN_GROUP_SIZE = 5             # group/cross-tab cells smaller than this are suppressed (privacy)
MAX_PLOTS = 12                 # max number of columns to plot per figure
MAX_CORR_COLS = 40             # cap for correlation matrix width
DATE_DAYFIRST = None           # None = never guess dd/mm vs mm/dd; True = dd/mm/yyyy; False = mm/dd/yyyy
EXTRA_SENSITIVE_COLUMNS = []   # add column names (raw or cleaned) you want masked
NOT_SENSITIVE_COLUMNS = []     # add column names the privacy heuristic masked by mistake

# ------------------------- Privacy / identifier helpers -------------------------
PII_TOKENS = {"name", "fname", "lname", "surname", "phone", "mobile", "contact", "email", "address",
              "aadhaar", "aadhar", "ssn", "passport", "mrn", "dob", "birth", "pincode", "zip",
              "postal", "insurance", "license", "licence", "guardian"}
PII_SUBSTRINGS = ("phone", "email", "address", "aadhaar", "aadhar", "passport", "mobile", "pincode", "surname")
ID_TOKENS = {"id", "uid", "uuid", "guid", "mrn", "uhid", "abha"}


def name_tokens(col):
    """Split a column name into lowercase word tokens (handles CamelCase, spaces, underscores)."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(col))
    return re.findall(r"[^\W_]+", text.lower())


def is_pii_name(col):
    tokens = name_tokens(col)
    joined = "".join(tokens)
    return bool(PII_TOKENS & set(tokens)) or any(k in joined for k in PII_SUBSTRINGS)


def is_id_name(col):
    return bool(ID_TOKENS & set(name_tokens(col)))


def is_sensitive(col):
    """Heuristic: column probably holds personal data or identifiers -> never print its values."""
    if col in NOT_SENSITIVE_COLUMNS:
        return False
    if col in EXTRA_SENSITIVE_COLUMNS:
        return True
    return is_pii_name(col) or is_id_name(col)


def safe_preview(frame, n=5, where="head"):
    """Return head/tail rows with sensitive columns masked."""
    sub = frame.head(n) if where == "head" else frame.tail(n)
    masked = sub.astype(object).copy()
    for i, col in enumerate(masked.columns):
        if is_sensitive(col):
            masked.iloc[:, i] = "[MASKED]"
    return masked


def join_cols(cols, limit=8):
    cols = [str(c) for c in cols]
    if len(cols) <= limit:
        return ", ".join(cols)
    return ", ".join(cols[:limit]) + f" (+{len(cols) - limit} more)"


# ------------------------- Issue registry (feeds Section 13) -------------------------
ISSUES = {}  # keyed by (issue, column) so re-running a cell never creates duplicates
PRIORITY_ORDER = {"High Priority": 0, "Medium Priority": 1, "Low Priority": 2}


def add_issue(issue, column, observation, action, priority):
    ISSUES[(issue, column)] = {
        "Issue": issue, "Affected Column": column, "Observation": observation,
        "Recommended Action": action, "Priority": priority,
    }


print("pandas", pd.__version__, "| numpy", np.__version__, "| seaborn available:", HAS_SEABORN)

# %% [markdown]
# ## 2. Loading Dataset
# **Purpose:** load CSV / TSV / Excel / JSON / Parquet into `df` without assuming any structure.
# **Expected output:** file format, encoding used, shape, and a masked preview.
# **Look for:** a single wide column (wrong delimiter), garbled characters (wrong encoding), or `NA`/`None` strings silently turned into NaN by the reader (a few real categories can look like that).

# %%
DATA_PATH = r"C:\Users\Shivam Gupta\Coding\sanjeevani-ml-service\data\Final_Augmented_dataset_Diseases_and_Symptoms.csv"
 # <-- change me (.csv, .tsv, .txt, .xlsx, .xls, .json, .parquet)
SHEET_NAME = 0                        # only used for Excel files


def load_dataset(path, sheet_name=0):
    """Load a tabular file into a DataFrame, trying sensible encodings for text files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at '{p}'. Update DATA_PATH.")
    ext = p.suffix.lower()

    if ext in {".csv", ".txt", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        last_error = None
        for enc in ("utf-8-sig", "latin-1"):          # utf-8-sig also strips a BOM
            try:
                frame = pd.read_csv(p, sep=sep, encoding=enc)
                if frame.shape[1] == 1 and ext != ".tsv":   # maybe a different delimiter
                    frame = pd.read_csv(p, sep=None, engine="python", encoding=enc)
                print(f"Loaded as delimited text (encoding={enc}).")
                return frame
            except UnicodeDecodeError as err:
                last_error = err
        raise last_error
    if ext in {".xlsx", ".xlsm", ".xls"}:
        print("Loaded as Excel (needs openpyxl/xlrd installed).")
        return pd.read_excel(p, sheet_name=sheet_name)
    if ext == ".json":
        print("Loaded as JSON.")
        return pd.read_json(p)
    if ext == ".parquet":
        print("Loaded as Parquet.")
        return pd.read_parquet(p)
    raise ValueError(f"Unsupported file type '{ext}'.")


try:
    df = load_dataset(DATA_PATH, SHEET_NAME)
except Exception as err:
    print(f"Could not load dataset: {err}")
    raise

# Snapshot so Section 15 can prove the original DataFrame was never altered
try:
    ORIGINAL_HASH = int(pd.util.hash_pandas_object(df.astype(str), index=True).sum())
except Exception:
    ORIGINAL_HASH = None
ORIGINAL_SHAPE, ORIGINAL_COLUMNS = df.shape, list(df.columns)

print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
display(safe_preview(df, 5))

# %% [markdown]
# ## 3. Initial Dataset Inspection
# **Purpose:** understand size, types, memory and basic statistics before touching anything.
# **What each call tells you**
# - `head()/tail()` - how records start/end (sorted? footers or totals row at the bottom?)
# - `shape` - rows x columns; `columns` - raw headers (spot spaces, symbols, duplicates)
# - `dtypes` / `info()` - stored types and non-null counts (numbers stored as `object` are a red flag)
# - `memory_usage(deep=True)` - real footprint, including strings
# - `describe()` / `describe(include="all")` - ranges, spread, top values (sensitive columns are excluded here)
#
# **Look for:** impossible minimums/maximums, `object` columns that should be numeric/date, very large memory use.

# %%
print("=== First 5 records (sensitive columns masked) ===")
display(safe_preview(df, 5, "head"))
print("\n=== Last 5 records (sensitive columns masked) ===")
display(safe_preview(df, 5, "tail"))

print(f"\n=== Dimensions ===\nRows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")
print("\n=== Raw column names ===")
print(list(df.columns))

print("\n=== Data types ===")
display(df.dtypes.astype(str).rename("dtype").to_frame())

print("\n=== df.info() ===")
df.info()

mem = df.memory_usage(deep=True)
print(f"\n=== Memory usage: {mem.sum() / 1024**2:,.2f} MB total ===")
display((mem / 1024).round(2).rename("KB").to_frame().T if df.shape[1] > 15 else (mem / 1024).round(2).rename("KB").to_frame())

# Statistical summaries: sensitive columns are left out so no personal values appear in output
non_sensitive_mask = [not is_sensitive(c) for c in df.columns]
df_safe_view = df.loc[:, non_sensitive_mask]
if df_safe_view.shape[1] == 0:
    print("\nAll columns look sensitive; statistical summaries skipped.")
else:
    print("\n=== describe() - numeric columns ===")
    if df_safe_view.select_dtypes(include="number").shape[1]:
        display(df_safe_view.describe())
    else:
        print("No numeric columns.")
    print("\n=== describe(include='all') ===")
    display(df_safe_view.describe(include="all"))
n_masked = len(df.columns) - df_safe_view.shape[1]
if n_masked:
    print(f"\n[Privacy] {n_masked} potentially sensitive column(s) were excluded from summaries/previews.")

# %% [markdown]
# ## 4. Column Header Cleaning
# **Purpose:** reusable function that converts every header to `snake_case`, keeps every column, and guarantees unique names.
#
# **Rules implemented:** lowercase - trim - spaces/special characters -> `_` - apostrophes removed - `%`, `&`, `#`, `@`, `+` spelled out (so `A+` stays distinguishable from `A-`) - multiple `_` collapsed - leading/trailing `_` removed - non-string names converted safely - empty names get `unnamed_column_<position>` - duplicates get `_2`, `_3`... **and are reported, never silently resolved**.
#
# Unit text in headers (e.g. `bpm`) is kept as written; no units are assumed. CamelCase splitting is **off** by default because it can damage names like `SpO2`; set `split_camel_case=True` if you want it.
# **Expected output:** counts of changed/flagged headers. **Look for:** collisions and empty names (need a human decision).

# %%
SYMBOL_WORDS = {"%": " percent ", "&": " and ", "#": " number ", "@": " at ", "+": " plus "}


def clean_column_name(name, split_camel_case=False):
    """Convert a single column label to snake_case (may return '' if nothing usable remains)."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ""
    text = unicodedata.normalize("NFKC", str(name)).replace("\u00a0", " ")
    text = re.sub(r"['\u2019`]", "", text)                      # patient's -> patients
    for symbol, word in SYMBOL_WORDS.items():
        text = text.replace(symbol, word)
    if split_camel_case:
        text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
        text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w]+", "_", text)                         # unicode-aware: keeps letters/digits of any script
    return re.sub(r"_+", "_", text).strip("_")


def standardize_columns(columns, split_camel_case=False):
    """Return a mapping DataFrame: position, original_name, cleaned_name, changed, notes."""
    originals = list(columns)
    bases = [clean_column_name(c, split_camel_case) for c in originals]
    base_counts = Counter(b for b in bases if b)
    reserved = set(b for b in bases if b)       # so a suffixed name never steals another column's natural name
    used, rows = set(), []

    for pos, (orig, base) in enumerate(zip(originals, bases)):
        notes = []
        if not isinstance(orig, str):
            notes.append(f"non_string_name({type(orig).__name__})")
        if base == "":
            base = f"unnamed_column_{pos + 1}"
            notes.append("empty_after_cleaning")
            reserved.add(base)
        final = base
        if final in used:
            k = 2
            while f"{base}_{k}" in used or f"{base}_{k}" in reserved:
                k += 1
            final = f"{base}_{k}"
        if base_counts.get(bases[pos], 0) > 1:
            notes.append(f"collision:{base_counts[bases[pos]]}_columns_share_'{base}'")
        if final[:1].isdigit():
            notes.append("starts_with_digit(df.col access needs brackets)")
        used.add(final)
        rows.append({"position": pos, "original_name": str(orig), "cleaned_name": final,
                     "changed": str(orig) != final, "notes": "; ".join(notes)})
    return pd.DataFrame(rows)


# Quick self-test on the examples from the specification
examples = ["Patient Name", "Age (Years)", "Blood Pressure", "Heart Rate (BPM)",
            "Symptoms Description", "Admission Date", "Patient's ID#", "Blood Group A+", " SpO2 % "]
for ex in examples:
    print(f"{ex!r:28} -> {clean_column_name(ex)!r}")

# %% [markdown]
# ## 5. Original-to-Cleaned Column Mapping
# **Purpose:** apply the function to the real headers, keep a full mapping, and verify the result.
# **Expected output:** mapping table (changed rows), flagged rows, and a pass/fail checklist.
# **Look for:** any row in the *flagged* table - these were not obvious and deserve a manual look. The original `df` keeps its original headers; `df_clean` carries the new ones.

# %%
column_mapping = standardize_columns(df.columns, split_camel_case=False)

df_clean = df.copy()                                   # original df stays untouched
df_clean.columns = column_mapping["cleaned_name"].tolist()

print(f"{int(column_mapping['changed'].sum())} of {len(column_mapping)} headers changed.")
print("\n=== Headers that changed ===")
display(column_mapping.loc[column_mapping["changed"], ["position", "original_name", "cleaned_name", "notes"]]
        if column_mapping["changed"].any() else "No headers needed changes.")

flagged = column_mapping[column_mapping["notes"] != ""]
print("\n=== Flagged headers (review manually - not silently resolved) ===")
display(flagged[["position", "original_name", "cleaned_name", "notes"]] if len(flagged) else "None.")

# ---- Verification ----
SNAKE_RE = re.compile(r"^[^\W_]+(?:_[^\W_]+)*$")
checks = {
    "same number of columns as original": len(df_clean.columns) == len(df.columns),
    "all names unique": df_clean.columns.is_unique,
    "all names lowercase": all(c == c.lower() for c in df_clean.columns),
    "all names valid snake_case": all(SNAKE_RE.match(c) for c in df_clean.columns),
    "original df headers untouched": list(df.columns) == ORIGINAL_COLUMNS,
}
for label, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")

# ---- Register header issues ----
n_changed = int(column_mapping["changed"].sum())
if n_changed:
    add_issue("Non-standard column headers", f"{n_changed} columns",
              "Headers contain capitals, spaces, symbols or padding.",
              "Use cleaned headers from df_clean / mapping table.", "Low Priority")
coll = column_mapping[column_mapping["notes"].str.contains("collision")]
if len(coll):
    add_issue("Duplicate/ambiguous column headers", join_cols([repr(x) for x in coll["original_name"]]),
              "Several headers reduce to the same cleaned name; suffixes _2, _3 were added.",
              "Confirm which column is which and rename meaningfully.", "High Priority")
empty = column_mapping[column_mapping["notes"].str.contains("empty_after_cleaning")]
if len(empty):
    add_issue("Empty/unusable column headers", join_cols([repr(x) for x in empty["original_name"]]),
              "Header was blank or only symbols; placeholder name assigned.",
              "Identify the real meaning of these columns.", "High Priority")

# %% [markdown]
# ## 6. Data Type Analysis
# **Purpose:** classify every column as numeric / categorical / text / datetime / boolean / identifier and flag type problems. Nothing is converted here.
# **Expected output:** one profile table plus lists of columns per role.
# **Look for:** `numeric_as_string`, `date_as_string`, `mixed_types`, `constant`, `mostly_unique`, and `sensitive_or_identifying` flags. Roles marked *(inferred)* are heuristics - confirm with the data owner.

# %%
BOOL_TEXT = {"yes", "no", "y", "n", "true", "false", "t", "f", "0", "1"}
DATE_NAME_HINTS = {"date", "time", "datetime", "timestamp", "dob", "dt"}
DATE_REGEX = r"\d{1,4}[-/.\s][A-Za-z0-9]{1,9}[-/.\s,]+\d{1,4}"
PLACEHOLDER_STRINGS = {"", "na", "n/a", "nan", "null", "none", "nil", "unknown", "?", "-", "--",
                       "not available", "missing", "undefined"}


def is_text_like(s):
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype))


def date_parse_ratio(strings):
    """Share of strings that parse as dates (only used for *detection*, nothing is converted)."""
    if len(strings) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(strings, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            parsed = pd.to_datetime(strings, errors="coerce")
    return float(parsed.notna().mean())


def profile_column(series, col):
    non_null = series.dropna()
    n_nn = len(non_null)
    n_unique = int(non_null.nunique()) if n_nn else 0
    unique_ratio = n_unique / n_nn if n_nn else 0.0
    flags, notes, role = [], [], "unknown"

    if is_sensitive(col):
        flags.append("sensitive_or_identifying")
    if n_nn == 0:
        flags.append("all_missing")
    elif n_unique <= 1:
        flags.append("constant")
    if pd.api.types.is_object_dtype(series) and n_nn:
        type_counts = non_null.map(lambda v: type(v).__name__).value_counts()
        if len(type_counts) > 1:
            flags.append("mixed_types")
            notes.append("python types: " + ", ".join(f"{k}={v}" for k, v in type_counts.items()))

    if pd.api.types.is_bool_dtype(series):
        role = "boolean"
    elif pd.api.types.is_datetime64_any_dtype(series):
        role = "datetime"
    elif is_id_name(col):
        role = "identifier"
    elif pd.api.types.is_numeric_dtype(series):
        role = "numeric"
        if (pd.api.types.is_integer_dtype(series) and n_nn > 20 and n_unique == n_nn
                and non_null.is_monotonic_increasing):
            role = "identifier"
            notes.append("unique, increasing integers (inferred sequence ID - verify)")
        elif n_unique == 2:
            flags.append("binary_numeric")
            notes.append("only 2 distinct values; may be a coded flag")
        elif 2 < n_unique <= 10 and n_nn >= 50:
            flags.append("low_cardinality_numeric")
            notes.append("few distinct values; may be coded categories")
    elif is_text_like(series):
        raw_str = non_null.astype(str)
        stripped = raw_str.str.strip()
        if (raw_str != stripped).any():
            flags.append("leading_trailing_whitespace")
        nb = stripped[stripped != ""]
        if len(nb) == 0:
            role = "text"
        else:
            lowered = set(nb.str.lower().unique())
            num_ratio = pd.to_numeric(nb.str.replace(",", "", regex=False), errors="coerce").notna().mean()
            if len(lowered) == 2 and lowered <= BOOL_TEXT:
                role = "boolean"
                flags.append("boolean_like_text")
            elif num_ratio >= 0.9:
                role = "numeric_as_string"
                flags.append("numeric_as_string")
                notes.append(f"{num_ratio:.0%} of values parse as numbers")
            else:
                sample = nb.head(500)
                parsed_ratio = date_parse_ratio(sample)
                pattern_ratio = float(sample.str.contains(DATE_REGEX).mean())
                name_hint = bool(DATE_NAME_HINTS & set(name_tokens(col)))
                if (parsed_ratio >= 0.8 and (pattern_ratio >= 0.8 or name_hint)) or (name_hint and parsed_ratio >= 0.5):
                    role = "date_as_string"
                    flags.append("date_as_string")
                    notes.append(f"{parsed_ratio:.0%} of sampled values parse as dates")
                else:
                    avg_len, avg_words = nb.str.len().mean(), nb.str.split().str.len().mean()
                    cat_limit = min(max(20, int(0.05 * n_nn)), CAT_MAX_UNIQUE_CAP)
                    if avg_words >= 4 or avg_len >= 40:
                        role = "text"
                    elif n_unique <= cat_limit:
                        role = "categorical"
                    elif unique_ratio >= 0.98:
                        role = "identifier"
                        notes.append("almost every value unique (inferred ID - verify)")
                    else:
                        role = "text"
                        flags.append("high_cardinality_text")

    if (unique_ratio >= 0.9 and n_nn > 20 and role not in {"identifier", "text"}
            and "constant" not in flags and pd.api.types.is_float_dtype(series) is False):
        flags.append("mostly_unique")

    recs = []
    if "numeric_as_string" in flags:
        recs.append("inspect non-numeric tokens, then convert with pd.to_numeric")
    if "date_as_string" in flags:
        recs.append("parse explicitly with a known format (see Section 11)")
    if "mixed_types" in flags:
        recs.append("find and fix the odd-typed entries")
    if "constant" in flags:
        recs.append("likely uninformative; confirm before dropping")
    if "mostly_unique" in flags:
        recs.append("check whether this is an identifier")
    if "sensitive_or_identifying" in flags:
        recs.append("de-identify / exclude from modelling features")
    if "leading_trailing_whitespace" in flags:
        recs.append("strip whitespace")
    if "binary_numeric" in flags or "low_cardinality_numeric" in flags:
        recs.append("confirm whether numeric or coded category")
    return {"column": col, "pandas_dtype": str(series.dtype), "inferred_role": role,
            "non_null": n_nn, "n_unique": n_unique, "unique_pct": round(unique_ratio * 100, 1),
            "flags": ", ".join(flags), "notes": "; ".join(notes), "recommendation": "; ".join(recs)}


type_summary = pd.DataFrame([profile_column(df_clean.iloc[:, i], c) for i, c in enumerate(df_clean.columns)])
display(type_summary)

role_of = type_summary.set_index("column")["inferred_role"]
numeric_cols = [c for c in df_clean.columns if role_of[c] == "numeric"]
categorical_cols = [c for c in df_clean.columns if role_of[c] == "categorical"]
text_cols = [c for c in df_clean.columns if role_of[c] == "text"]
datetime_cols = [c for c in df_clean.columns if role_of[c] in {"datetime", "date_as_string"}]
boolean_cols = [c for c in df_clean.columns if role_of[c] == "boolean"]
identifier_cols = [c for c in df_clean.columns if role_of[c] == "identifier"]
numeric_like_string_cols = [c for c in df_clean.columns if role_of[c] == "numeric_as_string"]
date_like_string_cols = [c for c in df_clean.columns if role_of[c] == "date_as_string"]
sensitive_cols = [c for c in df_clean.columns if is_sensitive(c)]
analysis_numeric_cols = [c for c in numeric_cols if c not in sensitive_cols]   # never summarise phone-like columns

for label, cols in [("Numeric", numeric_cols), ("Categorical", categorical_cols), ("Text", text_cols),
                    ("Date/time", datetime_cols), ("Boolean", boolean_cols), ("Identifier", identifier_cols),
                    ("Numeric stored as string (NOT converted)", numeric_like_string_cols),
                    ("Potentially sensitive (masked)", sensitive_cols)]:
    print(f"{label:42}: {join_cols(cols, 15) if cols else '-'}")

# ---- Register issues ----
def flagged_cols(flag):
    return type_summary.loc[type_summary["flags"].str.contains(flag, regex=False), "column"].tolist()

if flagged_cols("numeric_as_string"):
    add_issue("Numeric data stored as text", join_cols(flagged_cols("numeric_as_string")),
              "Values look numeric but are stored as strings.",
              "Review odd tokens, then convert with pd.to_numeric.", "Medium Priority")
if flagged_cols("date_as_string"):
    add_issue("Dates stored as text", join_cols(flagged_cols("date_as_string")),
              "Values look like dates but dtype is not datetime.",
              "Parse with an explicit, confirmed format.", "Medium Priority")
if flagged_cols("mixed_types"):
    add_issue("Mixed data types within a column", join_cols(flagged_cols("mixed_types")),
              "Different Python types coexist in the same column.", "Locate and correct inconsistent entries.", "High Priority")
if flagged_cols("constant"):
    add_issue("Constant / single-value column", join_cols(flagged_cols("constant")),
              "Only one distinct (or no) value.", "Confirm it carries no information before dropping.", "Low Priority")
if flagged_cols("mostly_unique"):
    add_issue("Mostly-unique non-ID column", join_cols(flagged_cols("mostly_unique")),
              "Nearly every value is distinct.", "Check if it is an identifier or free text.", "Low Priority")
if flagged_cols("leading_trailing_whitespace"):
    add_issue("Whitespace padding in text values", join_cols(flagged_cols("leading_trailing_whitespace")),
              "Values have leading/trailing spaces (can split categories).", "Strip whitespace.", "Low Priority")
if sensitive_cols:
    add_issue("Potentially sensitive / identifying columns", join_cols(sensitive_cols),
              "Column names suggest personal data or identifiers (heuristic).",
              "Confirm, then de-identify or restrict access before sharing or modelling.", "High Priority")
if identifier_cols:
    add_issue("Identifier columns present", join_cols(identifier_cols),
              "Identifier-like columns carry no predictive meaning.", "Keep for joins/auditing; exclude from features.", "Low Priority")

# %% [markdown]
# ## 7. Missing Value Analysis
# **Purpose:** quantify missingness per column and per row, and spot *disguised* missing values (e.g. `"NA"`, `"?"`, blank strings). Nothing is dropped or filled.
# **Expected output:** summary table (Column / Missing Count / Missing Percentage / treatment hint), row-level counts, a bar chart.
# **Look for:** columns >= `HIGH_MISSING_PCT`, rows with many gaps, and placeholder strings that pandas does not count as missing. In medical data, a gap can itself carry meaning (test not ordered, symptom not asked) - decide treatment with a domain expert.

# %%
def missing_summary_table(frame):
    n = len(frame)
    counts = frame.isna().sum()
    pct = (counts / n * 100) if n else counts * 0.0
    out = pd.DataFrame({"Column": counts.index, "Missing Count": counts.values.astype(int),
                        "Missing Percentage": pct.values})
    return out.sort_values(["Missing Count", "Column"], ascending=[False, True]).reset_index(drop=True)


def treatment_hint(pct):
    if pct == 0:
        return "none needed"
    if pct < 5:
        return "low: leave as-is or impute (median/mode) after checking the pattern"
    if pct < HIGH_MISSING_PCT:
        return "moderate: investigate pattern (random vs. systematic); consider imputation + missing-indicator"
    return "high: confirm if column is usable; do not drop/impute without domain input"


missing_summary = missing_summary_table(df_clean)
missing_summary["Suggested Treatment"] = missing_summary["Missing Percentage"].map(treatment_hint)
total_cells = df_clean.shape[0] * df_clean.shape[1]
total_missing = int(df_clean.isna().sum().sum())
rows_with_missing = int(df_clean.isna().any(axis=1).sum())

print(f"Total missing cells : {total_missing:,} of {total_cells:,} ({(total_missing / total_cells * 100) if total_cells else 0:.2f}%)")
print(f"Rows with >=1 missing: {rows_with_missing:,} ({(rows_with_missing / len(df_clean) * 100) if len(df_clean) else 0:.2f}%)")
print(f"Completely filled rows: {len(df_clean) - rows_with_missing:,}")
display(missing_summary)

high_missing = missing_summary[missing_summary["Missing Percentage"] >= HIGH_MISSING_PCT]
print(f"\nColumns with >= {HIGH_MISSING_PCT:.0f}% missing: {join_cols(high_missing['Column']) if len(high_missing) else 'none'}")

print("\nMissing values per row (how many rows have k missing fields):")
display(df_clean.isna().sum(axis=1).value_counts().sort_index().rename("rows").rename_axis("missing_fields_in_row").to_frame())

# Disguised missing values: strings that *mean* missing but are not NaN (reported only, never converted)
placeholder_counts = {}
for c in df_clean.columns:
    s = df_clean[c].dropna()
    if is_text_like(df_clean[c]) and len(s):
        n_ph = int(s.astype(str).str.strip().str.lower().isin(PLACEHOLDER_STRINGS).sum())
        if n_ph:
            placeholder_counts[c] = n_ph
print("\nPossible placeholder strings counted as real values (not converted):")
display(pd.Series(placeholder_counts, name="placeholder_count").to_frame() if placeholder_counts else "none found")

with_gaps = missing_summary[missing_summary["Missing Count"] > 0].head(25)
if len(with_gaps):
    ax = with_gaps.set_index("Column")["Missing Percentage"].sort_values().plot.barh(figsize=(8, max(2.5, 0.3 * len(with_gaps))))
    ax.set_xlabel("% missing")
    ax.set_title("Missing values by column (top 25)")
    plt.tight_layout()
    plt.show()

# ---- Register issues ----
for _, r in missing_summary[missing_summary["Missing Count"] > 0].iterrows():
    pct = r["Missing Percentage"]
    prio = "High Priority" if pct >= HIGH_MISSING_PCT else ("Medium Priority" if pct >= 5 else "Low Priority")
    add_issue("Missing values", r["Column"], f"{int(r['Missing Count']):,} missing ({pct:.1f}%).",
              treatment_hint(pct), prio)
for c, n_ph in placeholder_counts.items():
    add_issue("Possible disguised missing values", c, f"{n_ph:,} values look like placeholders (e.g. NA, ?, blank).",
              "Confirm meaning with data owner; convert to NaN only if truly missing.", "Medium Priority")

# %% [markdown]
# ## 8. Duplicate Analysis
# **Purpose:** find exact duplicate rows, identifier repeats, and rows that differ only in IDs or in case/whitespace.
# **Expected output:** counts, group sizes, a masked sample of duplicates.
# **Look for:** duplicates that are legitimate repeats (e.g. several visits by the same patient) versus accidental double entries.
# **Consequences of dropping:** deleting rows can erase genuine repeat events, shift class balance and bias later statistics; keeping them can inflate counts and leak identical rows across train/test splits. Decide only after checking with the data owner.

# %%
def safe_duplicated(frame, subset=None, keep="first"):
    """df.duplicated that survives unhashable cells (lists/dicts) by falling back to strings."""
    try:
        return frame.duplicated(subset=subset, keep=keep)
    except TypeError:
        return frame.astype(str).duplicated(subset=subset, keep=keep)


dup_extra = safe_duplicated(df_clean)                 # 2nd+ copies
dup_all = safe_duplicated(df_clean, keep=False)       # every row that belongs to a duplicate group
n_dup_extra, n_dup_all = int(dup_extra.sum()), int(dup_all.sum())
print(f"Exact duplicate rows (extra copies, keep='first'): {n_dup_extra:,}")
print(f"Rows belonging to any duplicate group           : {n_dup_all:,}")

if n_dup_all:
    row_key = pd.util.hash_pandas_object(df_clean.astype(str), index=False)
    group_sizes = row_key[dup_all].value_counts()
    print("\nDuplicate group sizes (rows per identical group):")
    display(group_sizes.value_counts().sort_index().rename("n_groups").rename_axis("rows_in_group").to_frame())
    print("Sample of duplicated rows (sensitive columns masked, so identical-looking rows are expected):")
    display(safe_preview(df_clean[dup_all].sort_index(), 10))
    print("To inspect in full privately:  df_clean[df_clean.duplicated(keep=False)]")

# Same content, different identifiers?
content_cols = [c for c in df_clean.columns if c not in identifier_cols]
n_dup_ignoring_ids = None
if identifier_cols and content_cols:
    n_dup_ignoring_ids = int(safe_duplicated(df_clean, subset=content_cols).sum())
    print(f"\nRows identical in every non-identifier column: {n_dup_ignoring_ids:,} (exact duplicates: {n_dup_extra:,})")

# Repeated identifier values
id_dups = {}
for c in identifier_cols:
    s = df_clean[c].dropna()
    n = int(s.duplicated(keep=False).sum())
    if n:
        id_dups[c] = n
print("\nIdentifier columns with repeated values (may be legitimate, e.g. repeat visits):")
display(pd.Series(id_dups, name="rows_sharing_an_id").to_frame() if id_dups else "none")

# Near-duplicates after trimming whitespace and ignoring case
df_norm = df_clean.copy()
for c in df_norm.columns:
    if pd.api.types.infer_dtype(df_norm[c], skipna=True) == "string":
        df_norm[c] = df_norm[c].str.strip().str.lower()
n_near = int(safe_duplicated(df_norm).sum()) - n_dup_extra
print(f"\nExtra duplicates revealed only after trimming/ignoring case: {max(n_near, 0):,}")

# ---- Register issues ----
if n_dup_extra:
    pct = n_dup_extra / len(df_clean) * 100
    add_issue("Exact duplicate rows", "all columns", f"{n_dup_extra:,} extra copies ({pct:.1f}% of rows).",
              "Check if legitimate repeats; if not, drop_duplicates() after approval.",
              "High Priority" if pct >= 5 else "Medium Priority")
if n_dup_ignoring_ids:
    add_issue("Rows identical except identifiers", join_cols(identifier_cols), f"{n_dup_ignoring_ids:,} rows repeat all non-ID content.",
              "Confirm whether these are re-entered records.", "Medium Priority")
for c, n in id_dups.items():
    add_issue("Repeated identifier values", c, f"{n:,} rows share an identifier with another row.",
              "Confirm whether the ID should be unique per row.", "Medium Priority")
if n_near > 0:
    add_issue("Near-duplicate rows (case/whitespace)", "text columns", f"{n_near:,} additional duplicates after normalising text.",
              "Standardise text before de-duplicating.", "Low Priority")

# %% [markdown]
# ## 9. Numerical Data Analysis
# **Purpose:** descriptive statistics, distribution shape, skewness, outliers (IQR rule) and unusual values for numeric, non-sensitive columns.
# **Expected output:** quick summary, detailed profile table, outlier table and histograms.
# **Look for:** skew beyond |1|, zeros/negatives that may be invalid for that variable, `inf` values, and many outliers. **Outliers are only flagged, never removed** - in clinical data extreme values can be real and important. If IQR = 0 the fences collapse, so every non-typical value looks like an outlier; judge those manually.

# %%
def numeric_profile(frame, cols, iqr_mult=OUTLIER_IQR_MULT):
    rows = []
    for c in cols:
        raw = pd.to_numeric(frame[c], errors="coerce").astype("float64")
        inf_count = int(np.isinf(raw).sum())
        s = raw.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            rows.append({"column": c, "count": 0})
            continue
        q1, q2, q3 = s.quantile([0.25, 0.50, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
        p1, p5, p95, p99 = np.percentile(s.to_numpy(), [1, 5, 95, 99])
        modes = s.mode()
        out_lo, out_hi = int((s < lo).sum()), int((s > hi).sum())
        rows.append({
            "column": c, "count": len(s), "mean": s.mean(), "median": q2,
            "mode": modes.iloc[0] if len(modes) else np.nan, "n_modes": len(modes),
            "std": s.std(), "variance": s.var(), "min": s.min(), "p1": p1, "p5": p5,
            "q1": q1, "q3": q3, "p95": p95, "p99": p99, "max": s.max(), "iqr": iqr,
            "skew": s.skew(), "kurtosis": s.kurt(),
            "lower_fence": lo, "upper_fence": hi, "outliers_low": out_lo, "outliers_high": out_hi,
            "outlier_pct": (out_lo + out_hi) / len(s) * 100,
            "zeros": int((s == 0).sum()), "negatives": int((s < 0).sum()), "inf_values": inf_count,
        })
    return pd.DataFrame(rows).set_index("column")


if not analysis_numeric_cols:
    print("No (non-sensitive) numeric columns detected - skipping numeric analysis.")
    numeric_summary = pd.DataFrame()
else:
    num_df = df_clean[analysis_numeric_cols].apply(pd.to_numeric, errors="coerce")

    print("=== Quick summary (vectorised pandas) ===")
    quick = pd.DataFrame({
        "mean": num_df.mean(numeric_only=True), "median": num_df.median(numeric_only=True),
        "std": num_df.std(numeric_only=True), "variance": num_df.var(numeric_only=True),
    })
    display(quick)
    print("\n=== Quartiles ===")
    display(num_df.quantile([0.25, 0.50, 0.75]))

    numeric_summary = numeric_profile(df_clean, analysis_numeric_cols)
    print("\n=== Detailed profile (mode, percentiles via np.percentile, skew, kurtosis) ===")
    display(numeric_summary.T)

    print("\n=== Potential outliers (IQR rule) - flagged only, nothing removed ===")
    outlier_view = numeric_summary[["lower_fence", "upper_fence", "outliers_low", "outliers_high", "outlier_pct"]]
    display(outlier_view[(outlier_view["outliers_low"] + outlier_view["outliers_high"]) > 0]
            if ((outlier_view["outliers_low"] + outlier_view["outliers_high"]) > 0).any() else "None flagged.")

    print("\n=== Unusual values worth checking against clinical/domain ranges ===")
    unusual = numeric_summary[["zeros", "negatives", "inf_values"]]
    display(unusual[(unusual > 0).any(axis=1)] if (unusual > 0).any().any() else "No zeros, negatives or infinities.")

    plot_cols = analysis_numeric_cols[:MAX_PLOTS]
    ncols = 3
    nrows = int(np.ceil(len(plot_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, c in zip(axes.ravel(), plot_cols):
        data = num_df[c].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(data, bins=min(30, max(5, int(np.sqrt(len(data))))) if len(data) else 5)
        sk = numeric_summary.loc[c, "skew"]
        ax.set_title(f"{c}\nskew={sk:.2f}" if pd.notna(sk) else c, fontsize=9)
    for ax in axes.ravel()[len(plot_cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    if len(analysis_numeric_cols) > MAX_PLOTS:
        print(f"Plotted first {MAX_PLOTS} of {len(analysis_numeric_cols)} numeric columns (raise MAX_PLOTS to see more).")

    # ---- Register issues ----
    for c, r in numeric_summary.iterrows():
        if r.get("count", 0) == 0:
            continue
        n_out = int(r["outliers_low"] + r["outliers_high"])
        if n_out:
            add_issue("Potential outliers (IQR)", c, f"{n_out:,} values ({r['outlier_pct']:.1f}%) outside [{r['lower_fence']:.4g}, {r['upper_fence']:.4g}].",
                      "Review against clinical/domain ranges; keep unless proven erroneous.",
                      "Medium Priority" if r["outlier_pct"] > 5 else "Low Priority")
        if r["negatives"] > 0:
            add_issue("Negative values", c, f"{int(r['negatives']):,} negative values.",
                      "Verify negatives are valid for this variable.", "Medium Priority")
        if r["inf_values"] > 0:
            add_issue("Infinite values", c, f"{int(r['inf_values']):,} infinite values.", "Trace source of infinities.", "High Priority")
        if pd.notna(r["skew"]) and abs(r["skew"]) > 1:
            add_issue("Highly skewed distribution", c, f"Skewness = {r['skew']:.2f}.",
                      "Consider robust statistics or transformations at modelling time.", "Low Priority")

# %% [markdown]
# ## 10. Categorical Data Analysis
# **Purpose:** for every categorical/boolean column report unique values, frequencies, dominance/imbalance, and *possible* inconsistent spellings.
# **Expected output:** per-column frequency table plus flags.
# **Look for:** variants like `"Male"/"male "/"M"`, very rare categories, one category dominating (>90%). Variant groups and fuzzy matches are **candidates only** - confirm they mean the same thing before merging. Sensitive/identifier columns are skipped.

# %%
def normalize_category(value):
    """Key used only to *detect* variants: case-folded, punctuation/space-collapsed."""
    if not isinstance(value, str):
        return value
    return re.sub(r"[\W_]+", " ", value.casefold()).strip()


def analyze_categorical_column(frame, col, top_n=10):
    s = frame[col]
    non_null = s.dropna()
    n = len(non_null)
    result = {"column": col, "n_unique": int(non_null.nunique()), "variant_groups": 0,
              "fuzzy_pairs": 0, "dominant_share": np.nan, "rare_categories": 0}
    print(f"\n--- {col} --- unique={result['n_unique']}, non-null={n:,}, missing={int(s.isna().sum()):,}")
    if n == 0:
        return result
    counts = s.value_counts(dropna=False)
    table = pd.DataFrame({"count": counts, "percent": counts / len(s) * 100}).head(top_n)
    display(table)
    if result["n_unique"] <= 15:
        print("unique():", list(non_null.unique()))

    shares = non_null.value_counts(normalize=True)
    result["dominant_share"] = float(shares.iloc[0])
    result["rare_categories"] = int((shares < 0.01).sum())
    if result["dominant_share"] >= 0.9 and result["n_unique"] > 1:
        print(f"  [!] Imbalance: '{shares.index[0]}' makes up {result['dominant_share']:.1%} of non-null values.")

    # Variants that differ only by case / punctuation / spacing
    frame_norm = pd.DataFrame({"raw": non_null.astype(str), "norm": non_null.map(normalize_category).astype(str)})
    variants = frame_norm.groupby("norm")["raw"].agg(lambda x: sorted(set(x)))
    variants = variants[variants.map(len) > 1]
    result["variant_groups"] = len(variants)
    if len(variants):
        print("  [!] Possible inconsistent spellings (same after case/punctuation normalisation):")
        for key, vals in variants.head(10).items():
            print("     ", vals)

    # Fuzzy near-matches among distinct normalised values (small cardinality only)
    uniques = sorted(set(frame_norm["norm"]))
    if 1 < len(uniques) <= 150:
        pairs = set()
        for u in uniques:
            for m in difflib.get_close_matches(u, [x for x in uniques if x != u], n=3, cutoff=0.88):
                pairs.add(tuple(sorted((u, m))))
        result["fuzzy_pairs"] = len(pairs)
        if pairs:
            print("  [?] Similar-looking categories (verify before merging):", sorted(pairs)[:10])
    return result


cat_targets = [c for c in categorical_cols + boolean_cols if c not in sensitive_cols and c not in identifier_cols]
cat_results = {}
if not cat_targets:
    print("No categorical columns detected - skipping.")
else:
    for c in cat_targets[:30]:
        cat_results[c] = analyze_categorical_column(df_clean, c)
    if len(cat_targets) > 30:
        print(f"\nShowing first 30 of {len(cat_targets)} categorical columns.")

# ---- Register issues ----
for c, r in cat_results.items():
    if r["variant_groups"]:
        add_issue("Inconsistent category spellings", c, f"{r['variant_groups']} group(s) of values differ only by case/punctuation/spacing.",
                  "Standardise after confirming they are the same concept.", "Medium Priority")
    if r["fuzzy_pairs"]:
        add_issue("Similar-looking categories", c, f"{r['fuzzy_pairs']} pair(s) of near-identical labels.",
                  "Manually review; possible typos.", "Low Priority")
    if r["dominant_share"] >= 0.9 and r["n_unique"] > 1:
        add_issue("Category imbalance", c, f"Top category = {r['dominant_share']:.1%}.",
                  "Keep in mind for sampling/evaluation later.", "Low Priority")

# %% [markdown]
# ## 11. Date & Time Analysis
# **Purpose:** check date columns (true datetime or text that looks like dates), count invalid values, find earliest/latest, view temporal distribution.
# **Expected output:** per-column status, range, invalid count, year/month/weekday distribution.
# **Look for:** `skipped_ambiguous` - dates like `03/04/2024` could be 3 April or 4 March, so **no guess is made** until you set `DATE_DAYFIRST` in Section 1. Timezone markers are reported but timezones are never assumed or converted. Future dates are flagged. For sensitive date columns (e.g. birth dates) only counts are shown, not ranges.

# %%
AMBIG_RE = re.compile(r"^\s*(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})")
TZ_RE = re.compile(r"\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?\s*(?:Z|[+-]\d{2}:?\d{2}|UTC|GMT|IST)\s*$")


def analyze_date_column(series, dayfirst=None):
    res = {"status": "ok", "n_present": 0, "n_parsed": 0, "n_invalid": 0, "ambiguous": 0,
           "tz_marked": 0, "parsed": None, "min": None, "max": None, "future": 0}
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = series
        res["n_present"] = res["n_parsed"] = int(series.notna().sum())
    else:
        text = series.astype("string").str.strip()
        present = series.notna() & text.ne("").fillna(False)
        res["n_present"] = int(present.sum())
        strs = text[present]
        parts = strs.str.extract(AMBIG_RE)
        a, b = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
        res["ambiguous"] = int(((a <= 12) & (b <= 12) & (a != b)).sum())
        res["tz_marked"] = int(strs.str.contains(TZ_RE).sum())
        if res["ambiguous"] > 0 and dayfirst is None:
            res["status"] = "skipped_ambiguous"
            return res
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                parsed = pd.to_datetime(series, errors="coerce", dayfirst=bool(dayfirst), format="mixed")
            except (ValueError, TypeError):
                parsed = pd.to_datetime(series, errors="coerce", dayfirst=bool(dayfirst))
        parsed = parsed.where(present)                    # blanks stay missing, not "invalid"
        res["n_parsed"] = int(parsed.notna().sum())
        res["n_invalid"] = res["n_present"] - res["n_parsed"]
        if res["n_invalid"] > 0:
            res["status"] = "has_invalid_values"
    res["parsed"] = parsed
    valid = parsed.dropna()
    if len(valid):
        res["min"], res["max"] = valid.min(), valid.max()
        try:
            res["future"] = int((valid > pd.Timestamp.now(tz=valid.dt.tz)).sum())
        except Exception:
            res["future"] = 0
    return res


date_results = {}
if not datetime_cols:
    print("No date/time columns detected - skipping.")
else:
    for c in datetime_cols:
        r = analyze_date_column(df_clean[c], DATE_DAYFIRST)
        date_results[c] = r
        print(f"\n--- {c} --- status: {r['status']}")
        print(f"  present values: {r['n_present']:,} | parsed: {r['n_parsed']:,} | invalid: {r['n_invalid']:,}")
        if r["ambiguous"]:
            print(f"  [!] {r['ambiguous']:,} values are day/month-ambiguous (e.g. 03/04/2024). "
                  + ("Set DATE_DAYFIRST to True/False once confirmed." if DATE_DAYFIRST is None else f"Parsed with dayfirst={DATE_DAYFIRST}."))
        if r["tz_marked"]:
            print(f"  [i] {r['tz_marked']:,} values carry timezone markers; no timezone conversion is applied.")
        if r["parsed"] is None:
            continue
        valid = r["parsed"].dropna()
        if is_sensitive(c):
            print("  (sensitive date column: range and distribution not displayed)")
            continue
        if len(valid):
            print(f"  earliest: {r['min']} | latest: {r['max']} | future-dated: {r['future']:,}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                by_year = valid.dt.year.value_counts().sort_index()
                by_month = valid.dt.to_period("M").value_counts().sort_index()
            print("  records per year:", by_year.to_dict())
            print("  records per weekday:", valid.dt.day_name().value_counts().to_dict())
            series_to_plot = by_month if 2 <= len(by_month) <= 120 else by_year
            if len(series_to_plot) >= 2:
                fig, ax = plt.subplots(figsize=(8, 2.8))
                ax.plot(series_to_plot.index.astype(str), series_to_plot.values, marker="o")
                ax.set_title(f"{c}: records over time")
                ax.tick_params(axis="x", rotation=60, labelsize=7)
                plt.tight_layout()
                plt.show()
        if r["status"] == "has_invalid_values" and not is_sensitive(c):
            text = df_clean[c].astype("string").str.strip()
            bad = text[r["parsed"].isna() & text.notna() & text.ne("")].value_counts().head(5)
            print("  most common unparseable values:", bad.to_dict())

# ---- Register issues ----
for c, r in date_results.items():
    if r["status"] == "skipped_ambiguous":
        add_issue("Ambiguous date format", c, f"{r['ambiguous']:,} values could be day-first or month-first; not converted.",
                  "Confirm format with data owner, then set DATE_DAYFIRST.", "High Priority")
    if r["n_invalid"]:
        add_issue("Invalid date values", c, f"{r['n_invalid']:,} values could not be parsed as dates.",
                  "Inspect and correct the unparseable entries.", "High Priority")
    if r["future"]:
        add_issue("Future-dated records", c, f"{r['future']:,} dates are later than today.",
                  "Check for typos or scheduled events.", "Medium Priority")

# %% [markdown]
# ## 12. Correlation & Relationship Analysis
# **Purpose:** correlation matrix/heatmap for numeric features, group-wise summaries, and cross-tabs of categorical variables.
# **Expected output:** Pearson and Spearman matrices, strongest pairs, group tables, cross-tabs. Groups/cells with fewer than `MIN_GROUP_SIZE` records are blanked to protect privacy.
# **Look for:** pairs with |r| >= 0.7 (possible redundancy), Pearson vs Spearman disagreement (non-linear or outlier-driven), and categories with very different group means.
# **Limitations of correlation:** it measures linear (Pearson) or rank (Spearman) association only; it does not show causation, can be driven by outliers or confounders, and says nothing about clinical meaning. Treat every pattern here as a hypothesis for review, not a finding.

# %%
corr_cols = [c for c in analysis_numeric_cols if df_clean[c].nunique(dropna=True) > 1][:MAX_CORR_COLS]


def top_pairs(corr_matrix, threshold=0.7):
    iu = np.triu_indices_from(corr_matrix.values, k=1)
    pairs = pd.DataFrame({"feature_1": corr_matrix.index[iu[0]], "feature_2": corr_matrix.columns[iu[1]],
                          "correlation": corr_matrix.values[iu]}).dropna()
    pairs = pairs[pairs["correlation"].abs() >= threshold]
    return pairs.reindex(pairs["correlation"].abs().sort_values(ascending=False).index)


if len(corr_cols) < 2:
    print("Need at least two non-constant numeric columns for correlation analysis.")
    pearson = spearman = pd.DataFrame()
else:
    num_float = df_clean[corr_cols].apply(pd.to_numeric, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)
    pearson, spearman = num_float.corr(method="pearson"), num_float.corr(method="spearman")

    for name, cm in [("Pearson", pearson), ("Spearman", spearman)]:
        fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(cm)), max(4, 0.5 * len(cm))))
        if HAS_SEABORN:
            sns.heatmap(cm, annot=len(cm) <= 15, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0, ax=ax)
        else:
            im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cm)))
            ax.set_xticklabels(cm.columns, rotation=90)
            ax.set_yticks(range(len(cm)))
            ax.set_yticklabels(cm.index)
            fig.colorbar(im, ax=ax)
        ax.set_title(f"{name} correlation")
        plt.tight_layout()
        plt.show()

    strong = top_pairs(pearson, 0.7)
    print("Strong Pearson correlations (|r| >= 0.7):")
    display(strong if len(strong) else "none")

# ---- Group-wise analysis (small groups suppressed) ----
group_candidates = [c for c in categorical_cols + boolean_cols
                    if c not in sensitive_cols and 2 <= df_clean[c].nunique(dropna=True) <= 8][:3]
value_cols = corr_cols[:5]
if group_candidates and value_cols:
    for gcol in group_candidates:
        sizes = df_clean.groupby(gcol, observed=True).size()
        table = df_clean.groupby(gcol, observed=True)[value_cols].agg(["count", "mean", "median"])
        small = sizes.index[sizes < MIN_GROUP_SIZE]
        if len(small):
            table.loc[small, :] = np.nan
        print(f"\nGroup-wise summary by '{gcol}' (groups < {MIN_GROUP_SIZE} rows blanked):")
        display(table)
else:
    print("\nNo suitable low-cardinality categorical + numeric combination for group-wise analysis.")

# ---- Cross-tabulations ----
xtab_cols = [c for c in categorical_cols + boolean_cols
             if c not in sensitive_cols and 2 <= df_clean[c].nunique(dropna=True) <= 10][:4]
shown = 0
for i in range(len(xtab_cols)):
    for j in range(i + 1, len(xtab_cols)):
        if shown >= 3:
            break
        a, b = xtab_cols[i], xtab_cols[j]
        ct = pd.crosstab(df_clean[a], df_clean[b])
        pct = pd.crosstab(df_clean[a], df_clean[b], normalize="index") * 100
        print(f"\nCross-tab: {a} x {b} (cells < {MIN_GROUP_SIZE} blanked)")
        display(ct.where(ct >= MIN_GROUP_SIZE))
        display(pct.where(ct >= MIN_GROUP_SIZE).round(1))
        shown += 1

# %% [markdown]
# ## 13. Data Quality Report
# **Purpose:** consolidate every issue found above into one prioritised table.
# **Expected output:** table with Issue / Affected Column / Observation / Recommended Action / Priority, plus counts per priority.
# **Look for:** work through *High Priority* first. This is an automated screen - it cannot judge clinical validity, so domain review is still required.

# %%
if ISSUES:
    quality_report = pd.DataFrame(list(ISSUES.values()))
    quality_report["_order"] = quality_report["Priority"].map(PRIORITY_ORDER)
    quality_report = (quality_report.sort_values(["_order", "Issue", "Affected Column"])
                      .drop(columns="_order").reset_index(drop=True))
    print("Issues by priority:")
    display(quality_report["Priority"].value_counts().reindex(list(PRIORITY_ORDER)).fillna(0).astype(int).rename("count").to_frame())
    with pd.option_context("display.max_rows", 500, "display.max_colwidth", 110):
        display(quality_report)
else:
    quality_report = pd.DataFrame(columns=["Issue", "Affected Column", "Observation", "Recommended Action", "Priority"])
    print("The automated checks flagged no issues (manual review is still required).")

# %% [markdown]
# ## 14. Exporting Cleaned Dataset
# **Purpose:** write a *separate* cleaned file. The original `df` is untouched. Standardised headers are always applied; every other transformation is a switch you control below.
# **Defaults:** whitespace stripping and *unambiguous, fully parseable* date conversion are ON (both lossless in meaning). Numeric conversion, placeholder->NaN, duplicate dropping and dropping sensitive columns are **OFF** because they need human approval.
# **Expected output:** a log of what was changed, and three files: cleaned data, header mapping, quality report.
# **Look for:** the transformation log - if it is empty, only headers changed. **The export still contains any sensitive columns** unless you switch `drop_sensitive_columns` on; de-identify before sharing.

# %%
OUTPUT_DIR = Path(".")
CLEANED_CSV = "sanjeevani_cleaned_dataset.csv"
MAPPING_CSV = "column_name_mapping.csv"
QUALITY_CSV = "data_quality_report.csv"

CLEANING_OPTIONS = {
    "strip_text_whitespace": True,        # trims leading/trailing spaces in text values
    "convert_unambiguous_dates": True,    # only columns with status 'ok' and zero invalid values
    "convert_numeric_strings": False,     # only columns where 100% of non-null values parse
    "placeholders_to_nan": False,         # NA / ? / unknown ... -> NaN (needs domain approval)
    "drop_exact_duplicates": False,       # needs approval; see Section 8
    "drop_sensitive_columns": False,      # drops columns masked by the privacy heuristic
}

df_final = df_clean.copy()
transform_log = [{"step": "standardize_headers", "detail": f"{int(column_mapping['changed'].sum())} headers changed", "rows_affected": 0}]

if CLEANING_OPTIONS["strip_text_whitespace"]:
    for c in df_final.columns:
        if pd.api.types.infer_dtype(df_final[c], skipna=True) == "string":
            stripped = df_final[c].str.strip()
            changed = int((stripped.ne(df_final[c]) & df_final[c].notna()).sum())
            if changed:
                df_final[c] = stripped
                transform_log.append({"step": "strip_text_whitespace", "detail": c, "rows_affected": changed})

if CLEANING_OPTIONS["convert_unambiguous_dates"]:
    for c, r in date_results.items():
        if r["status"] == "ok" and r["n_invalid"] == 0 and r["parsed"] is not None and not pd.api.types.is_datetime64_any_dtype(df_clean[c]):
            blanks = int((df_final[c].notna() & r["parsed"].isna()).sum())
            df_final[c] = r["parsed"]
            transform_log.append({"step": "convert_to_datetime", "detail": f"{c} (blank strings -> NaT: {blanks})",
                                  "rows_affected": int(r["n_parsed"])})

if CLEANING_OPTIONS["convert_numeric_strings"]:
    for c in numeric_like_string_cols:
        if c in sensitive_cols:
            continue
        text = df_final[c].dropna().astype(str).str.replace(",", "", regex=False).str.strip()
        text = text[text != ""]
        converted = pd.to_numeric(text, errors="coerce")
        if len(text) and converted.notna().all():
            df_final[c] = pd.to_numeric(df_final[c].astype("string").str.replace(",", "", regex=False).str.strip(), errors="coerce")
            transform_log.append({"step": "convert_to_numeric", "detail": c, "rows_affected": int(converted.notna().sum())})
        else:
            print(f"Skipped numeric conversion for '{c}': some values do not parse.")

if CLEANING_OPTIONS["placeholders_to_nan"]:
    for c in placeholder_counts:
        mask = df_final[c].astype("string").str.strip().str.lower().isin(PLACEHOLDER_STRINGS).fillna(False)
        if mask.any():
            df_final.loc[mask, c] = np.nan
            transform_log.append({"step": "placeholders_to_nan", "detail": c, "rows_affected": int(mask.sum())})

if CLEANING_OPTIONS["drop_exact_duplicates"]:
    before = len(df_final)
    df_final = df_final[~safe_duplicated(df_final)]
    transform_log.append({"step": "drop_exact_duplicates", "detail": "all columns", "rows_affected": before - len(df_final)})

if CLEANING_OPTIONS["drop_sensitive_columns"]:
    df_final = df_final.drop(columns=sensitive_cols)
    transform_log.append({"step": "drop_sensitive_columns", "detail": join_cols(sensitive_cols), "rows_affected": 0})

print("=== Transformation log ===")
display(pd.DataFrame(transform_log))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUTPUT_DIR / CLEANED_CSV, index=False, encoding="utf-8")
column_mapping.to_csv(OUTPUT_DIR / MAPPING_CSV, index=False, encoding="utf-8")
quality_report.to_csv(OUTPUT_DIR / QUALITY_CSV, index=False, encoding="utf-8")
print(f"\nSaved: {OUTPUT_DIR / CLEANED_CSV}, {OUTPUT_DIR / MAPPING_CSV}, {OUTPUT_DIR / QUALITY_CSV}")
if sensitive_cols and not CLEANING_OPTIONS["drop_sensitive_columns"]:
    print(f"[Privacy] Exported file still contains potentially sensitive columns: {join_cols(sensitive_cols)}")

# %% [markdown]
# ## 15. Final Verification
# **Purpose:** confirm the cleaned dataset and the saved file are what you expect, and that the original was never modified.
# **Expected output:** shape, headers, dtypes, remaining missing values and duplicates, round-trip check of the CSV.
# **Look for:** any `FAIL`. Remaining missing values/duplicates are expected unless you enabled the matching options - they are reported, not hidden. Datetime columns read back from CSV come in as text; that is normal for CSV.

# %%
print(f"Final dimensions: {df_final.shape[0]:,} rows x {df_final.shape[1]:,} columns (original: {ORIGINAL_SHAPE[0]:,} x {ORIGINAL_SHAPE[1]:,})")
print("\nFinal column names:")
print(list(df_final.columns))
print("\nFinal data types:")
display(df_final.dtypes.astype(str).rename("dtype").to_frame())

remaining_missing = missing_summary_table(df_final)
print(f"\nRemaining missing cells: {int(df_final.isna().sum().sum()):,}")
display(remaining_missing[remaining_missing["Missing Count"] > 0].head(20)
        if (remaining_missing["Missing Count"] > 0).any() else "No missing values remain.")
print(f"Remaining exact duplicate rows: {int(safe_duplicated(df_final).sum()):,}")

# Round-trip check on the exported file
reloaded = pd.read_csv(OUTPUT_DIR / CLEANED_CSV)
try:
    current_hash = int(pd.util.hash_pandas_object(df.astype(str), index=True).sum())
except Exception:
    current_hash = None

final_checks = {
    "exported file has same shape as df_final": reloaded.shape == df_final.shape,
    "exported headers match df_final": list(reloaded.columns) == list(df_final.columns),
    "final headers unique": df_final.columns.is_unique,
    "final headers valid snake_case": all(SNAKE_RE.match(c) for c in df_final.columns),
    "original df shape unchanged": df.shape == ORIGINAL_SHAPE,
    "original df headers unchanged": list(df.columns) == ORIGINAL_COLUMNS,
    "original df content unchanged": (ORIGINAL_HASH is None or current_hash is None or ORIGINAL_HASH == current_hash),
}
print()
for label, ok in final_checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
