"""
Patient Health Records - Cleaning Pipeline (sklearn version)
==============================================================

Converts the manual pandas cleaning notebook (July8_Outliers.ipynb) into a
proper sklearn Pipeline: FunctionTransformer (parsing/mapping) -> ColumnTransformer
(impute/clip/scale/encode per column) -> Pipeline.

IMPORTANT DESIGN NOTE (read this before using):
------------------------------------------------
The original notebook DROPPED rows for outliers (e.g. `df[(df["Age"]>=0) & ...]`,
z-score filtering on Heart_Rate). A sklearn transformer inside a ColumnTransformer
is NOT allowed to change the number of rows (every column's output has to line up
in the final matrix), so real row-dropping cannot live inside this pipeline.

Fix used here: row-dropping outlier logic has been converted to CLIPPING
(same thing the notebook itself did for Age at the very end with `.clip(lower,
upper)`). So Age and Heart_Rate get clipped to a valid range AND to their IQR
bounds, instead of having rows removed. This keeps row count constant so the
pipeline is valid and reusable on new/incoming data (which is the whole point
of putting this in a pipeline).

If you actually want to DROP rows (not clip), do that once on the raw
dataframe BEFORE this pipeline (I've left a helper function `drop_hard_outliers`
at the bottom for that, run manually, not part of the sklearn Pipeline).
"""

import re
import numpy as np
import pandas as pd
from word2number import w2n

from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer


# ---------------------------------------------------------------------------
# 1) Raw parsing functions - same logic as the notebook, just cleaned up
#    so they return np.nan instead of None (sklearn/pandas plays nicer with nan)
# ---------------------------------------------------------------------------

def parse_bp(x):
    """'120 over 80' / '120-80' / '120/80' -> 120.08 style single number"""
    if pd.isna(x):
        return np.nan
    x = str(x).lower().replace(" over ", "/").replace(" - ", "/").replace(" ", "")
    nums = re.findall(r"\d+", x)
    if len(nums) >= 2:
        return int(nums[0]) + int(nums[1]) / 1000
    return np.nan


def word_or_int(x):
    """Handles Age / Heart_Rate / Follow_Up: '45', 'forty five', etc."""
    try:
        return int(x)
    except (ValueError, TypeError):
        try:
            return w2n.word_to_num(x)
        except Exception:
            return np.nan


def extract_bmi(x):
    m = re.search(r"(\d+\.?\d*)", str(x))
    return float(m.group(1)) if m else np.nan


def parse_cholesterol(x):
    try:
        return int(x)
    except (ValueError, TypeError):
        return 240.0 if x == "high" else 50.0


def parse_date_to_ordinal(x):
    """Parses the date then converts to an ordinal int, because scalers/
    models can't work with raw datetime objects. Returns np.nan if unparsable."""
    x = str(x).strip()
    if x.isdigit() and len(x) == 8:
        d = pd.to_datetime(x, format="%Y%m%d", errors="coerce")
    else:
        d = pd.to_datetime(x, errors="coerce")
    return d.toordinal() if pd.notna(d) else np.nan


GENDER_MAP = {"FEMALE": "F", "F": "F", "male": "M", "M": "M"}
DIABETIC_MAP = {"Y": "Yes", "y": "Yes", "yes": "Yes", "YES": "Yes",
                "N": "No", "n": "No", "no": "No", "NO": "No"}
SMOKER_MAP = {"yes": 1, "No": 0, "Former": 2, "EX-smoker": 2}
CITY_MAP = {"NWEYROK": "NEWYORK", "LA": "LOSANGELES", "NY": "NEWYORK"}


# ---------------------------------------------------------------------------
# 2) Generic wrappers so any plain function/mapping can become a
#    FunctionTransformer that works on ONE column inside a ColumnTransformer.
#    ColumnTransformer always hands each transformer a 2D DataFrame slice,
#    so we grab column 0, run .apply(), and hand back a 2D frame.
# ---------------------------------------------------------------------------

def _as_series(X):
    return X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.ravel(X))


def column_func_transformer(func):
    return FunctionTransformer(
        lambda X: _as_series(X).apply(func).to_frame(),
        feature_names_out="one-to-one",
    )


def column_map_transformer(mapping):
    return FunctionTransformer(
        lambda X: _as_series(X).replace(mapping).to_frame(),
        feature_names_out="one-to-one",
    )


# ---------------------------------------------------------------------------
# 3) Custom fit/transform transformers (need to "learn" something from
#    training data, so a plain FunctionTransformer isn't enough)
# ---------------------------------------------------------------------------

class RangeClipper(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Hard physical-range clip, e.g. Age must be 0-120, Heart_Rate 30-220.
    No fitting needed - bounds are fixed domain knowledge."""

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        col = _as_series(X).astype(float)
        return col.clip(self.low, self.high).to_frame()


class IQRClipper(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Learns Q1/Q3/IQR bounds on the TRAINING data (fit), then clips any
    data (train or new/test) to those bounds. This replaces the notebook's
    row-dropping IQR outlier logic."""

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        col = _as_series(X).astype(float)
        q1, q3 = col.quantile(0.25), col.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        col = _as_series(X).astype(float)
        return col.clip(self.lower_, self.upper_).to_frame()


class PatientIDFiller(OneToOneFeatureMixin, BaseEstimator, TransformerMixin):
    """Fills missing Patient_ID with generated 'ID_0', 'ID_1', ... like the
    notebook did. Not really an ML feature, kept only so nothing is lost."""

    def fit(self, X, y=None):
        self.n_features_in_ = 1
        return self

    def transform(self, X):
        col = _as_series(X).copy()
        mask = col.isna()
        col.loc[mask] = [f"ID_{i}" for i in range(mask.sum())]
        return col.to_frame()


# ---------------------------------------------------------------------------
# 4) Per-column pipelines: parse/map -> impute -> (clip) -> scale/encode
# ---------------------------------------------------------------------------

blood_pressure_pipe = Pipeline([
    ("parse", column_func_transformer(parse_bp)),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("scale", StandardScaler()),
])

age_pipe = Pipeline([
    ("parse", column_func_transformer(word_or_int)),
    ("impute", SimpleImputer(strategy="median")),
    ("range_clip", RangeClipper(0, 120)),
    ("iqr_clip", IQRClipper()),
    ("scale", StandardScaler()),
])

bmi_pipe = Pipeline([
    ("parse", column_func_transformer(extract_bmi)),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

heart_rate_pipe = Pipeline([
    ("parse", column_func_transformer(word_or_int)),
    ("impute", SimpleImputer(strategy="median")),
    ("range_clip", RangeClipper(30, 220)),
    ("iqr_clip", IQRClipper()),
    ("scale", StandardScaler()),
])

cholesterol_pipe = Pipeline([
    ("parse", column_func_transformer(parse_cholesterol)),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

follow_up_pipe = Pipeline([
    ("parse", column_func_transformer(word_or_int)),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

diabetic_pipe = Pipeline([
    ("map", column_map_transformer(DIABETIC_MAP)),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder()),
])

smoker_pipe = Pipeline([
    ("map", column_map_transformer(SMOKER_MAP)),
    ("impute", SimpleImputer(strategy="most_frequent")),
])

gender_pipe = Pipeline([
    ("map", column_map_transformer(GENDER_MAP)),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder()),
])

city_pipe = Pipeline([
    ("map", column_map_transformer(CITY_MAP)),
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder()),
])

has_disease_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
])

last_visit_date_pipe = Pipeline([
    ("parse", column_func_transformer(parse_date_to_ordinal)),
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

patient_id_pipe = Pipeline([
    ("fill_id", PatientIDFiller()),
])

# Free-text / code columns: notebook only filled NaNs with mode, no encoding
# (too high-cardinality to one-hot / ordinal encode sensibly). Kept as-is.
text_impute_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
])


# ---------------------------------------------------------------------------
# 5) ColumnTransformer - stitches every column pipeline together
# ---------------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("blood_pressure", blood_pressure_pipe, ["Blood_Pressure"]),
        ("age", age_pipe, ["Age"]),
        ("bmi", bmi_pipe, ["BMI"]),
        ("heart_rate", heart_rate_pipe, ["Heart_Rate"]),
        ("cholesterol", cholesterol_pipe, ["Cholesterol_Level"]),
        ("follow_up", follow_up_pipe, ["Follow_Up"]),
        ("diabetic", diabetic_pipe, ["Diabetic"]),
        ("smoker", smoker_pipe, ["Smoker"]),
        ("gender", gender_pipe, ["Gender"]),
        ("city", city_pipe, ["City"]),
        ("has_disease", has_disease_pipe, ["Has_Disease"]),
        ("last_visit_date", last_visit_date_pipe, ["Last_Visit_Date"]),
        ("patient_id", patient_id_pipe, ["Patient_ID"]),
        ("medications", text_impute_pipe, ["Medications"]),
        ("notes", text_impute_pipe, ["Notes"]),
        ("diagnosis_code", text_impute_pipe, ["Diagnosis_Code"]),
    ],
    remainder="drop",
)

full_pipeline = Pipeline([
    ("preprocessing", preprocessor),
])


# ---------------------------------------------------------------------------
# 6) Optional: one-time HARD row-drop cleaning (run manually, NOT part of the
#    sklearn Pipeline above, for the reason explained at the top of the file)
# ---------------------------------------------------------------------------

def drop_hard_outliers(df):
    df = df.copy()
    df = df[(df["Age"] >= 0) & (df["Age"] <= 120)]
    df = df[(df["Heart_Rate"] >= 30) & (df["Heart_Rate"] <= 220)]
    return df


# ---------------------------------------------------------------------------
# 7) Usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = pd.read_csv(
        "Patient_Health_Records_Raw.csv",
        na_values=["unknown", "Unknown"],
    )

    # feature columns going into the pipeline
    feature_cols = [
        "Blood_Pressure", "Age", "BMI", "Heart_Rate", "Cholesterol_Level",
        "Follow_Up", "Diabetic", "Smoker", "Gender", "City", "Has_Disease",
        "Last_Visit_Date", "Patient_ID", "Medications", "Notes", "Diagnosis_Code",
    ]
    X = raw[feature_cols]

    X_transformed = full_pipeline.fit_transform(X)
    print("Output shape:", X_transformed.shape)

    # get readable column names back out
    out_cols = full_pipeline.named_steps["preprocessing"].get_feature_names_out()
    result_df = pd.DataFrame(X_transformed, columns=out_cols)
    print(result_df.head())
