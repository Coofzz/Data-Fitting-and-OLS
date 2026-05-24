"""
Data pipeline for Stack Overflow Developer Survey 2023.

Preprocessing steps (in order):
  1. Load raw CSV (selected columns only)
  2. Drop rows with no salary; Winsorize salary at IQR upper bound (~$238K)
  3. Log-transform target: y = log1p(salary)
  4. Encode features:
       YearsCodePro -> numeric (handle 'Less than 1 year', 'More than 50 years')
       EdLevel      -> ordinal int 1-6
       OrgSize      -> ordinal int 1-9
       ICorPM       -> binary 0/1
       Employment   -> is_fulltime flag, then drop original
       DevType      -> top-10 categories + 'Other'; one-hot encode
       Country      -> top-20 countries + 'Other'; one-hot encode
       RemoteWork   -> one-hot encode (3 categories)
  5. Drop remaining NaN rows; 80/20 train/test split (stratified by random seed)
  6. Standardize numeric columns (fit on train only, transform test)

Missing-value mechanism rationale:
  - ConvertedCompYearly (MNAR): respondents with very high/low salaries less
    likely to disclose -> listwise deletion (imputing target is invalid).
  - YearsCodePro / WorkExp / EdLevel (MAR): missingness correlates with
    employment status; median imputation implied by StandardScaler on encoded values.
  - ICorPM / OrgSize (MAR): skipped by freelancers/students; NaN rows dropped
    after encoding (treated as unknown category and excluded).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(_THIS_DIR, '..', 'data', 'survey_results_public.csv')
WINSOR_UPPER = 238_242
RANDOM_STATE = 42
TEST_SIZE    = 0.20
NUMERIC_COLS = ['YearsCodePro', 'WorkExp', 'EdLevel', 'OrgSize']

FEATURES = [
    'ConvertedCompYearly',
    'YearsCodePro', 'WorkExp',
    'EdLevel', 'DevType', 'OrgSize',
    'Country', 'RemoteWork', 'ICorPM',
    'Employment',
]

EDLEVEL_MAP = {
    'Primary/elementary school':                                                                    1,
    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)':          2,
    'Some college/university study without earning a degree':                                       3,
    'Associate degree (A.A., A.S., etc.)':                                                         3,
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)":                                               4,
    "Master's degree (M.A., M.S., M.Eng., MBA, etc.)":                                            5,
    'Professional degree (JD, MD, Ph.D, Ed.D, etc.)':                                             6,
}

ORGSIZE_MAP = {
    'Just me - I am a freelancer, sole proprietor, etc.': 1,
    '2 to 9 employees':                                   2,
    '10 to 19 employees':                                 3,
    '20 to 99 employees':                                 4,
    '100 to 499 employees':                               5,
    '500 to 999 employees':                               6,
    '1,000 to 4,999 employees':                           7,
    '5,000 to 9,999 employees':                           8,
    '10,000 or more employees':                           9,
}


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load selected columns from the survey CSV."""
    return pd.read_csv(path, usecols=FEATURES)


def winsorize_series(series: pd.Series, cap: float) -> pd.Series:
    """Clip values to [0, cap]. Lower=0 enforces non-negative salary."""
    return series.clip(lower=0, upper=cap)


def log1p_transform(series: pd.Series) -> pd.Series:
    """Apply log(1+x) transformation; returns a new Series."""
    return np.log1p(series)


def ordinal_encode(series: pd.Series, mapping: dict) -> pd.Series:
    """Map categorical labels to ordinal integers; unmapped values become NaN."""
    return series.map(mapping)


def filter_and_winsorize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without salary and Winsorize at WINSOR_UPPER."""
    df = df.dropna(subset=['ConvertedCompYearly']).copy()
    df['ConvertedCompYearly'] = winsorize_series(df['ConvertedCompYearly'], WINSOR_UPPER)
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    """Return log1p-transformed salary as regression target y."""
    return log1p_transform(df['ConvertedCompYearly'])


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature encoding steps described in the module docstring."""
    df = df.copy()

    df['YearsCodePro'] = pd.to_numeric(
        df['YearsCodePro'].replace({'Less than 1 year': 0, 'More than 50 years': 51}),
        errors='coerce'
    )

    df['EdLevel'] = ordinal_encode(df['EdLevel'], EDLEVEL_MAP)
    df['OrgSize'] = ordinal_encode(df['OrgSize'], ORGSIZE_MAP)

    df['ICorPM']    = (df['ICorPM'] == 'People manager').astype(float)
    df['is_fulltime'] = df['Employment'].str.contains('Employed, full-time', na=False).astype(int)
    df = df.drop(columns=['Employment'])

    df['DevType'] = df['DevType'].str.split(';').str[0].str.strip()
    top_dev = df['DevType'].value_counts().nlargest(10).index
    df['DevType'] = df['DevType'].where(df['DevType'].isin(top_dev), other='Other')

    top_countries = df['Country'].value_counts().nlargest(20).index
    df['Country'] = df['Country'].where(df['Country'].isin(top_countries), other='Other')

    df = pd.get_dummies(df, columns=['RemoteWork', 'Country', 'DevType'], drop_first=True)
    bool_cols = df.select_dtypes(include='bool').columns
    df[bool_cols] = df[bool_cols].astype(int)

    return df


def build_pipeline(path: str = DATA_PATH):
    """
    Full pipeline: load -> filter/winsorize -> log-transform -> encode -> split -> scale.

    Returns
    -------
    X_train, X_test : pd.DataFrame  (standardized numeric columns)
    y_train, y_test : pd.Series     (log1p salary)
    scaler          : StandardScaler fitted on train
    features        : list[str]     column names of X
    """
    df_raw     = load_raw(path)
    df         = filter_and_winsorize(df_raw)
    y          = make_target(df)
    df_encoded = encode_features(df)

    df_encoded = df_encoded.drop(columns=['ConvertedCompYearly'])
    df_encoded['_y'] = y.values
    df_encoded = df_encoded.dropna()
    y_clean    = df_encoded.pop('_y')
    X          = df_encoded

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train, X_test = X_train.copy(), X_test.copy()
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS]  = scaler.transform(X_test[NUMERIC_COLS])

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Features ({X_train.shape[1]}): {list(X_train.columns)[:6]} ...")
    return X_train, X_test, y_train, y_test, scaler, list(X_train.columns)


if __name__ == '__main__':
    build_pipeline()
