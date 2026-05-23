import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH    = 'data/survey_results_public.csv'
WINSOR_UPPER = 238_242
RANDOM_STATE = 42
TEST_SIZE    = 0.20

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
    'Something else':                                                                               np.nan,
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
    "I don't know":                                       np.nan,
}


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=FEATURES)


def filter_and_winsorize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['ConvertedCompYearly']).copy()
    df['ConvertedCompYearly'] = df['ConvertedCompYearly'].clip(upper=WINSOR_UPPER)
    return df


def make_target(df: pd.DataFrame) -> pd.Series:
    return np.log1p(df['ConvertedCompYearly'])


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['YearsCodePro'] = pd.to_numeric(
        df['YearsCodePro'].replace({'Less than 1 year': 0, 'More than 50 years': 51}),
        errors='coerce'
    )

    df['EdLevel'] = df['EdLevel'].map(EDLEVEL_MAP)
    df['OrgSize'] = df['OrgSize'].map(ORGSIZE_MAP)

    df['ICorPM'] = (df['ICorPM'] == 'People manager').astype(int)

    df['is_fulltime'] = df['Employment'].str.contains(
        'Employed, full-time', na=False
    ).astype(int)
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
    df_raw = load_raw(path)
    df = filter_and_winsorize(df_raw)
    y = make_target(df)
    df_encoded = encode_features(df)

    df_encoded = df_encoded.drop(columns=['ConvertedCompYearly'])
    df_encoded['_y'] = y
    df_encoded = df_encoded.dropna()
    y_clean = df_encoded.pop('_y')
    X = df_encoded

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    numeric_cols = ['YearsCodePro', 'WorkExp', 'EdLevel', 'OrgSize']
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}  |  y_test: {y_test.shape}")
    print(f"Features ({X_train.shape[1]}): {list(X_train.columns)}")

    return X_train, X_test, y_train, y_test, scaler, list(X_train.columns)


if __name__ == '__main__':
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    X_train, X_test, y_train, y_test, scaler, features = build_pipeline()
