"""
F1 Race Points Prediction

This script builds a binary classification model to predict whether a driver
will score points in a Formula 1 race (finish in P1–P10) based on
pre-race information such as season, starting grid position, and qualifying
position.

It uses the "Formula 1 World Championship History (1950–2024)" dataset
structure, in particular:

- Race_Schedule.csv       : race metadata (raceId, year, circuit, etc.)
- Race_Results.csv        : final results per driver per race
- Qualifying_Results.csv  : qualifying results per driver per race

Expected columns:
- Race_Schedule:   raceId, year
- Race_Results:    raceId, driverId, constructorId, grid, positionOrder, points
- Qualifying_Results: raceId, driverId, position (qualifying position)

The dataset itself is not included in this repository.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

DATA_DIR = "data"  # directory where the CSV files are stored

RACE_SCHEDULE_CSV = f"{DATA_DIR}/Race_Schedule.csv"
RACE_RESULTS_CSV = f"{DATA_DIR}/Race_Results.csv"
QUALI_RESULTS_CSV = f"{DATA_DIR}/Qualifying_Results.csv"

# Column names in this dataset
RACE_ID_COL        = "raceId"
SEASON_COL         = "year"
DRIVER_ID_COL      = "driverId"
CONSTRUCTOR_ID_COL = "constructorId"
POINTS_COL         = "points"
FINISH_POS_COL     = "positionOrder"
GRID_COL_RACE      = "grid"          # starting grid (from Race_Results)
GRID_COL_QUALI     = "position"      # qualifying position (from Qualifying_Results)

TARGET_COL = "points_scored"


# -------------------------------------------------------------------
# Data loading and feature construction
# -------------------------------------------------------------------

def load_data():
    race_schedule = pd.read_csv(RACE_SCHEDULE_CSV)
    race_results = pd.read_csv(RACE_RESULTS_CSV)
    qualifying = pd.read_csv(QUALI_RESULTS_CSV)

    return race_schedule, race_results, qualifying


def build_modeling_table(
    race_schedule: pd.DataFrame,
    race_results: pd.DataFrame,
    qualifying: pd.DataFrame,
) -> pd.DataFrame:
    # Merge race results with schedule to add season/year
    schedule_subset = race_schedule[[RACE_ID_COL, SEASON_COL]].copy()

    base = race_results.merge(
        schedule_subset,
        on=RACE_ID_COL,
        how="left",
    )

    # Add qualifying position as an additional feature
    quali_subset = qualifying[[RACE_ID_COL, DRIVER_ID_COL, GRID_COL_QUALI]].copy()
    quali_subset = quali_subset.rename(columns={GRID_COL_QUALI: "quali_position"})

    df = base.merge(
        quali_subset,
        on=[RACE_ID_COL, DRIVER_ID_COL],
        how="left",
    )

    # Target: whether the driver scored points in the race
    df[TARGET_COL] = (df[POINTS_COL] > 0).astype(int)

    return df


def prepare_features(df: pd.DataFrame):
    # Simple baseline feature set: season, race grid, and qualifying position
    feature_cols = [SEASON_COL, GRID_COL_RACE, "quali_position"]

    df_model = df.dropna(subset=feature_cols + [TARGET_COL]).copy()

    X = df_model[feature_cols]
    y = df_model[TARGET_COL]

    return X, y, feature_cols


# -------------------------------------------------------------------
# Model training and evaluation
# -------------------------------------------------------------------

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size:  {X_test.shape[0]}")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    return model


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

if __name__ == "__main__":
    race_schedule, race_results, qualifying = load_data()

    df = build_modeling_table(race_schedule, race_results, qualifying)
    print("Modeling table shape:", df.shape)

    X, y, feature_cols = prepare_features(df)
    print("Feature columns:", feature_cols)
    print("Final dataset shape:", X.shape)

    model = train_and_evaluate(X, y)
