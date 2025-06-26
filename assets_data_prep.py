import pandas as pd
import numpy as np

def prepare_data(df: pd.DataFrame, dataset_type: str) -> pd.DataFrame:
    if dataset_type == "train":
        df = df.dropna(subset=["price"]).copy()
    else:
        df = df.copy()

    # טיפול בנתונים חסרים והמרות

    df["room_num"] = pd.to_numeric(df["room_num"], errors="coerce").fillna(df["room_num"].median())
    df["area"] = pd.to_numeric(df["area"], errors="coerce").fillna(df["area"].median())

    df["floor"] = pd.to_numeric(
        df["floor"].astype(str).str.extract(r'(\d+)')[0],
        errors="coerce"
    ).fillna(-1)

    df["total_floors"] = pd.to_numeric(
        df["total_floors"].astype(str).str.extract(r'(\d+)')[0],
        errors="coerce"
    ).fillna(-1)

    df["monthly_arnona"] = pd.to_numeric(df["monthly_arnona"], errors="coerce").fillna(0)
    df["building_tax"] = pd.to_numeric(df["building_tax"], errors="coerce").fillna(0)
    df["garden_area"] = pd.to_numeric(df["garden_area"], errors="coerce").fillna(0)
    df["days_to_enter"] = pd.to_numeric(df["days_to_enter"], errors="coerce").fillna(0)
    df["num_of_payments"] = pd.to_numeric(df["num_of_payments"], errors="coerce").fillna(1)
    df["num_of_images"] = pd.to_numeric(df["num_of_images"], errors="coerce").fillna(0)
    df["distance_from_center"] = pd.to_numeric(df["distance_from_center"], errors="coerce").fillna(df["distance_from_center"].mean())

    # עמודות בינאריות
    bool_cols = [
        "has_parking", "has_storage", "elevator", "ac", "handicap",
        "has_bars", "has_safe_room", "has_balcony", "is_furnished", "is_renovated"
    ]
    for col in bool_cols:
        df[col] = df[col].fillna(0).astype(int)

    # קידוד קטגוריות
    for col in ["neighborhood", "property_type"]:
        df[col] = df[col].fillna("unknown").astype("category").cat.codes

    # מאפיינים נבחרים
    selected_features = [
        "room_num", "area", "floor", "total_floors", "monthly_arnona", "building_tax",
        "garden_area", "days_to_enter", "num_of_payments", "num_of_images", "distance_from_center",
        "has_parking", "has_storage", "elevator", "ac", "handicap",
        "has_bars", "has_safe_room", "has_balcony", "is_furnished", "is_renovated",
        "neighborhood", "property_type"
    ]

    if "price" in df.columns:
        selected_features.append("price")

    return df[selected_features]
