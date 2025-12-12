from dataclasses import dataclass
from typing import List, Optional, Dict
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@dataclass
class CleanerConfig:
    drop_duplicates: bool = True
    duplicate_subset: Optional[List[str]] = None

    handle_missing: str = "simple"          # "simple", "drop", "none"
    numeric_strategy: str = "median"        # "mean" or "median"
    categorical_strategy: str = "mode"      # "mode" or "constant"
    constant_fill_value: str = "Unknown"

    standardize_columns: bool = True
    target_case: str = "snake"              # "snake" or "lower"

    encode_categories: str = "label"        # "label", "onehot", "none"
    max_unique_for_encoding: int = 30


def to_snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    name = name.lower()
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


class Cleaner:
    def __init__(self, config: CleanerConfig | None = None):
        self.config = config or CleanerConfig()
        self.numeric_fill_values_: Dict[str, float] = {}
        self.categorical_fill_values_: Dict[str, str] = {}
        self.label_encoders_: Dict[str, LabelEncoder] = {}
        self.fitted_: bool = False

    # ---------- internal helpers ----------

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.drop_duplicates:
            return df
        if self.config.duplicate_subset:
            return df.drop_duplicates(subset=self.config.duplicate_subset)
        return df.drop_duplicates()

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.standardize_columns:
            return df
        if self.config.target_case == "snake":
            return df.rename(columns=to_snake_case)
        if self.config.target_case == "lower":
            return df.rename(columns=lambda c: c.strip().lower())
        return df

    def _detect_categorical_for_encoding(self, df: pd.DataFrame) -> List[str]:
        # Only treat object and category dtypes as categorical
        cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
        return cat_cols


    def _fit_label_encoders(self, df: pd.DataFrame) -> None:
        cat_cols = self._detect_categorical_for_encoding(df)
        for col in cat_cols:
            le = LabelEncoder()
            series = df[col].astype(str).fillna("__MISSING__")
            le.fit(series)
            self.label_encoders_[col] = le

    def _apply_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, le in self.label_encoders_.items():
            if col in df.columns:
                series = df[col].astype(str).fillna("__MISSING__")
                df[col] = le.transform(series)
        return df

    def _apply_onehot(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = self._detect_categorical_for_encoding(df)
        cat_cols = [c for c in cat_cols if c in df.columns]
        if not cat_cols:
            return df
        return pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # ---------- public API ----------

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._standardize_columns(df)

        if self.config.handle_missing == "simple":
            num_cols = df.select_dtypes(include="number").columns
            for col in num_cols:
                series = df[col]
                if self.config.numeric_strategy == "mean":
                    value = float(series.mean())
                else:
                    value = float(series.median())
                self.numeric_fill_values_[col] = value

            cat_cols = df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                if self.config.categorical_strategy == "mode":
                    mode = df[col].mode(dropna=True)
                    if not mode.empty:
                        fill_value = mode.iloc[0]
                    else:
                        fill_value = self.config.constant_fill_value
                else:
                    fill_value = self.config.constant_fill_value
                self.categorical_fill_values_[col] = str(fill_value)

        if self.config.encode_categories == "label":
            self._fit_label_encoders(df)

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform().")

        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._standardize_columns(df)

        if self.config.handle_missing == "simple":
            for col, value in self.numeric_fill_values_.items():
                if col in df.columns:
                    df[col] = df[col].fillna(value)
            for col, value in self.categorical_fill_values_.items():
                if col in df.columns:
                    df[col] = df[col].fillna(value)
        elif self.config.handle_missing == "drop":
            df = df.dropna()

        if self.config.encode_categories == "label":
            df = self._apply_label_encoding(df)
        elif self.config.encode_categories == "onehot":
            df = self._apply_onehot(df)

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
