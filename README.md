
datacleaner-aditi is a lightweight Python package for automated data cleaning on pandas DataFrames. It handles duplicates, missing values, messy column names, and categorical encoding with a simple, configurable interface.​

Features
Remove duplicate rows with optional subset of columns.​

Handle missing values for numeric and categorical columns using common strategies (mean/median/mode or constant).​

Standardize column names to a consistent style (e.g., snake_case).​

Encode categorical variables using label or one‑hot encoding, ready for modeling.​

Installation
From a local clone of this repository (project root):
pip install -e .

This installs the package in editable mode so changes to the source code take effect immediately.​

Quick start
import pandas as pd
from datacleaner_aditi import Cleaner

# Example raw data
df = pd.DataFrame({
    "a": [1, 2, None],
    "b": ["x", "x", "y"],
})

cleaner = Cleaner()
df_clean = cleaner.fit_transform(df)
print(df_clean)

Example Output:
     a  b
0  1.0  1
1  2.0  1
2  1.5  2

Column a: missing value filled using a numeric strategy (e.g., median of non‑null values).​

Column b: categories encoded into integers (e.g., "x" → 1, "y" → 2).

API overview
from dataclasses import dataclass
from datacleaner_aditi import Cleaner

How it works (pipeline steps)
When you call fit_transform, the cleaner applies a sequence of common data‑preparation steps:​

Drop duplicates

Optionally uses duplicate_subset if specified.

Standardize column names

Converts headers to a consistent style (e.g., snake_case: Total Sales → total_sales).​

Handle missing values

Numeric columns: filled using mean or median.

Categorical columns: filled using mode or a constant like "Unknown".

Encode categorical columns

Label encoding: maps categories to integers.

One‑hot encoding: expands categories into dummy columns for models that need binary indicators.​

The fit step learns any required statistics (e.g., medians, modes, encoders) on the training data, and transform applies them to new data consistently.​

Running tests
From the project root:
pytest

This runs the test suite to verify behavior on duplicates, missing values, column normalization, and categorical encoding.​

Project structure
A typical layout for this package:
data_cleaner/                # repo root
├── pyproject.toml
├── README.md
├── src/
│   └── datacleaner_aditi/
│       ├── __init__.py
│       └── core.py
└── tests/
    └── test_core.py

This follows modern Python packaging recommendations (src layout, pyproject‑based builds).​

License
This project respects intellectual property and copyright.
Add your chosen license here (for example, MIT License) and include the corresponding LICENSE file in the repository.​

TestPyPl link: https://test.pypi.org/project/datacleaner-aditi/
