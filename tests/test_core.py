import pandas as pd
from data_cleaner import Cleaner, CleanerConfig
def test_drop_duplicates_default():
    df = pd.DataFrame(
        {
            "A": [1, 1, 2],
            "B": ["x", "x", "y"],
        }
    )

    cleaner = Cleaner()
    df_clean = cleaner.fit_transform(df)

    assert len(df_clean) == 2


def test_simple_missing_imputation():
    df = pd.DataFrame(
        {
            "num": [1.0, None, 3.0],
            "cat": ["a", None, "a"],
        }
    )

    config = CleanerConfig(
        handle_missing="simple",
        numeric_strategy="median",
        categorical_strategy="mode",
    )
    cleaner = Cleaner(config=config)
    df_clean = cleaner.fit_transform(df)

    assert df_clean["num"].isna().sum() == 0
    assert df_clean["cat"].isna().sum() == 0


def test_standardize_columns_to_snake_case():
    df = pd.DataFrame(
        {
            "First Name": [1, 2],
            "Last-Name": [3, 4],
        }
    )

    cleaner = Cleaner()
    df_clean = cleaner.fit_transform(df)

    assert "first_name" in df_clean.columns
    assert "last_name" in df_clean.columns


def test_label_encoding():
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "red"],
        }
    )

    config = CleanerConfig(
        encode_categories="label",
    )
    cleaner = Cleaner(config=config)
    df_clean = cleaner.fit_transform(df)

    assert pd.api.types.is_integer_dtype(df_clean["color"]) or pd.api.types.is_numeric_dtype(
        df_clean["color"]
    )


def test_onehot_encoding():
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "red"],
        }
    )

    config = CleanerConfig(
        encode_categories="onehot",
    )
    cleaner = Cleaner(config=config)
    df_clean = cleaner.fit_transform(df)

    assert "color_red" in df_clean.columns or "color_blue" in df_clean.columns
    assert "color" not in df_clean.columns
