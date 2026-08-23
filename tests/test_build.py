import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype
from src.data.build import clean,fill_missing

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "practice_time": ["2026-08-23 12:30:00"],
        "ui_language": ["tr"],
        "learning_language": ["en"],
        "pos": ["NOUN"],
        "lemma": ["  apple  "],
        "surface_form": [" apples "],
        "grammar_tags": [None]
    })

def test_clean_strips_whitespace(sample_df):
    result = clean(sample_df)
    assert result["lemma"].iloc[0] == "apple"
    assert result["surface_form"].iloc[0] == "apples"

def test_clean_converts_practice_time_to_datetime(sample_df):
    result = clean(sample_df)
    assert is_datetime64_any_dtype(result["practice_time"])

def test_clean_converts_categorical_columns(sample_df):
    result = clean(sample_df)
    assert result["ui_language"].dtype.name == "category"
    assert result["learning_language"].dtype.name == "category"
    assert result["pos"].dtype.name == "category"

def test_clean_invalid_practice_time_becomes_nat(sample_df):
    sample_df.loc[0, "practice_time"] = "banana"
    result = clean(sample_df)
    assert pd.isna(result["practice_time"].iloc[0])

def test_fill_missing_grammar_tags(sample_df):
    result = fill_missing(sample_df)
    assert result["grammar_tags"].iloc[0] == "no_gram"

def test_fill_missing_preserves_existing_grammar_tags(sample_df):
    sample_df.loc[0, "grammar_tags"] = "plural"
    result = fill_missing(sample_df)
    assert result["grammar_tags"].iloc[0] == "plural"