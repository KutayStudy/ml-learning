import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype
from src.data.build import clean,fill_missing,add_features,join_difficulty

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

@pytest.fixture
def feature_df():
    return pd.DataFrame({
        "lexeme_id": ["lex1", "lex2", "lex3"],
        "lag_days": [0.0, 1.0, 9.0],
        "history_seen": [4, 10, 2],
        "history_correct": [1, 5, 2],
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

def test_add_features_creates_expected_columns(feature_df):
    result = add_features(feature_df)
    assert "lag_days_log" in result.columns
    assert "history_accuracy" in result.columns

def test_add_features_values_match_notebook_formulas(feature_df):
    result = add_features(feature_df)
    assert np.allclose(result["lag_days_log"], np.log1p([0.0, 1.0, 9.0]))
    assert np.allclose(result["history_accuracy"], [0.25, 0.5, 1.0])

def test_add_features_history_seen_zero_matches_notebook(feature_df):
    # Part 4 divided the two columns directly, so a zero denominator gives
    # inf (or nan for 0/0) instead of raising. Keep that behaviour identical
    # here, otherwise the pipeline would stop reproducing v5.
    feature_df.loc[0, "history_seen"] = 0   # history_correct = 1 -> inf
    feature_df.loc[1, "history_seen"] = 0
    feature_df.loc[1, "history_correct"] = 0  # 0 / 0 -> nan

    result = add_features(feature_df)

    assert np.isinf(result["history_accuracy"].iloc[0])
    assert np.isnan(result["history_accuracy"].iloc[1])
    assert result["history_accuracy"].iloc[2] == 1.0

def test_join_difficulty_adds_column_and_keeps_rows(feature_df, tmp_path):
    difficulty = tmp_path / "word_difficulty.csv"
    pd.DataFrame({
        "lexeme_id": ["lex1", "lex3"],
        "difficulty_rank_in_language": [7, 42],
        "unrelated_column": ["a", "b"],
    }).to_csv(difficulty, index=False)

    result = join_difficulty(feature_df, str(difficulty))

    assert "difficulty_rank_in_language" in result.columns
    assert len(result) == len(feature_df)
    assert result["difficulty_rank_in_language"].tolist()[0] == 7
    assert pd.isna(result["difficulty_rank_in_language"].iloc[1])  # no match
    assert "unrelated_column" not in result.columns

def test_join_difficulty_preserves_row_order(feature_df, tmp_path):
    difficulty = tmp_path / "word_difficulty.csv"
    pd.DataFrame({
        "lexeme_id": ["lex3", "lex1", "lex2"],   # deliberately out of order
        "difficulty_rank_in_language": [42, 7, 13],
    }).to_csv(difficulty, index=False)

    result = join_difficulty(feature_df, str(difficulty))

    assert result["lexeme_id"].tolist() == ["lex1", "lex2", "lex3"]