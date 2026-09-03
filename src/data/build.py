import typer
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

# Input is the sampled flagship CSV from part 2 (16,382 rows), not the raw
# learning_traces_sample.csv. The user sampling in part 2 used an unseeded
# ORDER BY random(), so re-running it would pick a different 2,500 users and
# invalidate data/split_users.csv. That sample stays frozen as the input here.
schema = pa.DataFrameSchema({
    "p_recall": Column(float, Check.in_range(0, 1)),
    "history_seen": Column(int, Check.greater_than_or_equal_to(0)),
    "history_correct": Column(int, Check.greater_than_or_equal_to(0)),
    "lag_days_log": Column(float, Check.greater_than_or_equal_to(0)),
    "history_accuracy": Column(float, Check.in_range(0, 1)),
    "difficulty_rank_in_language": Column(
        float, Check.greater_than_or_equal_to(1), nullable=True
    ),
})

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["practice_time"] = pd.to_datetime(df["practice_time"],errors = "coerce")
    df["ui_language"] = df["ui_language"].astype("category")
    df["learning_language"] = df["learning_language"].astype("category")
    df["pos"] = df["pos"].astype("category")
    df["lemma"] = df["lemma"].str.strip()
    df["surface_form"] = df["surface_form"].str.strip()
    return df

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    df["grammar_tags"] = df["grammar_tags"].fillna("no_gram")
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """The two engineered columns from part 4.

    history_accuracy is a plain division, same as the notebook: a zero
    history_seen yields inf (or nan for 0/0) rather than raising. The real
    sample has no such rows, and the schema rejects them if they ever appear.
    """
    df["lag_days_log"] = np.log1p(df["lag_days"])
    df["history_accuracy"] = df["history_correct"] / df["history_seen"]
    return df

def join_difficulty(df: pd.DataFrame, difficulty_csv: str) -> pd.DataFrame:
    """The word-difficulty LEFT JOIN from part 14.

    lexeme_id is unique in the difficulty table, so this cannot duplicate rows.
    Unlike the DuckDB version this preserves input row order.
    """
    difficulty = pd.read_csv(difficulty_csv, usecols=["lexeme_id", "difficulty_rank_in_language"])
    before = len(df)
    df = df.merge(difficulty, on="lexeme_id", how="left")
    assert len(df) == before, "difficulty join changed the row count"
    return df

def main(
    input_csv: str,
    output_csv: str,
    difficulty_csv: str = typer.Option("data/word_difficulty.csv", help="Word difficulty table to join on lexeme_id."),
):
    df = pd.read_csv(input_csv)
    df = clean(df)
    df = fill_missing(df)
    df = add_features(df)
    df = join_difficulty(df, difficulty_csv)
    schema.validate(df)
    df.to_csv(output_csv, index=False)
    print(f"Pipeline OK — {len(df)} rows, {len(df.columns)} columns written to {output_csv}")

if __name__ == "__main__":
    typer.run(main)
