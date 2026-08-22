import typer
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check

schema = pa.DataFrameSchema({
    "p_recall": Column(float, Check.in_range(0, 1)),
    "history_seen": Column(int, Check.greater_than_or_equal_to(0)),
    "history_correct": Column(int, Check.greater_than_or_equal_to(0)),
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

def main(input_csv: str, output_csv: str):
    df = pd.read_csv(input_csv)
    df = clean(df)
    df = fill_missing(df)
    schema.validate(df)
    df.to_csv(output_csv, index=False)
    print(f"Pipeline OK — {len(df)} rows written to {output_csv}")

if __name__ == "__main__":
    typer.run(main)