import typer
import pandas as pd
import pandera as pa
from pandera import Column,Check

schema = pa.DataFrameSchema({
    "value": Column(float,Check.greater_than_or_equal_to(0)),
})

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df["value"] = df["value"].fillna(df["value"].mean())
    return df

def add_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["value_doubled"] = df["value"] * 2
    return df

def main(input_csv: str):
    df = pd.read_csv(input_csv)
    df = clean(df)
    schema.validate(df)
    df = add_feature(df)
    print(df)
    print("Pipeline OK")

if __name__ == "__main__":
    typer.run(main)