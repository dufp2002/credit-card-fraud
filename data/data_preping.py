from pathlib import Path

import pandas as pd

def load_to_df(path: str) -> pd.DataFrame:
    """
    Load a single file path into a pandas DataFrame.

    Supported formats: csv, json, parquet, xls/xlsx, feather, pickle, tsv, txt.
    """
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""

    readers = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "parquet": pd.read_parquet,
        "xls": pd.read_excel,
        "xlsx": pd.read_excel,
        "feather": pd.read_feather,
        "pkl": pd.read_pickle,
        "pickle": pd.read_pickle,
        "tsv": lambda p: pd.read_csv(p, sep="\t"),
        "txt": pd.read_table,
    }

    if ext in readers:
        return readers[ext](path)

    raise ValueError(
        f"Unsupported file format: .{ext}. "
        "Use one of: csv, json, parquet, xls, xlsx, feather, pkl, pickle, tsv, txt."
    )

def drop_rows_with_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that contain at least one null value.
    Prints pass percentage only when it is below 100%.
    """
    total_rows = len(df)
    cleaned_df = df.dropna()

    if total_rows == 0:
        print('Empty dataframe')
        return cleaned_df

    pass_percent = (len(cleaned_df) / total_rows) * 100
    if pass_percent < 100:
        print(f"Rows passing null check: {pass_percent:.2f}%")

    return cleaned_df


def change_class_to_bool(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function is specific to the creditcard_csv.csv file
    input: dataframe were class is or isn't boolean
    output: dataframe were class is boolean
    """
    mapped = (
        df["Class"].astype(str).str.strip("'\" ").map({"0": False, "1": True})
    )
    if mapped.isna().any():
        bad = df.loc[mapped.isna(), "Class"].unique()
        raise ValueError(f"Unexpected Class values: {bad}")
    df["Class"] = mapped

    return df


def cast_v_columns_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all columns starting with 'V' (e.g., V1, V2, ...) to float32.
    """
    v_cols = [col for col in df.columns if isinstance(col, str) and col.startswith("V")]
    if not v_cols:
        return df

    df[v_cols] = df[v_cols].astype("float32")
    return df


def cast_time_to_int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the 'Time' column to int and cap values at max_time.
    The df shouldn't have nulls in it at this point.
    """
    if "Time" not in df.columns:
        raise KeyError("Column 'Time' not found in DataFrame.")

    time_values = pd.to_numeric(df["Time"], errors="coerce")
    if time_values.isna().any():
        raise ValueError("Column 'Time' contains non-numeric values.")

    df["Time"] = time_values.astype("int64")
    return df


def save_df(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to disk based on file extension.

    Supported formats: csv, json, parquet, xls/xlsx, feather, pkl, pickle, tsv, txt.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower().lstrip(".")

    if ext == "csv":
        df.to_csv(output_path, index=False)
    elif ext == "json":
        df.to_json(output_path, orient="records")
    elif ext == "parquet":
        df.to_parquet(output_path, index=False)
    elif ext in {"xls", "xlsx"}:
        df.to_excel(output_path, index=False)
    elif ext == "feather":
        df.to_feather(output_path)
    elif ext in {"pkl", "pickle"}:
        df.to_pickle(output_path)
    elif ext == "tsv":
        df.to_csv(output_path, sep="\t", index=False)
    elif ext == "txt":
        df.to_csv(output_path, sep="\t", index=False)
    else:
        raise ValueError(
            f"Unsupported file format: .{ext}. "
            "Use one of: csv, json, parquet, xls, xlsx, feather, pkl, pickle, tsv, txt."
        )


if __name__ == "__main__":
    path = 'archive/data/creditcard_csv.csv'
    df = load_to_df(path)
    df = drop_rows_with_nulls(df)
    df = cast_time_to_int(df)
    df = cast_v_columns_to_float32(df)
    df = change_class_to_bool(df)
    source_path = Path(path)
    output_path = source_path.with_name(f"{source_path.stem}_gold{source_path.suffix}")
    save_df(df, 'archive/data/creditcard_csv_gold.csv')
