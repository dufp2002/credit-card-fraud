from pathlib import Path
import re

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
    Remove rows that contain null values outside of the Class column.
    Nulls in Class are kept, they could be usefull for unsupervised learning.
    The rest won't be useful unless the dataset contains many nulls.
    Prints pass percentage only when it is below 100%.
    """
    total_rows = len(df)
    non_class_cols = [
        col for col in df.columns if not (isinstance(col, str) and col.lower() == "class")
    ]
    cleaned_df = df.dropna(subset=non_class_cols) if non_class_cols else df.copy()

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
    mapped = df["Class"].astype(str).str.strip("'\" ").map({"0": False, "1": True})
    df["Class"] = mapped

    return df


def cast_v_columns_to_Float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast all columns starting with 'V' (e.g., V1, V2, ...) to pandas Float32.
    Non-numeric values are coerced to null.
    """
    v_cols = [col for col in df.columns if isinstance(col, str) and col.startswith("V")]
    if not v_cols:
        return df

    df[v_cols] = df[v_cols].apply(pd.to_numeric, errors="coerce").astype("Float32")
    return df


def cast_time_to_Int(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast the 'Time' column to int and cap values at max_time.
    The df shouldn't have nulls in it at this point.
    """
    if "Time" not in df.columns:
        raise KeyError("Column 'Time' not found in DataFrame.")

    time_values = pd.to_numeric(df["Time"], errors="coerce")
    # if time_values.isna().any():
    #     raise ValueError("Column 'Time' contains non-numeric values.")

    df["Time"] = time_values.apply(pd.to_numeric, errors="coerce").astype("Int64")
    return df


def remove_extra_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only Time, Class, and columns matching V<digits> (e.g., V1, V12).
    Drop all other columns.
    """
    rename_map = {
        col: f"V{col[1:]}"
        for col in df.columns
        if isinstance(col, str) and col.startswith("v")
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    v_col_pattern = re.compile(r"^V\d+$")
    keep_cols = []
    for col in df.columns:
        if not isinstance(col, str):
            continue
        col_lower = col.lower()
        if col_lower in {"time", "class", 'amount'} or v_col_pattern.match(col):
            keep_cols.append(col)
    return df[keep_cols].copy()


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


def raw_data_to_silver(path: str) -> None:
    """
    Uses remove_extra_columns first since to remove it first tries to
    rename columns.
    """
    df = load_to_df(path)
    df = remove_extra_columns(df)
    df = cast_time_to_Int(df)
    df = cast_v_columns_to_Float32(df)
    df = change_class_to_bool(df)
    df = drop_rows_with_nulls(df)
    source_path = Path(path)
    output_path = source_path.with_name(f"{source_path.stem}_silver{source_path.suffix}")
    save_df(df, output_path)


if __name__ == "__main__":
    path = 'archive/data/creditcard_csv.csv'
    raw_data_to_silver(path)
