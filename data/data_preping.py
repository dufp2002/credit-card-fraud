from pathlib import Path
import re

import pandas as pd

def _resolve_input_path(path: str | Path) -> Path:
    """
    Resolve input path regardless of current working directory.
    - If absolute, use as-is.
    - If relative, resolve from project root.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    project_root = Path(__file__).resolve().parent.parent
    resolved = project_root / candidate
    if resolved.exists():
        return resolved

    # Fallback: when only a filename is provided, search the project tree.
    matches = sorted(
        p for p in project_root.rglob(candidate.name) if p.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Ambiguous filename '{candidate.name}'. Matches: "
            + ", ".join(str(p) for p in matches)
        )

    raise FileNotFoundError(f"Input file not found: {path}")

def load_to_df(path: str | Path) -> tuple[pd.DataFrame, Path]:
    """
    Load a single file path into a pandas DataFrame.

    Supported formats: csv, json, parquet, xls/xlsx, feather, pickle, tsv, txt.
    """
    resolved_path = _resolve_input_path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Input file not found: {resolved_path}")

    ext = resolved_path.suffix.lower().lstrip(".")

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
        return readers[ext](resolved_path), resolved_path

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


def save_df(df: pd.DataFrame, path: str | Path, ext: str | None = 'parquet') -> None:
    """
    Save a DataFrame to disk based on file extension.

    Supported formats: csv, json, parquet, xls/xlsx, feather, pkl, pickle, tsv, txt.
    """
    path_obj = path if isinstance(path, Path) else Path(path)
    path_ext = path_obj.suffix.lstrip(".").lower()
    final_ext = ext.lstrip(".").lower() if ext is not None else path_ext
    if not final_ext:
        raise ValueError("File extension is required in path or ext argument.")

    output_path = path_obj.with_suffix(f".{final_ext}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if final_ext == "parquet":
        df.to_parquet(output_path, index=False)
    elif final_ext == "csv":
        df.to_csv(output_path, index=False)
    elif final_ext == "json":
        df.to_json(output_path, orient="records")
    elif final_ext in {"xls", "xlsx"}:
        df.to_excel(output_path, index=False)
    elif final_ext == "feather":
        df.to_feather(output_path)
    elif final_ext in {"pkl", "pickle"}:
        df.to_pickle(output_path)
    elif final_ext == "tsv":
        df.to_csv(output_path, sep="\t", index=False)
    elif final_ext == "txt":
        df.to_csv(output_path, sep="\t", index=False)
    else:
        raise ValueError(
            f"Unsupported file format: .{final_ext}. "
            "Use one of: csv, json, parquet, xls, xlsx, feather, pkl, pickle, tsv, txt."
        )


def raw_data_to_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses remove_extra_columns first since to remove it first tries to
    rename columns.
    """
    df = remove_extra_columns(df)
    df = cast_time_to_Int(df)
    df = cast_v_columns_to_Float32(df)
    df = change_class_to_bool(df)
    df = drop_rows_with_nulls(df)

    return df
