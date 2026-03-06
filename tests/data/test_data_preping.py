from pathlib import Path
import pytest
import pandas as pd
from data.data_preping import *
from data.create_dummy_dataset import *


def test_create_dummy_dataset_contains_expected_issues():
    create_dummy_dataset()
    fixed_name = "data/archive/data/dummy_dataset.csv"
    df, _ = load_to_df(fixed_name)
    assert len(df) == 5
    assert df.isna().any().any()
    assert "'0'" in set(df["Class"])
    assert "2" in set(df["Class"])
    assert "bad_time" in set(df["Time"])


def test_drop_rows_with_nulls_removes_rows_and_prints_percentage(capsys):
    df = pd.DataFrame({"A": [1, None], "B": [1, 2], "Class": [None, None]})
    cleaned = drop_rows_with_nulls(df)
    out = capsys.readouterr().out

    assert len(cleaned) == 1
    assert "Rows passing null check: 50.00%" in out


def test_change_class_to_bool_maps_quoted_values():
    df = pd.DataFrame({"Class": ["'0'", "'1'", " '0' "]})
    out = change_class_to_bool(df)
    assert out["Class"].tolist() == [False, True, False]


def test_change_class_to_bool_maps_unexpected_value_to_null():
    df = pd.DataFrame({"Class": ["'0'", "2"]})
    out = change_class_to_bool(df)
    assert out["Class"].iloc[0] is False
    assert pd.isna(out["Class"].iloc[1])


def test_cast_v_columns_to_float32_changes_only_v_columns():
    df = pd.DataFrame({"V1": ["1.2"], "V2": ["2.3"], "V3": ['str'], "Amount": [float(10.0)]})
    out = cast_v_columns_to_Float32(df)

    assert str(out["V1"].dtype) == "Float32"
    assert str(out["V2"].dtype) == "Float32"
    assert str(out["V3"].dtype) == "Float32"
    assert str(out["Amount"].dtype) != "Float32"


def test_cast_time_to_int_converts_numeric_values():
    df = pd.DataFrame({"Time": ["0", "10", 15.0]})
    out = cast_time_to_Int(df)
    assert out["Time"].tolist() == [0, 10, 15]
    assert str(out["Time"].dtype) == "Int64"


def test_cast_time_to_int_raises_for_non_numeric_values():
    df = pd.DataFrame({"Time": ["0", "bad_time"]})
    out = cast_time_to_Int(df)
    assert pd.isna(out["Time"].iloc[1])
    assert str(out["Time"].dtype) == "Int64"


def test_save_df_writes_file():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    prefix = 'test/data/sample'
    ext = "parquet"
    output_path = None

    try:
        output_path = Path(f"{prefix}.{ext}")
        save_df(df, prefix, ext=ext)

        assert output_path.exists()
        loaded, _ = load_to_df(output_path)
        assert loaded.equals(df)
    finally:
        if output_path is not None and output_path.exists():
            output_path.unlink()


def test_raw_data_to_silver_writes_file():
    fixed_name = "data/archive/data/dummy_dataset.csv"
    df, _ = load_to_df(fixed_name)
    df = raw_data_to_silver(df)
    fixed_name = "data/archive/data/dummy_dataset_silver.parquet"
    save_df(df, fixed_name, 'parquet')
    df_silver, _ = load_to_df(fixed_name)
    df = pd.DataFrame({
    "Time": [100],
    "V1": [-1.5],
    "V2": [0.5],
    "V4": [0.5],
    "Amount": [20.0],
    "Class": [True],
    })

    pd.testing.assert_frame_equal(df_silver, df, check_dtype=False)
