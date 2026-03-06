import pandas as pd
from data.data_preping import save_df
from pathlib import Path


def create_dummy_dataset() -> None:
    """
    Build a small dataset containing common data quality issues.

    Included issues:
    - Null values (for null-drop testing)
    - Quoted class labels ("'0'", "'1'")
    - Invalid class label ("2")
    - Non-numeric time value ("bad_time")
    - V* columns stored as strings
    - Lowercase v column
    - Non V name column
    """
    fixed_name = "archive/data/dummy_dataset.csv"
    ext = "csv"
    df = pd.DataFrame(
        {
            "Time": ["0", "100", "175001", "bad_time", "250"],
            "V1": ["0.1", "-1.5", "2.3", "0.0", None],
            "V2": ["foo", "0.5", None, "-0.7", "2.0"],
            "S3": ["foo", "0.5", None, "-0.7", "2.0"],
            "v4": ["foo", "0.5", None, "-0.7", "2.0"],
            "Amount": [10.5, 20.0, 5.2, None, 99.9],
            "Class": ["'0'", "'1'", "2", "'1'", "'0'"],
        }
    )
    save_df(df, fixed_name, ext=ext)


def delete_dummy_dataset() -> bool:
    """
    Delete the fixed dummy dataset file.
    Returns True when a file was deleted, otherwise False.
    """
    fixed_name = "archive/data/dummy_dataset.csv"
    fixed_path = Path(__file__).resolve().parent / fixed_name
    if fixed_path.exists():
        fixed_path.unlink()
        return True
    return False
