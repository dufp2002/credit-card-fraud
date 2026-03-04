import pandas as pd
import data.data_preping
from data.data_preping import save_df


def create_dummy_dataset() -> pd.DataFrame:
    """
    Build a small dataset containing common data quality issues.

    Included issues:
    - Null values (for null-drop testing)
    - Quoted class labels ("'0'", "'1'")
    - Invalid class label ("2")
    - Non-numeric time value ("bad_time")
    - V* columns stored as strings
    """
    return pd.DataFrame(
        {
            "Time": ["0", "100", "175001", "bad_time", "250"],
            "V1": ["0.1", "-1.5", "2.3", "0.0", None],
            "V2": ["foo", "0.5", None, "-0.7", "2.0"],
            "Amount": [10.5, 20.0, 5.2, None, 99.9],
            "Class": ["'0'", "'1'", "2", "'1'", "'0'"],
        }
    )


if __name__ == "__main__":
    df = create_dummy_dataset()
    save_df(df, 'archive/data/dummy_dataset.csv')
