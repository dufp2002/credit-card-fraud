from data.data_preping import *

def build_silver_dataset(path: str | Path) -> None:
    df, source_path = load_to_df(path)
    df = raw_data_to_silver(df)
    output_path = source_path.with_name(f"{source_path.stem}_silver{source_path.suffix}")
    save_df(df, output_path, 'parquet')

if __name__ == "__main__":
    path = "data/archive/data/creditcard_csv.csv"
    build_silver_dataset(path)
