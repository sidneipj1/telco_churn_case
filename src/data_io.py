import pandas as pd
import numpy as np

def load_raw(path: str) -> pd.DataFrame:
    "Load csv and strip whitespace from string columns."
    df = pd.read_csv(path)
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())
    return df
