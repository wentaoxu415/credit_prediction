import os
import pandas as pd
from IPython.display import display


def print_full(df):
    num_rows = df.shape[0]
    num_cols = df.shape[1]
    pd.set_option('display.max_rows', num_rows, "display.max_columns", num_cols)
    display(df)
    pd.reset_option('display.max_rows', 'display.max_columns')
