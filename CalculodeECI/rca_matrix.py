import pandas as pd
import sys
import numpy as np

def rca(df_raw):

    # fill missing values with zeros
    df_raw = df_raw.fillna(0)

    col_sums = df_raw.sum(axis=1)
    col_sums = col_sums.values.reshape((len(col_sums), 1))

    rca_numerator = np.divide(df_raw, col_sums)
    row_sums = df_raw.sum(axis=0)

    total_sum = df_raw.sum().sum()
    rca_denominator = row_sums / total_sum
    rcas = rca_numerator / rca_denominator

    return rcas