import pandas as pd
import sys
import numpy as np

def complexity(rca_matrix, iterations=20, drop=True):
    """compute the ECI and ACI from a RCA matrix  

    Args:
        rca_matrix (pandas dataframe): RCA matrix in a pandas dataframe type.
        iterations (int, optional): cutoff of recursive calculation of kp and kc. Defaults to 20.
        drop (bool, optional): validation variable to ensure that return include NaN values. Defaults to True.

    Returns:
        geo_complexity, activity_complexity (pandas series): the complexity of the rows (municipalities int the EC theoretical framework) and columns (activities) respectively. 
    """
    #Binarize rca input 
    rca_matrix = rca_matrix.copy()
    rca_matrix[rca_matrix >= 1] = 1
    rca_matrix[rca_matrix < 1] = 0

    # drop columns / rows only if completely nan
    rca_clone = rca_matrix.copy()
    rca_clone = rca_clone.dropna(how="all")
    rca_clone = rca_clone.dropna(how="all", axis=1)


    if rca_clone.shape != rca_matrix.shape:
        print("[Warning] RCAs contain columns or rows that are entirely comprised of NaN values.")
    if drop:
        rca_matrix = rca_clone

    ka = rca_matrix.sum(axis=0) #sum columns
    kc = rca_matrix.sum(axis=1) #sum rows 
    ka0 = ka.copy()
    kc0 = kc.copy()

    for i in range(1, iterations):
        kc_temp = kc.copy()
        ka_temp = ka.copy()
        ka = rca_matrix.T.dot(kc_temp) / ka0
        if i < (iterations - 1):
            kc = rca_matrix.dot(ka_temp) / kc0

    geo_complexity = (kc - kc.mean()) / kc.std()
    activity_complexity = (ka - ka.mean()) / ka.std()

    return geo_complexity, activity_complexity