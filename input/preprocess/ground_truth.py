from collections import defaultdict
import numpy as np
import pandas as pd
def ATE_unadjusted(T, Y):
    x = defaultdict(list)
    for t, y in zip(T, Y):
        x[t].append(y)
    T0 = np.mean(x[0])
    T1 = np.mean(x[1])
    return T0 - T1

def ATE_adjusted(C, T, Y):
    x = defaultdict(list)
    for c, t, y in zip(C, T, Y):
        x[c, t].append(y)

    C0_ATE = np.mean(x[0,0]) - np.mean(x[0,1])
    C1_ATE = np.mean(x[1,0]) - np.mean(x[1,1])
    return np.mean([C0_ATE, C1_ATE])


if __name__ == "__main__":
    df = pd.read_csv("./Appliances_preprocess_contains_text_1122.csv")
    treatment = "light_or_dark"
    confounder = "contains_text"
    outcome = "outcome"
    print("ATE_unadjusted: ", ATE_unadjusted(df[treatment], df[outcome]))
    print("ATE_adjusted: ", ATE_adjusted(df[confounder], df[treatment],df[outcome]))