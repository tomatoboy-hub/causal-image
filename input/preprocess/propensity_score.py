import pandas as pd
import numpy as np
import numpy.random as random
from scipy.special import expit
from collections import defaultdict

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

def calculate_propensity_score(df, confounder, treatment):
    confounder_df = df[df[confounder] == 0]
    probability_0 = (confounder_df[treatment] == 1).mean()
    print(f"{confounder}が0のもののうち{treatment}が1である確率:", probability_0)
    confounder_df = df[df[confounder] == 1]
    probability_1 = (confounder_df[treatment] == 1).mean()
    print(f"{confounder}が1のもののうち{treatment}が1である確率:", probability_1)
    return probability_0,probability_1

def outcome_sim(beta0, beta1, gamma, treatment, confounding, noise, setting = "simple"):
    if setting == "simple":
        y0 = beta1 * confounding
        y1 = beta0 + y0

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise
    return simulated_score, y0,y1

def make_price_dark_probs(df,treat_strength,con_strength,probability_0, probability_1,noise_level, setting = "simple", seed = 0):
    price_ave_given_dark_probs = np.array([probability_0,probability_1])

    np.random.seed(seed)
    all_noise = np.array(random.normal(0,1,len(df)), dtype = np.float32)
    all_thresholds = np.array(random.uniform(0,1,len(df)), dtype = np.float32)
    
    outcomes = []
    y0s = []
    y1s = []

    for i, data in df.iterrows():
        light_or_dark = data["sharpness_ave"]
        treatment = data["light_or_dark"]

        confounding = (price_ave_given_dark_probs[light_or_dark] - 0.50)
        noise = all_noise[i]
        y,y0,y1 = outcome_sim(treat_strength, con_strength, noise_level,treatment, confounding, noise, setting = setting)
        simulated_prob = expit(y)
        y0 = expit(y0)
        y1 = expit(y1)
        threshold = all_thresholds[i]
        simulated_outcome = 1 if simulated_prob > threshold else 0
        outcomes.append(simulated_outcome)
        y0s.append(y0)
        y1s.append(y1)

    randoms = np.random.uniform(0, 1, len(df['light_or_dark']))
    accuracy = -1
    if isinstance(accuracy, tuple): # TODO this is a hack
        pThatGivenT = accuracy
    elif accuracy > 0:
        pThatGivenT = [1 - accuracy, accuracy]
    else:
        pThatGivenT = [0.2, 0.8]
    mask = np.array([pThatGivenT[ti] for ti in df['light_or_dark']])
    df['T_proxy'] = (randoms < mask).astype(int)
    df['outcome'] = outcomes
    df['y0'] = y0s
    df['y1'] = y1s
    return df

def adjust_precision_recall(df, target_precision, target_recall):
    # now balance data again so proxy treatment has right precision
    x = defaultdict(int)
    for t_true, t_proxy in zip(df.light_or_dark, df.T_proxy):
        x[t_true, t_proxy] += 1
    true_precision = x[1, 1] / (x[0, 1] + x[1, 1])
    true_recall = x[1, 1] / (x[1, 1] + x[1, 0])

    true1_subset = df.loc[df.light_or_dark == 1]
    true0_subset = df.loc[df.light_or_dark == 0]
    true1_proxy1_subset = true1_subset.loc[true1_subset.T_proxy == 1]
    true1_proxy0_subset = true1_subset.loc[true1_subset.T_proxy == 0]
    true0_proxy1_subset = true0_subset.loc[true0_subset.T_proxy == 1]

    if target_precision > true_precision:
        # adjust precision with inverse of y = tp / (tp + x)
        tgt_num_t0p1 = -len(true1_proxy1_subset) * (target_precision - 1) / target_precision
        drop_prop = (len(true0_proxy1_subset) - tgt_num_t0p1) / len(true0_proxy1_subset)
        df = df.drop(true0_proxy1_subset.sample(frac=drop_prop).index)
    else:
        # adjust down with inverse of y = x / (x + fp)
        tgt_num_t1p1 = - (len(true0_proxy1_subset) * target_precision) / (target_precision - 1)
        drop_prop = (len(true1_proxy1_subset) - tgt_num_t1p1) / len(true1_proxy1_subset)
        df = df.drop(true1_proxy1_subset.sample(frac=drop_prop).index)

    # refresh subsets (TODO refactor)
    true1_subset = df.loc[df.light_or_dark == 1]
    true0_subset = df.loc[df.light_or_dark == 0]
    true1_proxy1_subset = true1_subset.loc[true1_subset.T_proxy == 1]
    true1_proxy0_subset = true1_subset.loc[true1_subset.T_proxy == 0]
    true0_proxy1_subset = true0_subset.loc[true0_subset.T_proxy == 1]

    if target_recall > true_recall:
        # adjust recall with inverse of t1p1 / (t1p1 + x)
        tgt_num_t1p0 = -len(true1_proxy1_subset) * (target_recall - 1) / target_recall
        drop_prop = (len(true1_proxy0_subset) - tgt_num_t1p0) / len(true1_proxy0_subset)
        df = df.drop(true1_proxy0_subset.sample(frac=drop_prop).index)
    else:
        # adjust down with inverse of y = x / (x + fn)
        tgt_num_t1p1 = - (len(true1_proxy0_subset) * target_recall) / (target_recall - 1)
        drop_prop = (len(true1_proxy1_subset) - tgt_num_t1p1) / len(true1_proxy1_subset)
        df = df.drop(true1_proxy1_subset.sample(frac=drop_prop).index)

    return df

def adjust_propensity(df, target_propensities,true_propensities):
    # subset to to desired propensities (must be smaller than true)
    true_propensities = true_propensities
    target_propensities = [0.70, 0.80] # works well

    for i, (pi_tgt, pi_true) in enumerate(zip(target_propensities, true_propensities)):
        # drop enough samples so that we get the desired propensity
        # inverse of y = x / (x + out sample size) gives you tgt number for proportion
        Ci_subset = df.loc[df.contains_text == i]
        Ci_T0_subset = Ci_subset.loc[Ci_subset.light_or_dark == 0]
        Ci_T1_subset = Ci_subset.loc[Ci_subset.light_or_dark == 1]
        tgt_num = -len(Ci_T0_subset) * pi_tgt / (pi_tgt - 1)
        drop_prop = (len(Ci_T1_subset) - tgt_num) / len(Ci_T1_subset)
        print(drop_prop)
        df = df.drop(Ci_T1_subset.sample(frac=drop_prop).index)
    return df

if __name__ == "__main__":
    csv_path = "/root/graduation_thetis/causal-bert-pytorch/input/shoclo_output.csv"
    confounder = "contains_text"
    treatment = "light_or_dark"
    df = pd.read_csv(csv_path)
    probability_0, probability_1 = calculate_propensity_score(df, confounder, treatment)
    #df = adjust_propensity(df,[0.8, 0.7],[probability_0,probability_1])
    treatment_strength =0.8
    confounding_strength = 10.0
    noise_level = 0.5
    df = make_price_dark_probs(df,treatment_strength, confounding_strength,probability_0,probability_1,noise_level,"simple", 0)
    df = adjust_precision_recall(df, target_precision=0.94, target_recall=0.98)
    print(df[df[treatment] == 1]["outcome"].sum()/ len(df[df[treatment] == 1]))
    print(df[df[treatment] == 0]["outcome"].sum()/ len(df[df[treatment] == 0]))
    df.to_csv(f"./modelinput/ShoesCloth_{confounder}_t{treatment_strength}c{confounding_strength}_noise{noise_level}_1221.csv", index = None)
    print("ATE_unadjusted: ", ATE_unadjusted(df["T_proxy"], df["outcome"]))
    print("ATE_adjusted: ", ATE_adjusted(df[confounder], df[treatment],df["outcome"]))