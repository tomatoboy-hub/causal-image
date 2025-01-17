import random
import numpy as np
import torch
import yaml
from collections import defaultdict
import os
import csv
# 関数定義
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

    C0_ATE = np.mean(x[0, 0]) - np.mean(x[0, 1])
    C1_ATE = np.mean(x[1, 0]) - np.mean(x[1, 1])
    return np.mean([C0_ATE, C1_ATE])

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_experiment_result(file_path, exp_name, model_name, batch_size, epochs,seed,ATE,desc="",treatment_column="",ATE_unadj=0,ATE_adj=0):
    new_entry = {
        exp_name: {
            "seed": seed,
            "model_name": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "ATE": ATE,
            "desc":desc,
            "treatment_column":treatment_column,
            "ATE_unadjusted": ATE_unadj,  
            "ATE_adjusted": ATE_adj
        }
    }

    try:
        # 既存の結果を読み込む
        with open(file_path, 'r') as file:
            existing_data = yaml.safe_load(file)
            if existing_data is None:
                existing_data = {}
    except FileNotFoundError:
        existing_data = {}

    # 新しい結果を追加
    existing_data.update(new_entry)

    # 変更を保存
    with open(file_path, 'w') as file:
        yaml.dump(existing_data, file)
def save_experiment_result_to_csv(file_path, 
                                  exp_name, 
                                  model_name, 
                                  batch_size, 
                                  epochs, 
                                  seed, 
                                  ATE, 
                                  desc="", 
                                  treatment_column="", 
                                  confounds_column="",
                                  ATE_unadj=0, 
                                  ATE_adj=0):

    # CSVの列名を定義
    columns = [
        "exp_name",
        "seed",
        "model_name",
        "batch_size",
        "epochs",
        "ATE",
        "desc",
        "treatment_column",
        "confounds_column",
        "ATE_unadjusted",
        "ATE_adjusted"
    ]
    
    # 追記する内容を辞書にまとめる
    new_row = {
        "exp_name": exp_name,
        "seed": seed,
        "model_name": model_name,
        "batch_size": batch_size,
        "epochs": epochs,
        "ATE": ATE,
        "desc": desc,
        "treatment_column": treatment_column,
        "confounds_column": confounds_column,
        "ATE_unadjusted": ATE_unadj,
        "ATE_adjusted": ATE_adj
    }
    
    # ファイルの存在有無をチェック
    file_exists = os.path.exists(file_path)
    
    # 'a' モードで開き、既存ファイルがない場合のみヘッダを書き込み
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        
        # CSVがまだ存在しない場合、ヘッダを書き込む
        if not file_exists:
            writer.writeheader()
        
        # データを書き込む（1行分）
        writer.writerow(new_row)

