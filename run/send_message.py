import yaml
import numpy as np
import pandas as pd
from collections import defaultdict
import yagmail
from dotenv import load_dotenv
import os
import hydra
from omegaconf import DictConfig
# .envファイルを読み込む
load_dotenv()

# メール設定を環境変数から取得
EMAIL_SENDER = os.getenv("EMAIL_SENDER")  # 送信元メールアドレス
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # 送信元メールのアプリパスワード
EMAIL_RECIPIENT = os.getenv("EMAIL_RECIPIENT")  # 送信先メールアドレス

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


@hydra.main(config_path="conf", config_name="train")
def main(cfg: DictConfig) -> None:
    # train.yamlからファイルパスを取得
    with open("/root/graduation_thetis/causal-bert-pytorch/run/conf/train.yaml", 'r') as yml:
        train_config = yaml.safe_load(yml)

    yaml_path = cfg["file_name"]
    csv_path = cfg["df_path"]
    email_body = ""
    # yamlデータを読み込む
    with open(yaml_path, 'r') as yml:
        yaml_data = yaml.safe_load(yml)
    for treat in ["T_proxy" ,"T_plus_pu", "T_plus_reg"]:
        eva_ate, vit_ate, eff_ate = [], [], []
        for k, v in yaml_data.items():
            if v["treatment_column"] != treat:
                continue
            if v['model_name'] == 'timm/eva02_tiny_patch14_224.mim_in22k':
                eva_ate.append(v['ATE'])
            elif v['model_name'] == 'timm/efficientvit_m2.r224_in1k':
                eff_ate.append(v['ATE'])
            elif v['model_name'] == 'timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k':
                vit_ate.append(v['ATE'])

        # ATEの統計量を計算
        eva_stats = f"eva_ate: mean={np.mean(eva_ate):.4f}, std={np.std(eva_ate):.4f}"
        vit_stats = f"vit_ate: mean={np.mean(vit_ate):.4f}, std={np.std(vit_ate):.4f}"
        eff_stats = f"eff_ate: mean={np.mean(eff_ate):.4f}, std={np.std(eff_ate):.4f}"

        # ATE_unadjusted, ATE_adjustedを計算
        df = pd.read_csv(csv_path)
        treatment = "light_or_dark"
        confounder = cfg["confounds_column"]
        outcome = cfg["outcome_column"]
        T_proxy = "T_proxy"
        T_boost = treat

        unadjusted_ate = ATE_unadjusted(df[T_proxy], df[outcome])
        adjusted_ate = ATE_adjusted(df[confounder], df[treatment], df[outcome])
        T_boost_ate = ATE_adjusted(df[confounder], df[T_boost], df[outcome])

        unadjusted_stats = f"ATE_unadjusted: {unadjusted_ate:.4f}"
        adjusted_stats = f"ATE_adjusted: {adjusted_ate:.4f}"
        T_boost_stats = f"ATE_adjusted({treat}): {T_boost_ate:.4f}"

        # メール本文を作成
        email_body += "\n".join([
            "パラメータ:",
            f"treat={treat}",
            "ATE統計情報:",
            eva_stats,
            vit_stats,
            eff_stats,
            "",
            "ATE分析:",
            unadjusted_stats,
            adjusted_stats,
            T_boost_stats,
        ])

    # メールを送信
    try:
        yag = yagmail.SMTP(EMAIL_SENDER, EMAIL_PASSWORD)
        yag.send(
            to=EMAIL_RECIPIENT,
            subject="ATE分析結果",
            contents=email_body
        )
        print("メールが送信されました。")
    except Exception as e:
        print(f"メール送信中にエラーが発生しました: {e}")
    
    return

if __name__ == "__main__":
    main()