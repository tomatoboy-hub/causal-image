#!/bin/bash
# 環境変数の設定（必要に応じて）
export CUDA_VISIBLE_DEVICES=0


# 組み合わせたいパラメータを定義
pretrained_models=("timm/eva02_tiny_patch14_224.mim_in22k" "timm/efficientvit_m2.r224_in1k" "timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k")
batch_sizes=(32)
epochs=(3)
seeds=(42 123 456 789 1011 1213 1415 1617 1819 2021 2223 2425 2627 2829 3031 3233 3435 3637 3839 4041) 
treats=("T_plus_pu" "T_plus_reg" "T_proxy") 

# 初期のexp_idを設定
exp_id=1


# パラメータの組み合わせでループを回して実行
for model in "${pretrained_models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for epoch in "${epochs[@]}"; do
            for seed in "${seeds[@]}"; do
                for treat in "${treats[@]}"; do
                # Hydraの設定をオーバーライドして実行
                    python train.py \
                        pretrained_model=$model \
                        batch_size=$batch_size \
                        epoch=$epoch \
                        exp_id=$exp_id \
                        seed=$seed \
                        treatments_column=$treat

                    # exp_idをインクリメント
                    exp_id=$((exp_id + 1))
                    
                    # 少し待機して次の実行（オプション）
                    sleep 5
                done
            done
        done
    done
done

python send_message.py 