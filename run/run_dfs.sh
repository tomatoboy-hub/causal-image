#!/bin/bash
# 環境変数の設定（必要に応じて）
export CUDA_VISIBLE_DEVICES=0


# 組み合わせたいパラメータを定義
pretrained_models=("timm/eva02_tiny_patch14_224.mim_in22k" "timm/efficientvit_m2.r224_in1k" "timm/vit_base_patch32_clip_224.laion2b_ft_in12k_in1k")
batch_sizes=(32)
epochs=(3 5 10 20)
seeds=(42)
#123 456 789 1000 2000 3000 4000 5000 6000  
# dfs_S=("/root/graduation_thetis/causal-bert-pytorch/input/modelinput/shoclo_T-boost/ShoesCloth_sharpness_ave_t0.8c10.0_noise0.5_1221-T_boost.csv"
#      "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/shoclo_T-boost/ShoesCloth_sharpness_ave_t0.8c0.8_noise0.5_1221-T_boost.csv"
#      ) 
# dfs_C=("/root/graduation_thetis/causal-bert-pytorch/input/modelinput/T-boost/Appliances_preprocess_contains_text_t0.8c0.8_1203-T_boost.csv" 
#      "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/T-boost/Appliances_preprocess_containstext_t0.8c10.0_1202-T_boost.csv")
# dfs=("/root/graduation_thetis/causal-bert-pytorch/input/modelinput/T-boost/Appliances_preprocess_sharpness_ave_t0.8c0.8_1210-T_boost.csv"
#     "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/T-boost/Appliances_preprocess_sharpness_ave_t0.8c10.0_1202-T_boost.csv"
#     "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/T-boost/Watch_preprocess_sharpness_ave_t0.8c0.8_1206-T_boost.csv"
#     "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/T-boost/Watch_preprocess_sharpness_ave_t0.8c10.0_1206-T_boost.csv"
#     )

dfs_C=(/root/graduation_thetis/causal-bert-pytorch/input/modelinput/shoclo_T-boost/ShoesCloth_contains_text_t0.8c10.0_noise0.5_1221-T_boost.csv "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/shoclo_T-boost/ShoesCloth_contains_text_t0.8c0.8_noise0.5_1221-T_boost.csv" )
# 初期のexp_idを設定
exp_id=1


# # パラメータの組み合わせでループを回して実行
# for model in "${pretrained_models[@]}"; do
#     for batch_size in "${batch_sizes[@]}"; do
#         for epoch in "${epochs[@]}"; do
#             for seed in "${seeds[@]}"; do
#                 for df in "${dfs_S[@]}"; do
#                 # Hydraの設定をオーバーライドして実行
#                     python train_fold.py \
#                         pretrained_model=$model \
#                         batch_size=$batch_size \
#                         epoch=$epoch \
#                         exp_id=$exp_id \
#                         seed=$seed \
#                         df_path=$df \
#                         confounds_column="sharpness_ave"

#                     # exp_idをインクリメント
#                     exp_id=$((exp_id + 1))
                    
#                     # 少し待機して次の実行（オプション）
#                     sleep 5
#                 done
#             done
#         done
#     done
# done

# python send_message_df.py confounds_column="sharpness_ave"

exp_id=1
# パラメータの組み合わせでループを回して実行
for model in "${pretrained_models[@]}"; do
    for batch_size in "${batch_sizes[@]}"; do
        for epoch in "${epochs[@]}"; do
            for seed in "${seeds[@]}"; do
                for df in "${dfs_C[@]}"; do
                # Hydraの設定をオーバーライドして実行
                    python train_fold_duplicate.py \
                        pretrained_model=$model \
                        batch_size=$batch_size \
                        epoch=$epoch \
                        exp_id=$exp_id \
                        seed=$seed \
                        df_path=$df \
                        confounds_column="contains_text"

                    # exp_idをインクリメント
                    exp_id=$((exp_id + 1))
                    
                    # 少し待機して次の実行（オプション）
                    sleep 5
                done
            done
        done
    done
done

python send_message_df.py confounds_column="contains_text"