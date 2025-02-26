#!/bin/bash
#!/bin/bash

# 環境変数の設定（必要に応じて）
export CUDA_VISIBLE_DEVICES=0


# 組み合わせたいパラメータを定義
seeds=(42 123 456 789 1011 1213 1415 1617 1819 2021 2223 2425 2627 2829 3031 3233 3435 3637 3839 4041)  # シード値のリスト
batch_sizes=(32)
epochs=(3)

# 初期のexp_idを設定
exp_id=17
for seed in "${seeds[@]}"; do
    # パラメータの組み合わせでループを回して実行
    for batch_size in "${batch_sizes[@]}"; do
        for epoch in "${epochs[@]}"; do

            # Hydraの設定をオーバーライドして実行
            python causal_bert_train.py \
                batch_size=$batch_size \
                epoch=$epoch \
                exp_id=$exp_id \
                seed=$seed

            # exp_idをインクリメント
            exp_id=$((exp_id + 1))
            
            # 少し待機して次の実行（オプション）
            sleep 5
        done
    done
done

