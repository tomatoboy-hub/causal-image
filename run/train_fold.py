import sys
sys.path.append('/root/graduation_thetis/causal-bert-pytorch/')
from src.modelmodule.modelmodule import ImageCausalModel
import pandas as pd
import numpy as np
from src.datamodule.datamodule import CausalImageDataModule
from pytorch_lightning import Trainer
import torch
import hydra
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from src.utils.common import set_seed, save_experiment_result,save_experiment_result_to_csv
from sklearn.model_selection import train_test_split, KFold
from src.utils.common import ATE_unadjusted,ATE_adjusted      


@hydra.main(config_path = "conf", config_name = "train", version_base = '1.3')
def main(cfg:DictConfig):
    set_seed(cfg.seed)
    df = pd.read_csv(cfg.df_path)
    kfold = KFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    wandb_logger = WandbLogger(project='causal_image',
                               name = f'{cfg.pretrained_model}-{cfg.batch_size}-{cfg.seed}', 
                               save_dir='logs')
    train, predict_df = train_test_split(df,test_size=0.2, random_state=42, stratify=df[cfg.outcome_column])
    for fold, (train_indices, valid_indices) in enumerate(kfold.split(train)):
        print(f"Fold {fold + 1}/{cfg.n_splits}")
        train_df = train.iloc[train_indices]
        valid_df = train.iloc[valid_indices]

        data_module = CausalImageDataModule(cfg, train_df)
        valid_df = valid_df.reset_index(drop=True)
        data_module.valid_df = valid_df
    
        train_dataset_size = len(train_df)
        steps_per_epoch = train_dataset_size // cfg.batch_size
        total_training_steps = steps_per_epoch * cfg.epoch
        cfg.total_training_steps = total_training_steps
        data_module.setup("fit")
        model = ImageCausalModel(cfg)
        trainer = Trainer(
            logger = wandb_logger,
            max_epochs=cfg.epoch,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            losg_every_n_steps=cfg.total_training_steps // cfg.batch_size
        )
        trainer.fit(model, train_dataloaders=data_module.train_dataloader(), val_dataloaders=data_module.valid_dataloader())
    
    predict_df = predict_df.reset_index(drop=True)
    data_module.predict_df = predict_df
    data_module.setup("predict")
    trainer.predict(model, dataloaders=data_module.predict_dataloader())
    ATE_value = model.ate_value
    print(ATE_value)
    ate_unadj = ATE_unadjusted(predict_df[cfg.treatments_column], predict_df[cfg.outcome_column])
    ate_adj = ATE_adjusted(predict_df[cfg.confounds_column],predict_df[cfg.treatments_column], predict_df[cfg.outcome_column])
    ate_unadj = float(ate_unadj)
    ate_adj = float(ate_adj)
    
    if cfg.confounds_column == "sharpness_ave":
        file_path = cfg.file_name_T
        csv_path = cfg.csv_path_T
    elif cfg.confounds_column == "contains_text":
        file_path = cfg.file_name_C
        csv_path = cfg.csv_path_C

    save_experiment_result(
        file_path=file_path,
        exp_name=f'exp{cfg.exp_id}',
        model_name=cfg.pretrained_model,
        batch_size=cfg.batch_size,
        epochs=cfg.epoch,
        seed=cfg.seed,
        ATE=float(ATE_value),
        desc=cfg.df_path,
        treatment_column=cfg.treatments_column,
        ATE_unadj=ate_unadj,
        ATE_adj=ate_adj
    )

    save_experiment_result_to_csv(
        file_path=csv_path,
        exp_name=f'exp{cfg.exp_id}',
        model_name=cfg.pretrained_model,
        batch_size=cfg.batch_size,
        epochs=cfg.epoch,
        seed=cfg.seed,
        ATE=float(ATE_value),
        desc=cfg.df_path,
        treatment_column=cfg.treatments_column,
        confounds_column=cfg.confounds_column,
        ATE_unadj=ate_unadj,
        ATE_adj=ate_adj
    )




if __name__ == '__main__':
    main()
