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
from sklearn.model_selection import train_test_split
from src.utils.common import ATE_unadjusted,ATE_adjusted      

@hydra.main(config_path = "conf", config_name = "train", version_base = '1.3')
def main(cfg:DictConfig):
    print("done")
    set_seed(cfg.seed)
    df = pd.read_csv(cfg.df_path)
    #df["light_or_dark"] = df["light_or_dark"].apply(lambda x : 1 if x == "light" else 0)
    

    
    data_module = CausalImageDataModule(cfg,df)
    train_dataset_size = len(data_module.train_dataset)
    steps_per_epoch = train_dataset_size // cfg.batch_size
    total_training_steps = steps_per_epoch * cfg.epoch
    cfg.total_training_steps = total_training_steps

    data_module.setup("fit")
    model = ImageCausalModel(cfg)
    wandb_logger = WandbLogger(project='causal_image',
                               name = f'{cfg.pretrained_model}-{cfg.batch_size}-{cfg.epoch}', 
                               save_dir='logs')
    wandb_logger.experiment.config.update(dict(cfg))

    trainer = Trainer(
        logger = wandb_logger,
        max_epochs = cfg.epoch,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True,
        log_every_n_steps=cfg.total_training_steps // cfg.batch_size
    )

    trainer.fit(model, data_module)
    
    """
    trainer = Trainer(
        max_epochs = 1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True
    )
    """
    
        # トレーニング後に保存
    #trainer.save_checkpoint("best_model.ckpt")

    # 別の場所で予測時に読み込み
    #model = ImageCausalModel.load_from_checkpoint("best_model.ckpt", cfg=cfg)

    trainer.predict(model, dataloaders=data_module.predict_dataloader())
    ATE_value = model.ate_value

    print(ATE_value)

    if cfg.confounds_column == "sharpness_ave":
        file_path = cfg.file_name_T
        csv_path = cfg.csv_path_T
    elif cfg.confounds_column == "contains_text":
        file_path = cfg.file_name_C
        csv_path = cfg.csv_path_C

    ate_unadj = ATE_unadjusted(df[cfg.treatments_column], df[cfg.outcome_column])
    ate_adj = ATE_adjusted(df[cfg.treatments_column], df[cfg.outcome_column],df[cfg.confounds_column])
    ate_unadj = float(ate_unadj)
    ate_adj = float(ate_adj)

    save_experiment_result(
        file_path=file_path,
        exp_name=f'exp{cfg.exp_id}',
        model_name=cfg.pretrained_model,
        batch_size=cfg.batch_size,
        epochs=cfg.epoch,
        seed=cfg.seed,
        ATE=float(ATE_value),
        desc=cfg.df_path,
        treatment_column=cfg.treatments_column
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
