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

@hydra.main(config_path = "conf", config_name = "train", version_base = '1.3')
def main(cfg:DictConfig):
    print(cfg)
    df = pd.read_csv("/root/graduation_thetis/causal-bert-pytorch/input/outputs_v4.csv")
    print(df.head())
    df["light_or_dark"] = df["light_or_dark"].apply(lambda x : 1 if x == "light" else 0)

    data_module = CausalImageDataModule(cfg,df)
    train_dataset_size = len(data_module.train_dataset)
    steps_per_epoch = train_dataset_size // cfg.batch_size
    total_training_steps = steps_per_epoch * cfg.epoch
    cfg.total_training_steps = total_training_steps

    model = ImageCausalModel(cfg)
    wandb_logger = WandbLogger(project='causal_image',
                               name = f'{cfg.pretrained_model}-{cfg.batch_size}-{cfg.epoch}', 
                               save_dir='logs')
    wandb_logger.experiment.config.update(dict(cfg))

    trainer = Trainer(
        logger = wandb_logger,
        max_epochs = 1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True
    )

    trainer.fit(model, data_module)
    """
    trainer = Trainer(
        max_epochs = 1,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar = True
    )
    """
    
    predictions = trainer.predict(model, dataloaders = data_module)
    print(predictions)
    Q0 = predictions[-1]["Q0s"]
    Q1 = predictions[-1]["Q1s"]
    def ATE(Q0, Q1):
        print("Q0:", Q0, "Q1:", Q1)
        return np.mean(Q0 - Q1)
    
    ate = ATE(Q0, Q1)
    print("ATE:", ate)
    wandb_logger.log_metrics({'ATE':ate})




if __name__ == '__main__':
    main()
