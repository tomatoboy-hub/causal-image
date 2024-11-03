from torch.utils.data import Dataset, DataLoader, RandomSampler,SequentialSampler
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import transforms as AT
class CausalImageDataset(Dataset):
    def __init__(self,cfg,df):
        self.cfg = cfg
        self.image_paths = df[cfg.image_column]
        self.confounds = df[cfg.confounds_column]
        self.treatments = df[cfg.treatments_column]
        self.outcomes = df[cfg.outcome_column]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
            ]
        )
        if cfg.augmentation:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
                A.OneOf([
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                    A.GaussNoise(var_limit=(5.0, 30.0)),
                ], p=0.5),

                A.OneOf([
                    A.OpticalDistortion(distort_limit=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=1.),
                    A.ElasticTransform(alpha=3),
                ], p=0.5),

                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
                A.Resize(224, 224),
                A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=0.5),    
                A.Normalize(mean=0.5, std=0.5),
                AT.ToTensorV2()
            ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.cfg.augmentation:
            # PIL ImageをNumPy配列に変換
            image = np.array(image)
            # Albumentationsの変換を適用（キーワード引数で渡す）
            transformed = self.transform(image=image)
            image = transformed['image']
        else:
            # torchvision.transformsの変換を適用
            image = self.transform(image)
        
        confounds = self.confounds[idx]
        treatment = self.treatments[idx] if self.treatments is not None else -1
        outcome = self.outcomes[idx] if self.outcomes is not None else -1
        return image , confounds, treatment, outcome
    
class CausalImageDataset_validation(Dataset):
    def __init__(self,cfg,df):
        self.image_paths = df[cfg.image_column]
        self.confounds = df[cfg.confounds_column]
        self.treatments = df[cfg.treatments_column]
        self.outcomes = df[cfg.outcome_column]

        self.transform = transforms.Compose(
            [
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
            ]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        confounds = self.confounds[idx]
        treatment =  -1
        outcome = -1
        return image , confounds, treatment, outcome
    
"""
class CausalImageDataset(Dataset):
    def __init__(self, cfg, df, mode="train"):
        self.image_paths = df[cfg.image_column]
        self.confounds = df[cfg.confounds_column]
        self.treatments = df[cfg.treatments_column] if mode == "train" else None
        self.outcomes = df[cfg.outcome_column] if mode == "train" else None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        confounds = self.confounds[idx]
        treatment = self.treatments[idx] if self.treatments is not None else -1
        outcome = self.outcomes[idx] if self.outcomes is not None else -1
        return image, confounds, treatment, outcome
"""
class CausalImageDataModule(LightningDataModule):
    def __init__(self,cfg:DictConfig,df):
        super().__init__()
        self.cfg = cfg
        self.df = df
        self.train_dataset = CausalImageDataset(cfg = self.cfg,df = self.df)
        self.predict_dataset = CausalImageDataset_validation(cfg = self.cfg,df = self.df)
    def train_dataloader(self):
        sampler = SequentialSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.cfg.batch_size,
            sampler = sampler,
            num_workers = self.cfg.num_workers
        )
        return train_loader
    """
    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle = None,
            num_workers = self.cfg.num_workers
        )
        return valid_loader
    """ 
    def predict_dataloader(self):
        sampler = SequentialSampler(self.train_dataset)
        predict_loader = DataLoader(
            self.train_dataset,
            batch_size = self.cfg.batch_size,
            sampler = sampler,
            num_workers= self.cfg.num_workers
        )
        return predict_loader