from torch.utils.data import Dataset, DataLoader, RandomSampler,SequentialSampler,WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import transforms as AT
from sklearn.model_selection import train_test_split, KFold
import torch
class CausalImageDataset(Dataset):
    def __init__(self,cfg,df):
        self.cfg = cfg
        self.df = df.reset_index(drop=True)  # インデックスをリセット
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
            # self.transform = A.Compose([
            #     A.OneOf([
            #          A.MotionBlur(blur_limit=5),
            #          A.MedianBlur(blur_limit=5),
            #          A.GaussianBlur(blur_limit=5),
            #          A.GaussNoise(var_limit=(5.0, 30.0)),
            #      ], p=0.5),
            #      A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.5),
            #      A.Resize(224, 224),
            #      A.Normalize(mean=0.5, std=0.5),
            #      AT.ToTensorV2()
            # ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        try:
            image_path = self.image_paths.iloc[idx]
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image at index {idx}: {image_path}. Error: {e}")
        
        # 変換の適用
        if self.cfg.augmentation:
            # Albumentations を適用（NumPy配列 -> 辞書形式 -> テンソル）
            image = np.array(image)
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed['image']
        else:
            # torchvision.transforms を適用
            if self.transform is not None:
                image = self.transform(image)
        
        # 共変量、処置、アウトカムを取得
        confounds = torch.tensor(self.confounds.iloc[idx], dtype=torch.float)
        treatment = (
            torch.tensor(self.treatments.iloc[idx], dtype=torch.long)
            if self.treatments is not None else torch.tensor(-1, dtype=torch.long)
        )
        outcome = (
            torch.tensor(self.outcomes.iloc[idx], dtype=torch.long)
            if self.outcomes is not None else torch.tensor(-1, dtype=torch.long)
        )
        return image, confounds, treatment, outcome
    
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
        outcome = self.outcomes[idx] 
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
        self.valid_df = None
        self.predict_df = None  
     
        self.train_dataset = CausalImageDataset(cfg = self.cfg,df = self.df)
    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CausalImageDataset(cfg=self.cfg, df=self.df)
            self.valid_dataset = CausalImageDataset_validation(cfg=self.cfg, df=self.valid_df)
        if stage == 'predict' or stage is None:
            self.predict_dataset = CausalImageDataset_validation(cfg=self.cfg, df=self.predict_df)
    def train_dataloader(self):
        if self.cfg.sampler == "weighted":
            treatment_counts = self.df[self.cfg.treatments_column].value_counts()
            num_samples = len(self.df)
            weights = self.df[self.cfg.treatments_column].apply(lambda x: 1.0/treatment_counts[x]).values
            sampler = WeightedRandomSampler(
                    weights,
                    num_samples,
                    replacement=True
                    )
        elif self.cfg.sampler == "sequential":
            sampler = SequentialSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.cfg.batch_size,
            sampler = sampler,
            num_workers = self.cfg.num_workers
        )
        return train_loader
    

    def valid_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle = None,
            num_workers = self.cfg.num_workers
        )
        return valid_loader

    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size = self.cfg.batch_size,
            num_workers= self.cfg.num_workers
        )
        return predict_loader
"""
class CausalImageDataModule(LightningDataModule):
    def __init__(self,cfg:DictConfig,df):
        super().__init__()
        self.cfg = cfg
        self.df = df
        self.train_dataset = CausalImageDataset(cfg = self.cfg,df = self.df)
        self.predict_dataset = CausalImageDataset_validation(cfg = self.cfg,df = self.df)
    def train_dataloader(self):
        if self.cfg.sampler == "weighted":
            treatment_counts = self.df[self.cfg.treatments_column].value_counts()
            num_samples = len(self.df)
            weights = self.df[self.cfg.treatments_column].apply(lambda x: 1.0/treatment_counts[x]).values
            sampler = WeightedRandomSampler(
                    weights,
                    num_samples,
                    replacement=True
                    )
        elif self.cfg.sampler == "sequential":
            sampler = SequentialSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size = self.cfg.batch_size,
            sampler = sampler,
            num_workers = self.cfg.num_workers
        )
        return train_loader
    
    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle = None,
            num_workers = self.cfg.num_workers
        )
        return valid_loader
    def predict_dataloader(self):
        sampler = SequentialSampler(self.train_dataset)
        predict_loader = DataLoader(
            self.train_dataset,
            batch_size = self.cfg.batch_size,
            sampler = sampler,
            num_workers= self.cfg.num_workers
        )
        return predict_loader
"""