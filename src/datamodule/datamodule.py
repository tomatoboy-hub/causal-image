from torch.utils.data import Dataset, DataLoader, RandomSampler,SequentialSampler
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig

class CausalImageDataset(Dataset):
    def __init__(self,image_paths, confounds, treatments = None, outcomes = None,transform = None):
        self.image_paths = image_paths
        self.confounds = confounds
        self.treatments = treatments
        self.outcomes = outcomes

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
                ]
            )
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self,idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        confounds = self.confounds[idx]
        treatment = self.treatments[idx] if self.treatments is not None else -1
        outcome = self.outcomes[idx] if self.outcomes is not None else -1
        return image , confounds, treatment, outcome
    

class CausalImageDataModule(LightningDataModule):
    def __init__(self,cfg:DictConfig):
        super().__init__()
        self.cfg = cfg
    
    def train_dataloader(self):
        train_dataset = CausalImageDataset(cfg = self.cfg)
        train_loader = DataLoader(
            train_dataset,
            batch_size = self.cfg.batch_size,
            shuffle = True,
            num_workers = self.cfg.num_workers
        )
        return train_loader