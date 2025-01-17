from sklearn.linear_model import SGDClassifier
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import timm
from tqdm import tqdm
import numpy as np
import pandas as pd
import label_expansion

# Dataset 定義
class ImageDataset(Dataset):
    def __init__(self, df, image_column, transform=None):
        self.image_paths = df[image_column].values
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def prepare_covariates(df, model_name="resnet18"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageDataset(df, "file_path",transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = timm.create_model(model_name, pretrained=True,num_classes=0)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # 画像を特徴ベクトルに変換
    features = []
    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting Features"):
            if torch.cuda.is_available():
                images = images.cuda()
            outputs = model(images)
            features.append(outputs.cpu().numpy())

    # 特徴を結合して返す
    X = np.vstack(features)
    return X



def run_parameterized_estimators(
    df_path,
    T_true = "light_or_dark",
    threshold=0.8,
    only_zeros=True,
    inner_alpha=0.00359,
    outer_alpha=0.00077,
    ):
    """ Run all the ATE estimators based on models:
            regression expansion (+pu classifier), bert adjustment, and
                regression expansion + bert.
    """
    df = pd.read_csv(df_path)
    X =  prepare_covariates(df,model_name="resnet18")
    T_true = df[T_true].to_numpy()
    T_proxy = df['T_proxy'].to_numpy()

    # PU classifier expansion
    only_zeros=True
    pu = label_expansion.PUClassifier(
        inner_alpha=inner_alpha,
        outer_alpha=outer_alpha)
    pu.fit(X, T_proxy)
    T_plus_pu = label_expansion.expand_variable(pu, X, T_proxy,
        threshold=threshold,
        only_zeros=only_zeros)

    # Plain regression expansion
    reg = SGDClassifier(loss="log_loss", penalty="l2", alpha=outer_alpha)
    reg.fit(X, T_proxy)
    T_plus_reg = label_expansion.expand_variable(reg, X, T_proxy,
        threshold=threshold,
        only_zeros=only_zeros)
    # 元の DataFrame に追加
    df['T_plus_pu'] = T_plus_pu
    df['T_plus_reg'] = T_plus_reg

    df.to_csv(f"{df_path[:-4]}-T_boost.csv", index=False)
    return None

if __name__ == "__main__":
    dfs = [
        "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/ShoesCloth_contains_text_t0.8c0.8_noise0.5_1221.csv",
        "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/ShoesCloth_contains_text_t0.8c10.0_noise0.5_1221.csv",
        "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/ShoesCloth_sharpness_ave_t0.8c0.8_noise0.5_1221.csv",
        "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/ShoesCloth_sharpness_ave_t0.8c10.0_noise0.5_1221.csv",
        "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/ShoesCloth_is_shoe_t0.8c0.8_noise0.5_1221.csv",
        "/root/graduation_thetis/causal-bert-pytorch/input/modelinput/ShoesCloth_is_shoe_t0.8c10.0_noise0.5_1221.csv"
    ]
    for df in dfs:
        run_parameterized_estimators(df)
