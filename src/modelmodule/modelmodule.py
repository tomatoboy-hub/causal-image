from typing import Optional
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import CrossEntropyLoss
import timm 
from torch.optim import AdamW
from omegaconf import DictConfig

from transformers import get_linear_schedule_with_warmup

def random_mask(images, mask_ratio=0.15):
    """
    画像に対するランダムマスクを生成し適用。
    """
    # マスクを適用する領域を指定する
    mask = torch.rand(images.size(0), 1, images.size(2), images.size(3), device=images.device) < mask_ratio
    masked_images = images * (~mask)  # マスク部分を0に
    return masked_images, mask


class ImageCausalModel(LightningModule):
    def __init__(self,cfg:DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.base_model = timm.create_model(
            cfg.pretrained_model, 
            pretrained = cfg.pretrained,
            num_classes = 0
            )
        self.base_model.fc = nn.Identity()

        self.Q_cls = nn.ModuleDict()

        input_size = self.base_model.num_features + self.cfg.num_labels

        for T in range(2):
            self.Q_cls['%d' % T] = nn.Sequential(
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, self.cfg.num_labels)
            )
        self.g_cls = nn.Linear(self.base_model.num_features + self.cfg.num_labels, self.cfg.num_labels)
        self.init_weights()
        self.Q0s = []
        self.Q1s = []
        self.g_losses = []
        self.Q_losses = []
        self.mask_losses = []
        self.losses = []
        self.total_training_steps = cfg.total_training_steps
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.base_model.num_features, 128, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # 出力を0-1の範囲に収める
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, batch, batch_idx):
        g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss ,masking_loss = self.__share_step(batch, batch_idx)
        return g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss, masking_loss
    
    def training_step(self,batch,batch_idx):
        self.train()
        g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss, masking_loss  = self.forward(batch, batch_idx)
        loss = (self.cfg.loss_weights.g * g_loss + 
            self.cfg.loss_weights.Q * Q_loss + 
            (self.cfg.loss_weights.masking * masking_loss if self.cfg.use_mask_loss else 0.0))
        
        self.g_losses.append(g_loss)
        self.Q_losses.append(Q_loss)
        if self.cfg.use_mask_loss:
            self.mask_losses.append(masking_loss)
        
        self.log_dict({"g_loss": g_loss, 
                  "Q_loss": Q_loss, 
                  "masking_loss": masking_loss, 
                  "train_loss": loss}
                )
        self.losses.append(loss)

        return loss
    
    def on_train_epoch_end(self):
        avg_g_loss = torch.stack(self.g_losses).mean()
        avg_Q_loss = torch.stack(self.Q_losses).mean()
        avg_masking_loss = torch.stack(self.mask_losses).mean() if self.mask_losses else 0.0
        avg_loss = torch.stack(self.losses).mean()

        self.log_dict({
            "train_epoch_loss": avg_loss,
            "g_losses": avg_g_loss,
            "Q_losses": avg_Q_loss,
            "masking_losses": avg_masking_loss
        })
        self.losses.clear()
        self.g_losses.clear()
        self.Q_losses.clear()
        self.mask_losses.clear()
        return 
    """
    def validation_step(self,batch,batch_idx):
        g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss = self.forward(batch, batch_idx)
        self.Q0s += Q_prob_T0.detach().cpu().numpy().tolist()
        self.Q1s += Q_prob_T1.detach().cpu().numpy().tolist()
        loss = (self.cfg.loss_weights["g"] * g_loss + 
                self.cfg.loss_weights["Q"] * Q_loss)
        self.log("val_loss" , loss)
        return loss
    
    def on_validation_epoch_end(self):
        probs = np.array(list(zip(self.Q0s, self.Q1s)))
        preds = np.argmax(probs,axis = 1)

        return probs,preds
    """
    def predict_step(self,batch,batch_idx):
        self.eval()
        g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss,mask_loss = self.forward(batch, batch_idx)
        Q0s = Q_prob_T0.detach().cpu().numpy().tolist()
        Q1s = Q_prob_T1.detach().cpu().numpy().tolist()
        self.Q0s += Q0s
        self.Q1s += Q1s
        return 
    
    def on_predict_epoch_end(self):
        print("on_predict_epoch_end")
        probs = np.array(list(zip(self.Q0s, self.Q1s)))
        preds = np.argmax(probs,axis = 1)
        self.ate_value = self.ATE(probs)
        self.logger.experiment.log({"ATE": self.ate_value})
        return {"probs": probs, "preds": preds}
    
    def ATE(self,probs):
        ## [todo] ATEの計算方法
        Q0 = probs[:,0]
        Q1 = probs[:,1]
        return np.mean(Q0 - Q1)



    
    def __share_step(self, batch, batch_idx):
        images, confounds, treatment,outcome = batch
         # 画像にマスクを適用
        if self.cfg.use_mask_loss:
            masked_images, mask = random_mask(images, mask_ratio=self.cfg.mask_ratio)
            features = self.base_model(masked_images)
            # 特徴をデコーダーに通して元画像サイズに再構成
            reconstructed_images = self.decoder(features.unsqueeze(-1).unsqueeze(-1))
            
            # 必要に応じてリサイズ
            if reconstructed_images.shape[2:] != images.shape[2:]:
                reconstructed_images = F.interpolate(reconstructed_images, size=images.shape[2:])
            
            # MSE損失で再構成の精度を計算
            masking_loss = F.mse_loss(reconstructed_images, images)
        else:
            features = self.base_model(images)
            masking_loss = 0.0
    
        C = self._make_confound_vector(confounds.unsqueeze(1), self.cfg.num_labels)
        inputs = torch.cat((features, C), dim = 1)
        g = self.g_cls(inputs)
        if torch.all(outcome != -1):
            g_loss = CrossEntropyLoss()(g.view(-1, self.cfg.num_labels),treatment.view(-1))
        else:
            g_loss = 0.0

        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)
        if torch.all(outcome != -1):
            T0_indices = (treatment == 0).nonzero().squeeze()
            Y_T1_labels = outcome.clone().scatter(0,T0_indices, -100)

            T1_indices = (treatment == 1).nonzero().squeeze()
            Y_T0_labels = outcome.clone().scatter(0,T1_indices, -100)
            Q_loss_T1 = CrossEntropyLoss()(Q_logits_T1.view(-1,self.cfg.num_labels), Y_T1_labels)
            Q_loss_T0 = CrossEntropyLoss()(Q_logits_T0.view(-1, self.cfg.num_labels), Y_T0_labels)

            Q_loss = Q_loss_T1 + Q_loss_T0
        else:
            Q_loss = 0.0

        sm = torch.nn.Softmax(dim = 1)
        Q_prob_T0 = sm(Q_logits_T0)[:,1]
        Q_prob_T1 = sm(Q_logits_T1)[:,1]
        g_prob = sm(g)[:,1]
    

        return g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss,masking_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.cfg.learning_rate, eps = 1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0.1 * self.total_training_steps,
            num_training_steps = self.total_training_steps
        )
        return [optimizer],[{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        
    def _make_confound_vector(self,ids, vocab_size, use_counts = False):
        vec = torch.zeros(ids.shape[0],vocab_size)
        ones = torch.ones_like(ids,dtype = torch.float)
        
        if self.cfg.CUDA:
            vec = vec.cuda()
            ones = ones.cuda()
            ids = ids.cuda()
        vec[:,1] = 0.0
        if not use_counts:
            vec = (vec != 0).float()
        return vec.float()