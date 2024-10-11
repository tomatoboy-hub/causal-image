from typing import Optional
import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import timm 
from torch.optim import AdamW
from omegaconf import DictConfig

from transformers import get_linear_schedule_with_warmup


class ImageCausalModel(LightningModule):
    def __init__(self,cfg:DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.base_model = timm.create_model(
            cfg.pretrained_model, 
            pretrained = cfg.pretrained,
            num_classes = cfg.num_classes
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
        self.g_cls = nn.Linear(input_size, self.cfg.num_labels)

        self.Q0s = []
        self.Q1s = []

        self.total_training_steps = cfg.total_training_steps

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, images):
        features = self.base_model(images)
        return features
    
    def training_step(self,batch,batch_idx):
        g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss = self.__share_step(batch, batch_idx)
        loss = (self.cfg.loss_weights["g"] * g_loss + 
                self.cfg.loss_weights["Q"] * Q_loss)
        self.log("train_loss" , loss)
        return loss
    
    def validation_step(self,batch,batch_idx):
        g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss = self.__share_step(batch, batch_idx)
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

    
    def __share_step(self, batch, batch_idx):
        images, confounds, treatment,outcome = batch
        features = self.forward(images)
        C = self._make_confound_vector(confounds.unsqueeze(1), self.cfg.num_labels)
        inputs = torch.cat((features, C), dim = 1)
        g = self.g_cls(inputs)
        if outcome is not None:
            g_loss = CrossEntropyLoss()(g.view(-1, self.num_labels),treatment.view(-1))
        else:
            g_loss = 0.0

        Q_logits_T0 = self.Q_cls['0'](inputs)
        Q_logits_T1 = self.Q_cls['1'](inputs)

        if outcome is not None:
            T0_indices = (treatment == 0).nonzero().squeeze()
            Y_T1_labels = outcome.clone().scatter(0,T0_indices, -100)

            T1_indices = (treatment == 1).nonzero().squeeze()
            Y_T0_labels = outcome.clone().scatter(0,T1_indices, -100)
            Q_loss_T1 = CrossEntropyLoss()(Q_logits_T1.view(-1,self.num_labels), Y_T1_labels)
            Q_loss_T0 = CrossEntropyLoss()(Q_logits_T0.view(-1, self.num_labels), Y_T0_labels)
            Q_loss = Q_loss_T1 + Q_loss_T0
        else:
            Q_loss = 0.0

        sm = torch.nn.Softmax(dim = 1)
        Q_prob_T0 = sm(Q_logits_T0)[:,1]
        Q_prob_T1 = sm(Q_logits_T1)[:,1]
        g_prob = sm(g)[:,1]

        return g_prob, Q_prob_T0, Q_prob_T1, g_loss, Q_loss
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr = self.cfg.learning_rate, eps = 1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = int(0.1 * self.total_training_steps),
            num_training_steps = self.total_training_steps
        )
        return [optimizer], [scheduler]
        
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