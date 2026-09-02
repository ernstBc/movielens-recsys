from typing import Any
import torch
import pytorch_lightning as pl
from src.train.customs import MaskedMSELoss
from torch import optim
from torch import nn


class AutoencoderWrapper(pl.LightningModule):
    def __init__(self, model, optimizer_name:str, lr:float, weight_decay:float):
        super().__init__()
        self.model = model
        self.loss_fn = MaskedMSELoss()
        self.config = {'lr': lr, 'weight_decay': weight_decay, 'optimizer_name': optimizer_name}


    def forward(self, user_id, item_id):
        return self.model(user_id, item_id)


    def training_step(self, batch, batch_idx):
        x= batch
        x_hat = self.model(batch)
        loss = self.loss_fn(x_hat, x)
        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x = batch
        x_hat = self.model(batch)
        loss = self.loss_fn(x_hat, x)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        x= batch
        x_hat = self.model(batch)
        loss = self.loss_fn(x_hat, x)
        self.log('test_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        return optimizer


    def _get_optimizer(self):
        optimizer_name = self.config['optimizer_name']
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'nadam':
            optimizer = optim.NAdam(self.parameters(), lr=lr, weight_decay=weight_decay)

        else:
            raise ValueError(f"{optimizer_name} is not a supported optimizer")

        return optimizer



class MatrixFactorizationWrapper(pl.LightningModule):
    def __init__(self, model, negative_sampling:bool, optimizer_name:str, lr:float, weight_decay:float):
        super().__init__()
        self.model = model
        self.negative_sampling = negative_sampling
        self.config = {'lr': lr, 'weight_decay': weight_decay, 'optimizer_name': optimizer_name}

        if negative_sampling:
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = MaskedMSELoss()


    def forward(self, batch):
        user_indices, item_indices = batch
        return self.model(user_indices, item_indices)


    def training_step(self, batch, batch_idx):
        user_indices, item_indices, ratings = batch
        ratings_hat = self.model(user_indices, item_indices)

        loss = self.loss_fn(ratings_hat, ratings)

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        user_indices, item_indices, ratings = batch
        ratings_hat = self.model(user_indices, item_indices)
        loss = self.loss_fn(ratings_hat, ratings)

        self.log('val_loss', loss,on_epoch=True, prog_bar=True, logger=True)
        return loss


    def test_step(self, batch, batch_idx):
        user_indices, item_indices, ratings = batch
        ratings_hat = self.model(user_indices, item_indices)
        loss = self.loss_fn(ratings_hat, ratings)

        self.log('test_loss', loss, on_step=True, prog_bar=True, logger=True)
        return loss


    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        return optimizer


    def _get_optimizer(self):
        optimizer_name = self.config['optimizer_name']
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']

        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'nadam':
            optimizer = optim.NAdam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"{optimizer_name} is not a supported optimizer")

        return optimizer