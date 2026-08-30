from typing import Any

import torch
import pytorch_lightning as pl
from src.utils.utils import read_yaml
from src.models.autoencoders import AutoEncoder, DeepAutoEncoder
from src.models.matrix_factorization import MatrixFactorization, DeepMatrixFactorization
from src.models.wrappers import AutoencoderWrapper, MatrixFactorizationWrapper


class TrainerModelConfig:
    def __init__(self, model_type:str, config_path:str):
        self.config = read_yaml(config_path)[model_type.upper() + "_CONFIG"]


    def __call__(self, **kwds: Any) -> dict:
        new_config = self.config
        for nk, nv in kwds:
            new_config[nk] = nv

        return new_config


class HyperparamsConfig:
    def __init__(self, config_path:str):
        self.config = read_yaml(config_path)['HYPERPARAMS_CONFIG']


    def __call__(self, **kwds: Any) -> dict:
        new_config = self.config
        for nk, nv in kwds:
            new_config[nk] = nv

        return new_config
        

class Trainer:
    def __init__(self, 
                 model_type:str, 
                 model_config:TrainerModelConfig, 
                 model_extra_config:dict, 
                 hyperparams_config:HyperparamsConfig, 
                 hyperparams_extra_config:dict):
        
        self.model_config = model_config(**model_extra_config)
        self.hyperparams_config = hyperparams_config(**hyperparams_extra_config)
        self.model_type = model_type 
        self.model = None
        self.dataloader_module = None


    def train(self, max_epochs:int):
        if self.model is None:
            self.build_model()
        
        trainer = pl.Trainer(max_epochs=max_epochs)
        trainer.fit(self.model, self.dataloader_module)

        return self.model


    def build_model(self):
        model = self._set_model_and_dataloader()
        self.model = model


    def build_dataloader(self):
        self.dataloader_module = None
        pass


    def _set_model_and_dataloader(self):
        if self.model_type == 'autoencoder':
            model = AutoEncoder(**self.model_config)
            model_pl = AutoencoderWrapper(model, **self.hyperparams_config)
        elif self.model_type =='deep_autoencoder':
            model = DeepAutoEncoder(**self.model_config)
            model_pl = AutoencoderWrapper(model, **self.hyperparams_config)
        elif self.model_type =='matrix_factorization':
            model = MatrixFactorization(**self.model_config)
            model_pl = MatrixFactorizationWrapper(model, **self.hyperparams_config)
        elif self.model_type == 'deep_matrix_factorization':
            model = DeepMatrixFactorization(**self.model_config)
            model_pl = MatrixFactorizationWrapper(model, **self.hyperparams_config)
        else:
            raise ValueError("Invalid model type")
        
        return model_pl