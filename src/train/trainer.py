from typing import Any, Literal
import torch
import pytorch_lightning as pl
from src.utils.utils import read_yaml
from src.models.autoencoders import AutoEncoder, DeepAutoEncoder
from src.models.matrix_factorization import MatrixFactorization, DeepMatrixFactorization
from src.models.wrappers import AutoencoderWrapper, MatrixFactorizationWrapper
from src.data.wrappers import AutoencoderSampling, UserItemDataSampling


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

class DatasetConfig:
    def __init__(self, config_path:str):
        self.config = read_yaml(config_path)['DATASET_CONFIG']


    def __call__(self, **kwds: Any) -> dict:
        new_config = self.config
        for nk, nv in kwds:
            new_config[nk] = nv

        return new_config



class Trainer:
    def __init__(self, 
                 model_type:str, 
                 dataset_type:Literal['autoencoder', 'user_item', 'negative_sampling'],
                 dataset_size:Literal['100k', '10m'],
                 model_config:TrainerModelConfig, 
                 model_extra_config:dict, 
                 hyperparams_config:HyperparamsConfig, 
                 hyperparams_extra_config:dict,
                 dataset_config:DatasetConfig,
                 extra_dataset_config:dict):

        self.model_type = model_type 
        self.dataset_type = dataset_type
        self.model_config = model_config(**model_extra_config)
        self.hyperparams_config = hyperparams_config(**hyperparams_extra_config)
        self.dataset_config = dataset_config(**extra_dataset_config)
        self.dataset_size = dataset_size
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



    def _get_model(self):
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

    def _get_dataloader(self):
        if self.dataset_type == 'autoencoder':
            dataloader = AutoencoderSampling(self.dataset_config, self.dataset_size)
        elif self.dataset_type == 'user_item':
            dataloader = UserItemDataSampling(self.dataset_config, split_mode=self.dataset_config['SPLIT_MODE'], data_size=self.dataset_size, negative_sampling=False)
        elif self.dataset_type == 'negative_sampling':
            dataloader = UserItemDataSampling(self.dataset_config, split_mode=self.dataset_config['SPLIT_MODE'], data_size=self.dataset_size, negative_sampling=True)
        else:
            raise ValueError("Invalid dataset type")
        
        return dataloader