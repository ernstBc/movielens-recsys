import torch
import pytorch_lightning as pl
from typing import Any, Literal
from src.utils.utils import read_yaml
from src.models.autoencoders import AutoEncoder, DeepAutoEncoder
from src.models.matrix_factorization import MatrixFactorization, DeepMatrixFactorization
from src.models.wrappers import AutoencoderWrapper, MatrixFactorizationWrapper
from src.data.wrappers import AutoencoderSampling, UserItemDataSampling
import logging


class ConfigManager:
    def __init__(self, config_path:str, config_settings:str|None=None):
        config = read_yaml(config_path)
        if config_settings is not None:
            config = config[config_settings]
        self.config= config


    def __call__(self, **kwds: Any) -> dict:
        new_config = self.config.copy()
        for nk, nv in kwds.items():
            new_config[nk] = nv

        return new_config


class Trainer:
    def __init__(self, 
                 model_type:str,
                 dataset_type:Literal['autoencoder', 'user_item', 'negative_sampling'],
                 model_config:ConfigManager, 
                 hyperparams_config:ConfigManager, 
                 data_dir_config: ConfigManager,
                 dataset_config:ConfigManager,
                 model_extra_config:dict={}, 
                 hyperparams_extra_config:dict={},
                 data_dir_extra_config:dict={},
                 dataset_extra_config:dict={}):

        self.model_type = model_type 
        self.dataset_type = dataset_type
        self.model_config = model_config(**model_extra_config)
        self.hyperparams_config = hyperparams_config(**hyperparams_extra_config)
        self.data_dir_config = data_dir_config(**data_dir_extra_config)
        self.dataset_config = dataset_config(**dataset_extra_config)
        self.model = None
        self.dataloader_module = None


    def train(self, max_epochs:int, save_model=True, verbose:bool=True):
        trainer = self._get_trainer(max_epochs, save_model, verbose)
            
        trainer.fit(self.model, self.dataloader_module)

        self.trainer = trainer


    def setup(self):
        self.model, self.dataloader_module = self._get_model_and_dataloader()


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
        dataset_type = self.dataset_type
        
        if dataset_type == 'autoencoder':
            dataloader = AutoencoderSampling(data_dir_config=self.data_dir_config, dataset_config=self.dataset_config)
        elif dataset_type == 'user_item':
            dataloader = UserItemDataSampling(data_dir_config=self.data_dir_config, dataset_config=self.dataset_config)
        elif dataset_type == 'negative_sampling':
            dataloader = UserItemDataSampling(data_dir_config=self.data_dir_config, dataset_config=self.dataset_config)
        else:
            raise ValueError("Invalid dataset type")
        
        return dataloader


    def _get_model_and_dataloader(self):
        model = self._get_model()
        dataloader = self._get_dataloader()

        return model, dataloader


    def evaluate(self, verbose=True):
        if self.trainer is None:
            self.trainer = pl.Trainer()
        
        results = self.trainer.test(self.model, self.dataloader_module, verbose=verbose)
        return results


    def save_model(self, save_path:str):
        if self.model is None:
            raise ValueError("Model has not been initialized. Call setup() or train() first.")
        
        torch.save(self.model.state_dict(), save_path)


    def reset_params(self, config_settings:str, new_params:dict):
        if config_settings == 'model_config':
            self.model_config.update(new_params)
        elif config_settings == 'hyperparams_config':
            self.hyperparams_config.update(new_params)
        elif config_settings == 'dataset_config':
            self.dataset_config.update(new_params)
        else:
            raise ValueError(f"{config_settings} not supported")


    def _get_trainer(self, max_epochs:int, save_model:bool, verbose:bool):
        if self.trainer is None:
            if self.model is None:
                print('Initializing Model')
                self.setup()

            if verbose:
                trainer = pl.Trainer(max_epochs=max_epochs, enable_checkpointing=save_model)
                # Silence console warnings and info logs from Lightning
                #logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
                logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

            else:
                trainer = pl.Trainer(
                            max_epochs=max_epochs, 
                            enable_checkpointing=save_model,     
                            enable_progress_bar=False,
                            enable_model_summary=False)
        else:
            trainer = self.trainer

        return trainer