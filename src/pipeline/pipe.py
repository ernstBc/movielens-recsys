from src.train.trainer import Trainer, ConfigManager
from src.fine_tuning.tuner import Tuner
from src.prediction.predict_items import PredictUserAllItems
from typing import Optional


class Pipeline:
    def __init__(self, 
                 model_type:str,
                 dataset_type:str,
                 data_dir_config_path:str,
                 dataset_config_path:str,
                 model_config_path:str,
                 hyperparams_config_path:str,
                 finetuning_config_path:str|None=None,
                 ):
        self.model_type = model_type
        self.dataset_type = dataset_type
        self.data_dir_config = ConfigManager(data_dir_config_path)
        self.dataset_config  = ConfigManager(dataset_config_path)
        self.model_config = ConfigManager(model_config_path)
        self.hyperparams_config = ConfigManager(hyperparams_config_path)

        if finetuning_config_path is not None:
            self.finetuning_config = ConfigManager(finetuning_config_path)


    def run_pipeline(self,
                     model_kwargs:dict={}, 
                     hyperparams_kwargs:dict={},
                     dataset_kwargs:dict={},
                     data_dir_kwargs:dict={},
                     finetuning_kwargs:dict={}):

        # set configurations
        model_config = self.model_config(**model_kwargs)
        hyperparams_config = self.hyperparams_config(**hyperparams_kwargs)
        dataset_config = self.dataset_config(**dataset_kwargs)
        data_dir_config = self.data_dir_config(**data_dir_kwargs)
        if self.finetuning_config is not None:
            finetuning_config = self.finetuning_config(**finetuning_kwargs)

        # set pipeline components
        # TRAINER Handles both data pipeline (download and process) and model training

        # Finetune search the best set of hyperparams for a given type of model

        # Predictor predicts the items and saves the items with the higher values into a csv





        