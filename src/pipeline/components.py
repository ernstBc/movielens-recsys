import os
from typing import Literal
from src.train.trainer import ConfigManager
from src.data.wrappers import UserItemDataSampling, AutoencoderSampling
from src.train.trainer import Trainer
from src.fine_tuning.tuner import Tuner
from src.prediction.predict_items import PredictAutoencoderItems, PredictUserAllItems
from src.models.wrappers import AutoencoderWrapper, MatrixFactorizationWrapper




class DataComponent:
    def __init__(self, data_config:ConfigManager, dataset_config:ConfigManager):
        self.data_config = data_config
        self.dataset_config=dataset_config


    def get_component(self, dataset_type:str, dataset_size:str, process_data:bool=True, force_process:bool=True, data_kwargs:dict={}, dataset_kwargs:dict={}) -> AutoencoderSampling|UserItemDataSampling:
        if dataset_type.lower() == 'autoencoder':
            args = self.data_config(**data_kwargs) | self.dataset_config(**dataset_kwargs)

            dsize_url = 'SMALL_URL' if dataset_size.lower() == '100k' else 'FULL_URL'
            dl_args = {
                 'dataset_url': args['DATA_URL'][dsize_url],
                 'dataset_path': args['DATA_DIR'][dataset_size.upper()]['RAW'],
                 'splits':args['DATA_DIR'][dataset_size.upper()]['SPLITS'],
                 'dataset_name':args['DATA_DIR'][dataset_size.upper()]['NAME'], 
                 'split_mode':args['split_mode'], 
                 'batch_size': args['batch_size'], 
                 'num_workers': args['num_workers'],
                 'process_data':process_data
            }

            component = AutoencoderSampling(**dl_args)
        else:
            args = self.data_config(**data_kwargs) | self.dataset_config(**dataset_kwargs)

            dsize_url = 'SMALL_URL' if dataset_size.lower() == '100k' else 'FULL_URL'
            dl_args = {
                 'dataset_url': args['DATA_URL'][dsize_url],
                 'dataset_path': args['DATA_DIR'][dataset_size.upper()]['RAW'],
                 'splits':args['DATA_DIR'][dataset_size.upper()]['SPLITS'],
                 'dataset_name':args['DATA_DIR'][dataset_size.upper()]['NAME'], 
                 'train_dataset_path': args['DATA_DIR'][dataset_size.upper()]['TRAIN'],
                 'validation_dataset_path': args['DATA_DIR'][dataset_size.upper()]['EVAL'],
                 'test_dataset_path': args['DATA_DIR'][dataset_size.upper()]['TEST'],

                 'split_mode':args['split_mode'], 
                 'negative_sampling':args['negative_sampling'], 
                 'testing': args['testing'],
                 'batch_size': args['batch_size'], 
                 'num_workers': args['num_workers'],
                 'process_data':process_data,
                 'force_process': force_process
            }

            component = UserItemDataSampling(**dl_args)

        return component



class TrainerComponent:
    def __init__(self, model_config:ConfigManager, 
                 hyperparams_config:ConfigManager,
                 trainer_config:ConfigManager):
        self.trainer_config = trainer_config
        self.model_config = model_config
        self.hyperparams_config = hyperparams_config



    def get_component(self, model_type:str, trainer_kwargs:dict, model_kwargs:dict={}, hyperparams_config:dict={}) -> Trainer:
        model_args = self.model_config(**model_kwargs)
        hyperparams_args = self.hyperparams_config(**hyperparams_config)
        trainer_args = self.trainer_config(**trainer_kwargs)
        trainer_kargs = {
                 'model_type':model_type,
                 'max_epochs':trainer_args['max_epochs'],
                 'model_kwargs':model_args, 
                 'hyperparams_kwargs':hyperparams_args, 
                 'verbose':trainer_args['verbose'], 
                 'sanity_check_steps':trainer_args['sanity_check_steps'], 
                 'profiler':trainer_args['profiler'],
                 'save_model_path':None,
                 'save_intermediate_ckpts':True,
                 'from_checkpoint':None
        } 
        trainer = Trainer(**trainer_kargs)
        return trainer


class FineTuningComponent:
    def __init__(self, 
                 model_config:ConfigManager, 
                 hyperparams_config:ConfigManager, 
                 finetuning_config:ConfigManager, 
                 trainer_config:ConfigManager,
                 ):
        self.model_config = model_config
        self.hyperparams_config = hyperparams_config
        self.finetuning_config = finetuning_config
        self.trainer_config = trainer_config


    def get_component(self, 
                      study_name:str, 
                      n_trials:int, 
                      model_type:str,  
                      dataset_type:Literal['autoencoder', 'user_item', 'negative_sampling'],
                      storage:str|None,
                      trainer_kwargs:dict={}) -> Tuner:
        tuner = Tuner(
            study_name=study_name,
            n_trials=n_trials,
            model_type=model_type,
            dataset_type=dataset_type,
            fine_tuning_config=self.finetuning_config(),
            model_kwargs=self.model_config(),
            hyperparams_kwargs=self.hyperparams_config(),
            trainer_kwargs=self.trainer_config(**trainer_kwargs),
            storage=storage
        )

        return tuner



class PredictionComponent:
    def __init__(self, data_config:ConfigManager) -> None:
        self.data_config = data_config


    def get_component(self, dataset_type:str, dataset_size:str, padding:bool=True) -> PredictUserAllItems| PredictAutoencoderItems:
        if dataset_type.lower() == 'autoencoder':
            data_path = os.path.join(
                            self.data_config()['DATA_DIR'][dataset_size.upper()]['RAW'],
                            'ratings.csv')
            movies_df_path = self.data_config()['DATA_DIR'][dataset_size.upper()]['MOVIES_PATH']
            predictor = PredictAutoencoderItems(data_path=data_path, movies_df_path=movies_df_path)

        else:
            n_movies = self.data_config()['DATA_DIR'][dataset_size.upper()]['N_MOVIES']
            if padding:
                n_movies +=1

            movies_encoder = self.data_config()['ARTIFACTS']['ENCODER_PATH']
            movies_df_path = self.data_config()['DATA_DIR'][dataset_size.upper()]['MOVIES_PATH']

            predictor = PredictUserAllItems(n_movies=n_movies, 
                                            movies_encoder=movies_encoder,
                                            movies_df_path=movies_df_path)
        return predictor